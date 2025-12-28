import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.autograd.functional import jacobian as rm_jacobian
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    from torch.autograd.functional import jacobian as torch_jacobian
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的jacobian函数")
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
    """比较Riemann和PyTorch的jacobian结果是否接近"""
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
        for i, (r, t) in enumerate(zip(rm_result, torch_result)):
            if not compare_values(r, t, atol, rtol):
                all_passed = False
                break
        
        return all_passed
    
    # 转换为numpy数组
    if hasattr(rm_result, 'data'):
        rm_data = rm_result.data
    else:
        rm_data = rm_result
    
    if hasattr(torch_result, 'detach'):
        torch_data = torch_result.detach().cpu().numpy()
    else:
        torch_data = torch_result
    
    # 处理形状不匹配的情况
    try:
        # 增加容差参数
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

class TestJacobianFunctions(unittest.TestCase):
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
            
    def _run_test_case(self, case_name, test_func):
        """运行测试用例的通用方法，减少重复代码"""
        start_time = time.time()
        try:
            # 执行测试函数
            passed = test_func()
            
            # 计算耗时
            time_taken = time.time() - start_time
            
            # 记录结果
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"Jacobian计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_single_input_single_output(self):
        """测试场景1: 单张量输入，单张量输出"""
        case_name = "单张量输入，单张量输出"
        
        def test_func():
            # 定义测试函数: f(x) = x^2
            def func(x):
                return x ** 2.
            
            # 定义对应的PyTorch函数
            def torch_func(x):
                return x ** 2.
            
            # 创建输入
            rm_x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            else:
                torch_x = None
            
            # 计算Jacobian矩阵
            rm_jac = rm_jacobian(func, rm_x)
            if TORCH_AVAILABLE:
                torch_jac = torch_jacobian(torch_func, torch_x)
            else:
                torch_jac = None
            
            # 比较结果
            return compare_values(rm_jac, torch_jac)
        
        self._run_test_case(case_name, test_func)
    
    def test_multi_input_single_output(self):
        """测试场景2: 多张量输入，单张量输出"""
        case_name = "多张量输入，单张量输出"
        start_time = time.time()
        try:
            # 定义测试函数: f(x, y) = x * y + y ** 2
            def func(x, y):
                return x * y + y ** 2.
            
            # 定义对应的PyTorch函数
            def torch_func(x, y):
                return x * y + y ** 2.
            
            # 创建输入
            rm_x = rm.tensor(2.0, requires_grad=True)
            rm_y = rm.tensor(3.0, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(2.0, requires_grad=True)
                torch_y = torch.tensor(3.0, requires_grad=True)
            else:
                torch_x, torch_y = None, None
            
            # 计算Jacobian矩阵
            rm_jac = rm_jacobian(func, (rm_x, rm_y))
            if TORCH_AVAILABLE:
                torch_jac = torch_jacobian(torch_func, (torch_x, torch_y))
            else:
                torch_jac = None
            
            # 比较结果
            passed = compare_values(rm_jac, torch_jac)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"Jacobian计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_single_input_multi_output(self):
        """测试场景3: 单张量输入，多张量输出"""
        case_name = "单张量输入，多张量输出"
        start_time = time.time()
        try:
            # 定义测试函数: 返回两个输出
            def func(x):
                output1 = x ** 2.  # 第一个输出
                output2 = rm.sin(x)  # 第二个输出
                return output1, output2  # 以元组形式返回多个输出
            
            # 定义对应的PyTorch函数
            def torch_func(x):
                output1 = x ** 2.
                output2 = torch.sin(x)
                return output1, output2
            
            # 创建输入
            rm_x = rm.tensor([1.0, 2.0], requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor([1.0, 2.0], requires_grad=True)
            else:
                torch_x = None
            
            # 计算Jacobian矩阵
            rm_jac = rm_jacobian(func, rm_x)
            if TORCH_AVAILABLE:
                torch_jac = torch_jacobian(torch_func, torch_x)
            else:
                torch_jac = None
            
            # 比较结果
            passed = compare_values(rm_jac, torch_jac)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"Jacobian计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_multi_input_multi_output(self):
        """测试场景4: 多张量输入，多张量输出"""
        case_name = "多张量输入，多张量输出"
        start_time = time.time()
        try:
            # 定义测试函数: 接受两个输入张量，返回两个输出张量
            def func(x, y):
                out1 = x ** 2. + y ** 3.  # 第一个输出
                out2 = x * y             # 第二个输出
                return out1, out2        # 返回一个包含两个输出张量的元组
            
            # 定义对应的PyTorch函数
            def torch_func(x, y):
                out1 = x ** 2. + y ** 3.
                out2 = x * y
                return out1, out2
            
            # 创建输入
            rm_x = rm.tensor([1.0, 2.0], requires_grad=True)
            rm_y = rm.tensor([3.0, 4.0], requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor([1.0, 2.0], requires_grad=True)
                torch_y = torch.tensor([3.0, 4.0], requires_grad=True)
            else:
                torch_x, torch_y = None, None
            
            # 计算Jacobian矩阵
            rm_jac = rm_jacobian(func, (rm_x, rm_y))
            if TORCH_AVAILABLE:
                torch_jac = torch_jacobian(torch_func, (torch_x, torch_y))
            else:
                torch_jac = None
            
            # 比较结果
            passed = compare_values(rm_jac, torch_jac)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"Jacobian计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_error_non_tensor_input(self):
        """测试场景5: 错误场景，输入不是张量或张量列表"""
        case_name = "错误场景，输入不是张量或张量列表"
        start_time = time.time()
        try:
            # 定义测试函数: f(x) = x^2
            def func(x):
                return x ** 2.
            
            # 尝试使用非张量输入，预期会抛出异常
            try:
                rm_jac = rm_jacobian(func, 1.0)  # 传入普通Python标量
                passed = False
                error_msg = "预期抛出异常但未抛出"
            except RuntimeError as e:
                if "requires grad" in str(e) or "tensor" in str(e).lower():
                    passed = True
                    error_msg = None
                else:
                    passed = False
                    error_msg = f"抛出了非预期的异常: {str(e)}"
            except Exception as e:
                passed = False
                error_msg = f"抛出了非预期的异常类型: {type(e).__name__}: {str(e)}"
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  错误: {error_msg}")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"异常处理测试失败: {error_msg}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_strict_parameter(self):
        """测试场景6: 测试strict参数的行为"""
        test_cases = [
            {"name": "strict=False的基本功能", "strict": False, "should_pass": True},
            {"name": "strict=True时函数忽略部分输入", "strict": True, "should_pass": True},
            {"name": "strict=True与所有输入都相关的情况", "strict": True, "should_pass": True}
        ]
        
        for case in test_cases:
            start_time = time.time()
            try:
                if case["name"] == "strict=False的基本功能":
                    # 定义简单的单输入测试函数
                    def simple_func(x):
                        return x ** 2.
                    
                    # 创建输入
                    rm_x = rm.tensor(2.0, requires_grad=True)
                    
                    # 测试strict=False时的行为
                    try:
                        rm_jac_non_strict = rm_jacobian(simple_func, rm_x, strict=False)
                        passed = True
                    except Exception as e:
                        passed = False
                
                elif case["name"] == "strict=True时函数忽略部分输入":
                    # 定义多输入测试函数: f(x, y) = x^2（忽略y参数）
                    def func_partial_dep(x, y):
                        return x ** 2.
                    
                    rm_x = rm.tensor(2.0, requires_grad=True)
                    rm_y = rm.tensor(3.0, requires_grad=True)
                    
                    try:
                        rm_jac_strict = rm_jacobian(func_partial_dep, (rm_x, rm_y), strict=True)
                        passed = False  # 应该抛出异常
                    except Exception as e:
                        # 检查异常消息是否包含"independent"或类似内容
                        if "independent" in str(e).lower():
                            passed = True
                        else:
                            passed = False
                
                elif case["name"] == "strict=True与所有输入都相关的情况":
                    # 定义测试函数: f(x, y) = x^2 + y^3 + x*y（依赖所有输入）
                    def func_full_dep(x, y):
                        return x ** 2. + y ** 3. + x * y
                    
                    # 创建新的输入
                    rm_x2 = rm.tensor(2.0, requires_grad=True)
                    rm_y2 = rm.tensor(3.0, requires_grad=True)
                    
                    try:
                        # 计算Jacobian
                        rm_jac_strict_full = rm_jacobian(func_full_dep, (rm_x2, rm_y2), strict=True)
                        passed = True
                    except Exception as e:
                        passed = False
                
                # 检查结果是否符合预期
                expected_result = case["should_pass"]
                actual_result = passed
                final_passed = (expected_result == actual_result)
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], final_passed)
                    status = "通过" if final_passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if final_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not final_passed:
                        print(f"  结果: 期望{'通过' if expected_result else '失败'}，实际{'通过' if actual_result else '失败'}")
                
                # 断言确保测试通过
                self.assertTrue(final_passed, f"strict参数测试失败: {case['name']}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], False, [str(e)])
                    print(f"测试用例: {case['name']} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_create_graph_enabled(self):
        """测试场景7: 测试create_graph=True的情况"""
        case_name = "测试create_graph=True的情况"
        start_time = time.time()
        try:
            # 定义测试函数: f(x) = x^3
            def func(x):
                return x ** 3.
            
            # 创建输入
            rm_x = rm.tensor(2.0, requires_grad=True)
            
            # 计算Jacobian矩阵，启用计算图
            rm_jac = rm_jacobian(func, rm_x, create_graph=True)
            
            # 验证结果是否可求导（requires_grad为True）
            passed = rm_jac.requires_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  错误: Jacobian结果的requires_grad不为True")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"create_graph=True测试失败: Jacobian结果不可求导")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
            
    def test_single_input_single_scalar(self):
        """测试场景8: 单张量输入，单标量输出"""
        case_name = "单张量输入，单标量输出"
        start_time = time.time()
        try:
            # 定义测试函数: f(x) = sum(x^2)
            def func(x):
                return rm.sum(x ** 2.)
            
            # 定义对应的PyTorch函数
            def torch_func(x):
                return torch.sum(x ** 2.)
            
            # 创建输入
            rm_x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            else:
                torch_x = None
            
            # 计算Jacobian矩阵
            rm_jac = rm_jacobian(func, rm_x)
            if TORCH_AVAILABLE:
                torch_jac = torch_jacobian(torch_func, torch_x)
            else:
                torch_jac = None
            
            # 比较结果
            passed = compare_values(rm_jac, torch_jac)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"Jacobian计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
            
    def test_multi_input_single_scalar(self):
        """测试场景9: 多张量输入，单标量输出"""
        case_name = "多张量输入，单标量输出"
        start_time = time.time()
        try:
            # 定义测试函数: f(x, y) = sum(x * y)
            def func(x, y):
                return rm.sum(x * y)
            
            # 定义对应的PyTorch函数
            def torch_func(x, y):
                return torch.sum(x * y)
            
            # 创建输入
            rm_x = rm.tensor([1.0, 2.0], requires_grad=True)
            rm_y = rm.tensor([3.0, 4.0], requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor([1.0, 2.0], requires_grad=True)
                torch_y = torch.tensor([3.0, 4.0], requires_grad=True)
            else:
                torch_x, torch_y = None, None
            
            # 计算Jacobian矩阵
            rm_jac = rm_jacobian(func, (rm_x, rm_y))
            if TORCH_AVAILABLE:
                torch_jac = torch_jacobian(torch_func, (torch_x, torch_y))
            else:
                torch_jac = None
            
            # 比较结果
            passed = compare_values(rm_jac, torch_jac)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"Jacobian计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
            
    def test_single_input_multi_scalar(self):
        """测试场景10: 单张量输入，多标量输出"""
        case_name = "单张量输入，多标量输出"
        start_time = time.time()
        try:
            # 定义测试函数: f(x) = (sum(x), sum(x^2))
            def func(x):
                return rm.sum(x), rm.sum(x ** 2.)
            
            # 定义对应的PyTorch函数
            def torch_func(x):
                return torch.sum(x), torch.sum(x ** 2.)
            
            # 创建输入
            rm_x = rm.tensor([1.0, 2.0], requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor([1.0, 2.0], requires_grad=True)
            else:
                torch_x = None
            
            # 计算Jacobian矩阵
            rm_jac = rm_jacobian(func, rm_x)
            if TORCH_AVAILABLE:
                torch_jac = torch_jacobian(torch_func, torch_x)
            else:
                torch_jac = None
            
            # 比较结果
            passed = compare_values(rm_jac, torch_jac)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"Jacobian计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
            
    def test_multi_input_multi_scalar(self):
        """测试场景11: 多张量输入，多标量输出"""
        case_name = "多张量输入，多标量输出"
        start_time = time.time()
        try:
            # 定义测试函数: f(x, y) = (sum(x * y), sum(x + y))
            def func(x, y):
                return rm.sum(x * y), rm.sum(x + y)
            
            # 定义对应的PyTorch函数
            def torch_func(x, y):
                return torch.sum(x * y), torch.sum(x + y)
            
            # 创建输入
            rm_x = rm.tensor([1.0, 2.0], requires_grad=True)
            rm_y = rm.tensor([3.0, 4.0], requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor([1.0, 2.0], requires_grad=True)
                torch_y = torch.tensor([3.0, 4.0], requires_grad=True)
            else:
                torch_x, torch_y = None, None
            
            # 计算Jacobian矩阵
            rm_jac = rm_jacobian(func, (rm_x, rm_y))
            if TORCH_AVAILABLE:
                torch_jac = torch_jacobian(torch_func, (torch_x, torch_y))
            else:
                torch_jac = None
            
            # 比较结果
            passed = compare_values(rm_jac, torch_jac)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"Jacobian计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
            
    def test_single_input_tensor_and_scalar(self):
        """测试场景12: 单张量输入，张量和标量混合输出"""
        case_name = "单张量输入，张量和标量混合输出"
        start_time = time.time()
        try:
            # 定义测试函数: f(x) = (x^2, sum(x))
            def func(x):
                return x ** 2., rm.sum(x)
            
            # 定义对应的PyTorch函数
            def torch_func(x):
                return x ** 2., torch.sum(x)
            
            # 创建输入
            rm_x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            else:
                torch_x = None
            
            # 计算Jacobian矩阵
            rm_jac = rm_jacobian(func, rm_x)
            if TORCH_AVAILABLE:
                torch_jac = torch_jacobian(torch_func, torch_x)
            else:
                torch_jac = None
            
            # 比较结果
            passed = compare_values(rm_jac, torch_jac)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"Jacobian计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
            
    def test_multi_input_tensor_and_scalar(self):
        """测试场景13: 多张量输入，张量和标量混合输出"""
        case_name = "多张量输入，张量和标量混合输出"
        start_time = time.time()
        try:
            # 定义测试函数: f(x, y) = (x * y, sum(x + y))
            def func(x, y):
                return x * y, rm.sum(x + y)
            
            # 定义对应的PyTorch函数
            def torch_func(x, y):
                return x * y, torch.sum(x + y)
            
            # 创建输入
            rm_x = rm.tensor([1.0, 2.0], requires_grad=True)
            rm_y = rm.tensor([3.0, 4.0], requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor([1.0, 2.0], requires_grad=True)
                torch_y = torch.tensor([3.0, 4.0], requires_grad=True)
            else:
                torch_x, torch_y = None, None
            
            # 计算Jacobian矩阵
            rm_jac = rm_jacobian(func, (rm_x, rm_y))
            if TORCH_AVAILABLE:
                torch_jac = torch_jacobian(torch_func, (torch_x, torch_y))
            else:
                torch_jac = None
            
            # 比较结果
            passed = compare_values(rm_jac, torch_jac)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"Jacobian计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
            
    def test_higher_order_derivatives(self):
        """测试场景14: 测试高阶导数计算"""
        case_name = "测试高阶导数计算"
        
        def test_func():
            # 定义测试函数: f(x) = x^3
            def func(x):
                return x ** 3.
            
            # 创建输入
            rm_x = rm.tensor([2.0], requires_grad=True)
            
            # 计算一阶导数（雅可比矩阵）
            rm_jac1 = rm_jacobian(func, rm_x, create_graph=True)
            
            # 验证一阶导数是否正确 (d/dx x^3 = 3x^2)
            expected_jac1 = 3. * rm_x ** 2.
            jac1_correct = np.allclose(rm_jac1.data, expected_jac1.data)
            
            # 尝试计算二阶导数
            # 定义一个函数，该函数计算一阶导数
            def first_derivative(x):
                # 注意：这里要返回雅可比矩阵的值而不是整个结构
                return rm_jacobian(func, x, create_graph=True)
            
            # 计算二阶导数
            rm_jac2 = rm_jacobian(first_derivative, rm_x)
            
            # 验证二阶导数是否正确 (d²/dx² x^3 = 6x)
            expected_jac2 = 6. * rm_x
            jac2_correct = np.allclose(rm_jac2.data, expected_jac2.data)
            
            # 组合测试结果
            passed = jac1_correct and jac2_correct
            
            # 如果有PyTorch，也进行比较
            if TORCH_AVAILABLE:
                try:
                    # PyTorch实现
                    torch_x = torch.tensor([2.0], requires_grad=True)
                    torch_func = lambda x: x ** 3.
                    
                    # 一阶导数
                    torch_jac1 = torch_jacobian(torch_func, torch_x, create_graph=True)
                    torch_first_derivative = lambda x: torch_jacobian(torch_func, x, create_graph=True)
                    
                    # 二阶导数
                    torch_jac2 = torch_jacobian(torch_first_derivative, torch_x)
                    
                    # 比较结果
                    rm_result = (rm_jac1, rm_jac2)
                    torch_result = (torch_jac1, torch_jac2)
                    
                    # 先进行数值比较
                    value_match = compare_values(rm_result, torch_result)
                    
                    # 综合结果
                    passed = passed and value_match
                except Exception as e:
                    # PyTorch比较出错不影响主要测试结果，但记录错误
                    print(f"PyTorch高阶导数比较出错: {str(e)}")
            
            # 保存到实例变量以便在断言中使用
            self._jac1_correct = jac1_correct
            self._jac2_correct = jac2_correct
            
            # 高阶导数测试的特殊错误信息
            if not passed and IS_RUNNING_AS_SCRIPT:
                print(f"  一阶导数正确={jac1_correct}, 二阶导数正确={jac2_correct}")
            
            return passed
        
        start_time = time.time()
        try:
            passed = test_func()
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 自定义断言错误信息
            # 重新获取正确的值进行断言
            rm_x = rm.tensor([2.0], requires_grad=True)
            func = lambda x: x ** 3.
            rm_jac1 = rm_jacobian(func, rm_x, create_graph=True)
            expected_jac1 = 3. * rm_x ** 2.
            jac1_correct = np.allclose(rm_jac1.data, expected_jac1.data)
            
            def first_derivative(x):
                return rm_jacobian(func, x, create_graph=True)
            
            rm_jac2 = rm_jacobian(first_derivative, rm_x)
            expected_jac2 = 6. * rm_x
            jac2_correct = np.allclose(rm_jac2.data, expected_jac2.data)
            
            self.assertTrue(passed, f"高阶导数测试失败: 一阶导数正确={jac1_correct}, 二阶导数正确={jac2_correct}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise

# 如果作为独立脚本运行
if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行Jacobian函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestJacobianFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)