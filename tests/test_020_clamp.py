import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.tensordef import clamp
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的clamp函数")
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
    
    # 转换为numpy数组进行比较
    if hasattr(rm_result, 'data'):
        rm_data = rm_result.data
    else:
        rm_data = rm_result
    
    if hasattr(torch_result, 'numpy'):
        torch_data = torch_result.numpy()
    else:
        torch_data = torch_result
    
    # 处理形状不匹配的情况
    try:
        # 增加容差参数
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

class TestClampFunctions(unittest.TestCase):
    def setUp(self):
        # 设置随机种子以确保结果可重复
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch.set_default_dtype(torch.float32)
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
    
    def test_clamp_basic(self):
        """测试基本的clamp功能（同时设置min和max）"""
        case_name = "基本clamp功能"
        grad_case_name = "基本clamp功能 - 梯度跟踪"
        start_time = time.time()
        try:
            # 创建测试数据
            x_data = np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32)
            min_val = 2.0
            max_val = 8.0
            
            # 使用riemann
            rm_x = rm.tensor(x_data)
            rm_result = rm.clamp(rm_x, min_val, max_val)
            
            # 使用PyTorch
            if TORCH_AVAILABLE:
                t_x = torch.tensor(x_data)
                t_result = torch.clamp(t_x, min_val, max_val)
            else:
                t_result = None
            
            # 比较结果
            passed = compare_values(rm_result, t_result)
            passed_forward = passed
            
            # 梯度测试
            passed_grad = True
            if TORCH_AVAILABLE:
                # Riemann梯度计算
                rm_x_grad = rm.tensor(x_data, requires_grad=True)
                rm_result_grad = rm.clamp(rm_x_grad, min_val, max_val)
                rm_sum = rm.sum(rm_result_grad)
                rm_sum.backward()
                
                # PyTorch梯度计算
                t_x_grad = torch.tensor(x_data, requires_grad=True)
                t_result_grad = torch.clamp(t_x_grad, min_val, max_val)
                t_sum = torch.sum(t_result_grad)
                t_sum.backward()
                
                # 比较梯度
                passed_grad = compare_values(rm_x_grad.grad, t_x_grad.grad)
            
            passed = passed_forward and passed_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed_forward)
                stats.add_result(grad_case_name, passed_grad)
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed_forward else Colors.FAIL}{'通过' if passed_forward else '失败'}{Colors.ENDC}")
                print(f"测试用例: {grad_case_name} - {Colors.OKGREEN if passed_grad else Colors.FAIL}{'通过' if passed_grad else '失败'}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed_forward, f"基本clamp测试失败: {case_name}")
            self.assertTrue(passed_grad, f"基本clamp梯度测试失败: {grad_case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_clamp_only_min(self):
        """测试只设置min参数的情况"""
        case_name = "只设置min参数"
        grad_case_name = "只设置min参数 - 梯度跟踪"
        start_time = time.time()
        try:
            # 创建测试数据
            x_data = np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32)
            min_val = 2.0
            
            # 使用riemann
            rm_x = rm.tensor(x_data)
            rm_result = rm.clamp(rm_x, min=min_val)
            
            # 使用PyTorch
            if TORCH_AVAILABLE:
                t_x = torch.tensor(x_data)
                t_result = torch.clamp(t_x, min=min_val)
            else:
                t_result = None
            
            # 比较结果
            passed = compare_values(rm_result, t_result)
            passed_forward = passed
            
            # 梯度测试
            passed_grad = True
            if TORCH_AVAILABLE:
                # Riemann梯度计算
                rm_x_grad = rm.tensor(x_data, requires_grad=True)
                rm_result_grad = rm.clamp(rm_x_grad, min=min_val)
                rm_sum = rm.sum(rm_result_grad)
                rm_sum.backward()
                
                # PyTorch梯度计算
                t_x_grad = torch.tensor(x_data, requires_grad=True)
                t_result_grad = torch.clamp(t_x_grad, min=min_val)
                t_sum = torch.sum(t_result_grad)
                t_sum.backward()
                
                # 比较梯度
                passed_grad = compare_values(rm_x_grad.grad, t_x_grad.grad)
            
            passed = passed_forward and passed_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed_forward)
                stats.add_result(grad_case_name, passed_grad)
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed_forward else Colors.FAIL}{'通过' if passed_forward else '失败'}{Colors.ENDC}")
                print(f"测试用例: {grad_case_name} - {Colors.OKGREEN if passed_grad else Colors.FAIL}{'通过' if passed_grad else '失败'}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed_forward, f"只设置min参数测试失败: {case_name}")
            self.assertTrue(passed_grad, f"只设置min参数梯度测试失败: {grad_case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_clamp_only_max(self):
        """测试只设置max参数的情况"""
        case_name = "只设置max参数"
        grad_case_name = "只设置max参数 - 梯度跟踪"
        start_time = time.time()
        try:
            # 创建测试数据
            x_data = np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32)
            max_val = 6.0
            
            # 使用riemann
            rm_x = rm.tensor(x_data)
            rm_result = rm.clamp(rm_x, max=max_val)
            
            # 使用PyTorch
            if TORCH_AVAILABLE:
                t_x = torch.tensor(x_data)
                t_result = torch.clamp(t_x, max=max_val)
            else:
                t_result = None
            
            # 比较结果
            passed = compare_values(rm_result, t_result)
            passed_forward = passed
            
            # 梯度测试
            passed_grad = True
            if TORCH_AVAILABLE:
                # Riemann梯度计算
                rm_x_grad = rm.tensor(x_data, requires_grad=True)
                rm_result_grad = rm.clamp(rm_x_grad, max=max_val)
                rm_sum = rm.sum(rm_result_grad)
                rm_sum.backward()
                
                # PyTorch梯度计算
                t_x_grad = torch.tensor(x_data, requires_grad=True)
                t_result_grad = torch.clamp(t_x_grad, max=max_val)
                t_sum = torch.sum(t_result_grad)
                t_sum.backward()
                
                # 比较梯度
                passed_grad = compare_values(rm_x_grad.grad, t_x_grad.grad)
            
            passed = passed_forward and passed_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed_forward)
                stats.add_result(grad_case_name, passed_grad)
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed_forward else Colors.FAIL}{'通过' if passed_forward else '失败'}{Colors.ENDC}")
                print(f"测试用例: {grad_case_name} - {Colors.OKGREEN if passed_grad else Colors.FAIL}{'通过' if passed_grad else '失败'}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed_forward, f"只设置max参数测试失败: {case_name}")
            self.assertTrue(passed_grad, f"只设置max参数梯度测试失败: {grad_case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_clamp_edge_cases(self):
        """测试边界值情况"""
        case_name = "边界值情况"
        grad_case_name = "边界值情况 - 梯度跟踪"
        start_time = time.time()
        try:
            passed_cases = []
            grad_passed_cases = []
            
            # 测试1: 所有值都在范围内
            x_data1 = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
            min_val1, max_val1 = 2.0, 7.0
            
            rm_x1 = rm.tensor(x_data1)
            rm_result1 = rm.clamp(rm_x1, min_val1, max_val1)
            
            if TORCH_AVAILABLE:
                t_x1 = torch.tensor(x_data1)
                t_result1 = torch.clamp(t_x1, min_val1, max_val1)
            else:
                t_result1 = None
            
            passed1 = compare_values(rm_result1, t_result1)
            passed_cases.append(passed1)
            
            # 测试2: 所有值都小于min
            x_data2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            min_val2 = 5.0
            
            rm_x2 = rm.tensor(x_data2)
            rm_result2 = rm.clamp(rm_x2, min=min_val2)
            
            if TORCH_AVAILABLE:
                t_x2 = torch.tensor(x_data2)
                t_result2 = torch.clamp(t_x2, min=min_val2)
            else:
                t_result2 = None
            
            passed2 = compare_values(rm_result2, t_result2)
            passed_cases.append(passed2)
            
            # 测试3: 所有值都大于max
            x_data3 = np.array([[8.0, 9.0], [10.0, 11.0]], dtype=np.float32)
            max_val3 = 7.0
            
            rm_x3 = rm.tensor(x_data3)
            rm_result3 = rm.clamp(rm_x3, max=max_val3)
            
            if TORCH_AVAILABLE:
                t_x3 = torch.tensor(x_data3)
                t_result3 = torch.clamp(t_x3, max=max_val3)
            else:
                t_result3 = None
            
            passed3 = compare_values(rm_result3, t_result3)
            passed_cases.append(passed3)
            
            # 梯度测试
            if TORCH_AVAILABLE:
                # 测试所有值在范围内的梯度
                rm_x_grad1 = rm.tensor(x_data1, requires_grad=True)
                rm_result_grad1 = rm.clamp(rm_x_grad1, min_val1, max_val1)
                rm_sum1 = rm.sum(rm_result_grad1)
                rm_sum1.backward()
                
                t_x_grad1 = torch.tensor(x_data1, requires_grad=True)
                t_result_grad1 = torch.clamp(t_x_grad1, min_val1, max_val1)
                t_sum1 = torch.sum(t_result_grad1)
                t_sum1.backward()
                
                passed_grad1 = compare_values(rm_x_grad1.grad, t_x_grad1.grad)
                grad_passed_cases.append(passed_grad1)
                
                # 测试所有值小于min的梯度
                rm_x_grad2 = rm.tensor(x_data2, requires_grad=True)
                rm_result_grad2 = rm.clamp(rm_x_grad2, min=min_val2)
                rm_sum2 = rm.sum(rm_result_grad2)
                rm_sum2.backward()
                
                t_x_grad2 = torch.tensor(x_data2, requires_grad=True)
                t_result_grad2 = torch.clamp(t_x_grad2, min=min_val2)
                t_sum2 = torch.sum(t_result_grad2)
                t_sum2.backward()
                
                passed_grad2 = compare_values(rm_x_grad2.grad, t_x_grad2.grad)
                grad_passed_cases.append(passed_grad2)
                
                # 测试所有值大于max的梯度
                rm_x_grad3 = rm.tensor(x_data3, requires_grad=True)
                rm_result_grad3 = rm.clamp(rm_x_grad3, max=max_val3)
                rm_sum3 = rm.sum(rm_result_grad3)
                rm_sum3.backward()
                
                t_x_grad3 = torch.tensor(x_data3, requires_grad=True)
                t_result_grad3 = torch.clamp(t_x_grad3, max=max_val3)
                t_sum3 = torch.sum(t_result_grad3)
                t_sum3.backward()
                
                passed_grad3 = compare_values(rm_x_grad3.grad, t_x_grad3.grad)
                grad_passed_cases.append(passed_grad3)
            
            passed_forward = all(passed_cases)
            passed_grad = all(grad_passed_cases) if grad_passed_cases else True
            passed = passed_forward and passed_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(f"{case_name} - 所有值在范围内", passed1)
                stats.add_result(f"{case_name} - 所有值小于min", passed2)
                stats.add_result(f"{case_name} - 所有值大于max", passed3)
                if grad_passed_cases:
                    stats.add_result(f"{grad_case_name} - 所有值在范围内", passed_grad1)
                    stats.add_result(f"{grad_case_name} - 所有值小于min", passed_grad2)
                    stats.add_result(f"{grad_case_name} - 所有值大于max", passed_grad3)
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed_forward else Colors.FAIL}{'通过' if passed_forward else '失败'}{Colors.ENDC}")
                if grad_passed_cases:
                    print(f"测试用例: {grad_case_name} - {Colors.OKGREEN if passed_grad else Colors.FAIL}{'通过' if passed_grad else '失败'}{Colors.ENDC}")
                print(f" ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed_forward, f"边界值测试失败: {case_name}")
            self.assertTrue(passed_grad, f"边界值梯度测试失败: {grad_case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_clamp_scalar_input(self):
        """测试标量输入"""
        case_name = "标量输入"
        grad_case_name = "标量输入 - 梯度跟踪"
        start_time = time.time()
        try:
            # 创建标量测试数据
            x_scalar = np.array(5.0, dtype=np.float32)
            min_val = 2.0
            max_val = 8.0
            
            # 使用riemann
            rm_x = rm.tensor(x_scalar)
            rm_result = rm.clamp(rm_x, min_val, max_val)
            
            # 使用PyTorch
            if TORCH_AVAILABLE:
                t_x = torch.tensor(x_scalar)
                t_result = torch.clamp(t_x, min_val, max_val)
            else:
                t_result = None
            
            # 比较结果
            passed = compare_values(rm_result, t_result)
            passed_forward = passed
            
            # 梯度测试
            passed_grad = True
            if TORCH_AVAILABLE:
                # Riemann梯度计算
                rm_x_grad = rm.tensor(x_scalar, requires_grad=True)
                rm_result_grad = rm.clamp(rm_x_grad, min_val, max_val)
                rm_result_grad.backward()
                
                # PyTorch梯度计算
                t_x_grad = torch.tensor(x_scalar, requires_grad=True)
                t_result_grad = torch.clamp(t_x_grad, min_val, max_val)
                t_result_grad.backward()
                
                # 比较梯度
                passed_grad = compare_values(rm_x_grad.grad, t_x_grad.grad)
            
            passed = passed_forward and passed_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed_forward)
                stats.add_result(grad_case_name, passed_grad)
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed_forward else Colors.FAIL}{'通过' if passed_forward else '失败'}{Colors.ENDC}")
                print(f"测试用例: {grad_case_name} - {Colors.OKGREEN if passed_grad else Colors.FAIL}{'通过' if passed_grad else '失败'}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed_forward, f"标量输入测试失败: {case_name}")
            self.assertTrue(passed_grad, f"标量输入梯度测试失败: {grad_case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_clamp_different_shapes(self):
        """测试不同形状的输入"""
        case_name = "不同形状输入"
        grad_case_name = "不同形状输入 - 梯度跟踪"
        start_time = time.time()
        try:
            # 测试多维张量
            x_data_3d = np.random.randn(2, 3, 4).astype(np.float32)
            min_val = -1.0
            max_val = 1.0
            
            # 使用riemann
            rm_x_3d = rm.tensor(x_data_3d)
            rm_result_3d = rm.clamp(rm_x_3d, min_val, max_val)
            
            # 使用PyTorch
            if TORCH_AVAILABLE:
                t_x_3d = torch.tensor(x_data_3d)
                t_result_3d = torch.clamp(t_x_3d, min_val, max_val)
            else:
                t_result_3d = None
            
            # 比较结果
            passed_3d = compare_values(rm_result_3d, t_result_3d)
            passed_forward = passed_3d
            
            # 梯度测试
            passed_grad = True
            if TORCH_AVAILABLE:
                # Riemann梯度计算
                rm_x_grad_3d = rm.tensor(x_data_3d, requires_grad=True)
                rm_result_grad_3d = rm.clamp(rm_x_grad_3d, min_val, max_val)
                rm_sum_3d = rm.sum(rm_result_grad_3d)
                rm_sum_3d.backward()
                
                # PyTorch梯度计算
                t_x_grad_3d = torch.tensor(x_data_3d, requires_grad=True)
                t_result_grad_3d = torch.clamp(t_x_grad_3d, min_val, max_val)
                t_sum_3d = torch.sum(t_result_grad_3d)
                t_sum_3d.backward()
                
                # 比较梯度
                passed_grad = compare_values(rm_x_grad_3d.grad, t_x_grad_3d.grad)
            
            passed = passed_forward and passed_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed_forward)
                stats.add_result(grad_case_name, passed_grad)
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed_forward else Colors.FAIL}{'通过' if passed_forward else '失败'}{Colors.ENDC}")
                print(f"测试用例: {grad_case_name} - {Colors.OKGREEN if passed_grad else Colors.FAIL}{'通过' if passed_grad else '失败'}{Colors.ENDC} ({time_taken:.4f}秒)")
                print(f"  输入形状: {x_data_3d.shape}")
            
            # 断言确保测试通过
            self.assertTrue(passed_forward, f"不同形状输入测试失败: {case_name}")
            self.assertTrue(passed_grad, f"不同形状输入梯度测试失败: {grad_case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_clamp_out_parameter(self):
        """测试out参数"""
        case_name = "out参数测试"
        start_time = time.time()
        try:
            # 创建测试数据
            x_data = np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32)
            min_val = 2.0
            max_val = 8.0
            
            # 创建out参数张量
            out_shape = x_data.shape
            
            # 使用riemann
            rm_x = rm.tensor(x_data)
            rm_out = rm.zeros(out_shape)
            rm_result = rm.clamp(rm_x, min_val, max_val, out=rm_out)
            
            # 验证out参数是否被正确修改
            # 再次调用不使用out参数的版本进行比较
            rm_result_normal = rm.clamp(rm_x, min_val, max_val)
            passed_out = compare_values(rm_out, rm_result_normal)
            passed_forward = passed_out
            
            # 验证返回值是否是out参数
            passed_return = (rm_result is rm_out)
            passed_forward = passed_forward and passed_return
            
            # 使用PyTorch进行比较
            if TORCH_AVAILABLE:
                t_x = torch.tensor(x_data)
                t_out = torch.zeros(out_shape)
                t_result = torch.clamp(t_x, min_val, max_val, out=t_out)
                
                # 比较PyTorch的out结果和riemann的out结果
                passed_pytorch_compare = compare_values(rm_out, t_out)
                passed_forward = passed_forward and passed_pytorch_compare
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(f"{case_name} - out内容正确", passed_out)
                stats.add_result(f"{case_name} - 返回值是out", passed_return)
                if TORCH_AVAILABLE:
                    stats.add_result(f"{case_name} - 与PyTorch一致", passed_pytorch_compare)
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed_forward else Colors.FAIL}{'通过' if passed_forward else '失败'}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed_out, f"out参数内容不正确: {case_name}")
            self.assertTrue(passed_return, f"返回值不是out参数: {case_name}")
            if TORCH_AVAILABLE:
                self.assertTrue(passed_pytorch_compare, f"与PyTorch结果不一致: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_clamp_mixed_dtypes(self):
        """测试混合数据类型"""
        case_name = "混合数据类型"
        grad_case_name = "混合数据类型 - 梯度跟踪"
        start_time = time.time()
        try:
            # 创建不同数据类型的测试数据
            # 注意：整数类型通常不支持梯度，所以我们先测试前向计算
            x_data_int = np.array([[-3, 0, 3], [5, 7, 9]], dtype=np.int32)
            min_val = 2
            max_val = 8
            
            # 使用riemann
            rm_x_int = rm.tensor(x_data_int)
            rm_result_int = rm.clamp(rm_x_int, min_val, max_val)
            
            # 使用PyTorch
            if TORCH_AVAILABLE:
                t_x_int = torch.tensor(x_data_int)
                t_result_int = torch.clamp(t_x_int, min_val, max_val)
            else:
                t_result_int = None
            
            # 比较结果
            passed = compare_values(rm_result_int, t_result_int)
            passed_forward = passed
            
            # 梯度测试 - 使用浮点型数据进行梯度测试
            passed_grad = True
            if TORCH_AVAILABLE:
                x_data_float = x_data_int.astype(np.float32)
                
                # Riemann梯度计算
                rm_x_float = rm.tensor(x_data_float, requires_grad=True)
                rm_result_float = rm.clamp(rm_x_float, min_val, max_val)
                rm_sum = rm.sum(rm_result_float)
                rm_sum.backward()
                
                # PyTorch梯度计算
                t_x_float = torch.tensor(x_data_float, requires_grad=True)
                t_result_float = torch.clamp(t_x_float, min_val, max_val)
                t_sum = torch.sum(t_result_float)
                t_sum.backward()
                
                # 比较梯度
                passed_grad = compare_values(rm_x_float.grad, t_x_float.grad)
            
            passed = passed_forward and passed_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed_forward)
                stats.add_result(grad_case_name, passed_grad)
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed_forward else Colors.FAIL}{'通过' if passed_forward else '失败'}{Colors.ENDC}")
                print(f"测试用例: {grad_case_name} - {Colors.OKGREEN if passed_grad else Colors.FAIL}{'通过' if passed_grad else '失败'}{Colors.ENDC} ({time_taken:.4f}秒)")
                if passed_forward:
                    print(f"  结果数据类型: {rm_result_int.data.dtype}")
            
            # 断言确保测试通过
            self.assertTrue(passed_forward, f"混合数据类型测试失败: {case_name}")
            self.assertTrue(passed_grad, f"混合数据类型梯度测试失败: {grad_case_name}")
            
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
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行clamp函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestClampFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)