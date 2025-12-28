import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.autograd import grad as rm_grad
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    from torch import autograd as torch_autograd
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的grad函数")
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

class TestGradFunctions(unittest.TestCase):
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
    
    def test_scalar_gradient(self):
        """测试场景1: 基本的标量函数梯度计算 (f(x) = x^2)"""
        case_name = "基本的标量函数梯度计算"
        start_time = time.time()
        try:
            # 创建输入
            x_np = 2.0
            rm_x = rm.tensor(x_np, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(x_np, requires_grad=True)
            else:
                torch_x = None
            
            # 计算梯度
            y_riemann = rm_x ** 2.
            grad_x_riemann = rm_grad(y_riemann, rm_x)[0]
            
            if TORCH_AVAILABLE:
                y_torch = torch_x ** 2.
                grad_x_torch = torch_autograd.grad(y_torch, torch_x)[0]
            else:
                grad_x_torch = None
            
            # 比较结果
            passed = compare_values(grad_x_riemann, grad_x_torch)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"梯度计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_multi_input_gradient(self):
        """测试场景2: 多输入的梯度计算 (f(x, y) = 2x + y^2)"""
        case_name = "多输入的梯度计算"
        start_time = time.time()
        try:
            # 创建输入
            x_np = 3.0
            y_np = 4.0
            rm_x = rm.tensor(x_np, requires_grad=True)
            rm_y = rm.tensor(y_np, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(x_np, requires_grad=True)
                torch_y = torch.tensor(y_np, requires_grad=True)
            else:
                torch_x, torch_y = None, None
            
            # 计算梯度
            z_riemann = 2. * rm_x + rm_y ** 2.
            grad_x_riemann, grad_y_riemann = rm_grad(z_riemann, [rm_x, rm_y])
            
            if TORCH_AVAILABLE:
                z_torch = 2. * torch_x + torch_y ** 2.
                grad_x_torch, grad_y_torch = torch_autograd.grad(z_torch, [torch_x, torch_y])
            else:
                grad_x_torch, grad_y_torch = None, None
            
            # 比较结果
            rm_result = (grad_x_riemann, grad_y_riemann)
            torch_result = (grad_x_torch, grad_y_torch)
            passed = compare_values(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"梯度计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_unused_variable_gradient(self):
        """测试场景3: 包含未使用变量的梯度计算"""
        case_name = "包含未使用变量的梯度计算"
        start_time = time.time()
        try:
            # 创建输入
            x_np = 3.0
            y_np = 4.0
            y1_np = 5.0
            rm_x = rm.tensor(x_np, requires_grad=True)
            rm_y = rm.tensor(y_np, requires_grad=True)
            rm_y1 = rm.tensor(y1_np, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(x_np, requires_grad=True)
                torch_y = torch.tensor(y_np, requires_grad=True)
                torch_y1 = torch.tensor(y1_np, requires_grad=True)
            else:
                torch_x, torch_y, torch_y1 = None, None, None
            
            # 计算梯度
            z_riemann = 2. * rm_x + rm_y ** 2.
            grad_x_riemann, grad_y_riemann, grad_y1_riemann = rm_grad(z_riemann, [rm_x, rm_y, rm_y1], allow_unused=True)
            
            if TORCH_AVAILABLE:
                z_torch = 2. * torch_x + torch_y ** 2.
                grad_x_torch, grad_y_torch, grad_y1_torch = torch_autograd.grad(z_torch, [torch_x, torch_y, torch_y1], allow_unused=True)
            else:
                grad_x_torch, grad_y_torch, grad_y1_torch = None, None, None
            
            # 比较结果
            rm_result = (grad_x_riemann, grad_y_riemann, grad_y1_riemann)
            torch_result = (grad_x_torch, grad_y_torch, grad_y1_torch)
            passed = compare_values(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"梯度计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_tensor_gradient(self):
        """测试场景4: 张量的梯度计算"""
        case_name = "张量的梯度计算"
        start_time = time.time()
        try:
            # 创建输入
            x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            grad_outputs_np = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
            
            rm_x = rm.tensor(x_np, requires_grad=True)
            grad_outputs_riemann = rm.tensor(grad_outputs_np)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(x_np, requires_grad=True)
                grad_outputs_torch = torch.tensor(grad_outputs_np)
            else:
                torch_x, grad_outputs_torch = None, None
            
            # 计算梯度
            y_riemann = rm_x ** 2.
            grad_x_riemann = rm_grad(y_riemann, rm_x, grad_outputs=grad_outputs_riemann)[0]
            
            if TORCH_AVAILABLE:
                y_torch = torch_x ** 2.
                grad_x_torch = torch_autograd.grad(y_torch, torch_x, grad_outputs=grad_outputs_torch)[0]
            else:
                grad_x_torch = None
            
            # 比较结果
            passed = compare_values(grad_x_riemann, grad_x_torch)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"梯度计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_high_order_gradient(self):
        """测试场景5: 高阶导数计算 (使用create_graph参数)"""
        case_name = "高阶导数计算"
        start_time = time.time()
        try:
            # 创建输入
            x_np = 2.0
            rm_x = rm.tensor(x_np, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(x_np, requires_grad=True)
            else:
                torch_x = None
            
            # 计算一阶导数和二阶导数
            y_riemann = rm_x ** 3.
            dy_dx_riemann = rm_grad(y_riemann, rm_x, create_graph=True)[0]
            d2y_dx2_riemann = rm_grad(dy_dx_riemann, rm_x)[0]
            
            if TORCH_AVAILABLE:
                y_torch = torch_x ** 3.
                dy_dx_torch = torch_autograd.grad(y_torch, torch_x, create_graph=True)[0]
                d2y_dx2_torch = torch_autograd.grad(dy_dx_torch, torch_x)[0]
            else:
                dy_dx_torch, d2y_dx2_torch = None, None
            
            # 比较结果
            rm_result = (dy_dx_riemann, d2y_dx2_riemann)
            torch_result = (dy_dx_torch, d2y_dx2_torch)
            passed = compare_values(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"梯度计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_error_cases(self):
        """测试场景6: 错误情况处理"""
        start_time = time.time()
        
        # 测试1: 非叶节点错误
        case_name = "非叶节点错误处理"
        try:
            x = rm.tensor(1.0)  # 没有requires_grad=True
            with self.assertRaises(Exception):
                grad_x = rm_grad(x, x)
            passed = True
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            self.fail(f"错误测试失败: {case_name} - {str(e)}")
        
        # 测试2: 非标量输出且没有grad_outputs
        case_name = "非标量输出错误处理"
        try:
            x = rm.tensor([1.0, 2.0], requires_grad=True)
            y = x ** 2.
            with self.assertRaises(Exception):
                grad_x = rm_grad(y, x)  # 应该报错，因为y不是标量且没有提供grad_outputs
            passed = True
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            self.fail(f"错误测试失败: {case_name} - {str(e)}")
        
        # 测试3: grad_outputs形状不匹配
        case_name = "grad_outputs形状不匹配错误处理"
        try:
            x = rm.tensor([1.0, 2.0], requires_grad=True)
            y = x ** 2.
            grad_outputs = rm.tensor([1.0])  # 形状不匹配
            with self.assertRaises(Exception):
                grad_x = rm_grad(y, x, grad_outputs=grad_outputs)
            passed = True
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            self.fail(f"错误测试失败: {case_name} - {str(e)}")

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行梯度函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestGradFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)