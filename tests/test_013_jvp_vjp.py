import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.autograd.functional import jvp, vjp
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    from torch.autograd.functional import jvp as torch_jvp, vjp as torch_vjp
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的jvp和vjp函数")
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

class TestJvpVjpFunctions(unittest.TestCase):
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
    
    def _run_test_case(self, case_name, test_func, error_message=None):
        """运行单个测试用例，处理通用的计时、异常捕获和结果记录逻辑"""
        start_time = time.time()
        try:
            # 执行实际测试逻辑
            passed = test_func()
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
            # 断言确保测试通过
            error_msg = error_message or f"测试失败: {case_name}"
            self.assertTrue(passed, error_msg)
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_single_input_single_output(self):
        """测试场景1: 单输入单输出函数"""
        case_name = "单输入单输出函数 JVP/VJP"
        
        def test_func():
            # 定义测试函数
            def f(x):
                return x ** 2
            
            def pt_f(x):
                return x ** 2
            
            # 创建测试数据
            np_x = np.random.randn(3, 4)
            rm_x = rm.tensor(np_x, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(np_x, requires_grad=True)
            else:
                torch_x = None
            
            np_v = np.random.randn(3, 4)
            rm_v = rm.tensor(np_v)
            if TORCH_AVAILABLE:
                torch_v = torch.tensor(np_v)
            else:
                torch_v = None
            
            # 初始化变量以避免UnboundLocalError
            torch_jvp_result = None
            
            # 计算Riemann的JVP
            rm_outputs, rm_jvp = jvp(f, rm_x, rm_v)
            
            # 计算PyTorch的JVP
            if TORCH_AVAILABLE:
                torch_outputs, torch_jvp_result = torch_jvp(pt_f, torch_x, torch_v)
            
            # 比较结果
            passed = compare_values(rm_jvp, torch_jvp_result)
            
            # 计算VJP部分 - 修复v参数形状与输出匹配
            np_v_output = np.ones_like(np_x)  # 形状与输出x**2匹配
            rm_v_output = rm.tensor(np_v_output)
            if TORCH_AVAILABLE:
                torch_v_output = torch.tensor(np_v_output)
            else:
                torch_v_output = None
            
            # 初始化变量以避免UnboundLocalError
            torch_vjp_result = None
            
            # 计算Riemann的VJP
            rm_outputs_vjp, rm_vjp = vjp(f, rm_x, rm_v_output)
            
            # 计算PyTorch的VJP
            if TORCH_AVAILABLE:
                torch_outputs_vjp, torch_vjp_result = torch_vjp(pt_f, torch_x, torch_v_output)
            
            # 比较VJP结果
            vjp_passed = compare_values(rm_vjp, torch_vjp_result)
            final_passed = passed and vjp_passed
            
            # 详细错误信息
            if IS_RUNNING_AS_SCRIPT and not final_passed:
                jvp_status = "通过" if compare_values(rm_jvp, torch_jvp_result) else "失败"
                vjp_status = "通过" if vjp_passed else "失败"
                print(f"  JVP比较: {jvp_status}")
                print(f"  VJP比较: {vjp_status}")
            
            return final_passed
        
        self._run_test_case(case_name, test_func, error_message=f"JVP/VJP计算结果不匹配: {case_name}")
    
    def test_single_input_multiple_outputs(self):
        """测试场景2: 单输入多输出函数"""
        case_name = "单输入多输出函数 JVP/VJP"
        
        def test_func():
            # 定义测试函数
            def f(x):
                return x ** 2, x.sum()
            
            def pt_f(x):
                return x ** 2, x.sum()
            
            # 创建测试数据
            np_x = np.random.randn(3, 4)
            rm_x = rm.tensor(np_x, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(np_x, requires_grad=True)
            else:
                torch_x = None
            
            np_v = np.random.randn(3, 4)
            rm_v = rm.tensor(np_v)
            if TORCH_AVAILABLE:
                torch_v = torch.tensor(np_v)
            else:
                torch_v = None
            
            # 初始化变量以避免UnboundLocalError
            torch_jvp_result = None
            
            # 计算Riemann的JVP
            rm_outputs, rm_jvp = jvp(f, rm_x, rm_v)
            
            # 计算PyTorch的JVP
            if TORCH_AVAILABLE:
                torch_outputs, torch_jvp_result = torch_jvp(pt_f, torch_x, torch_v)
            
            # 比较结果 - 分别比较每个输出的JVP
            passed = True
            if TORCH_AVAILABLE and torch_jvp_result is not None:
                for i in range(len(rm_jvp)):
                    if not compare_values(rm_jvp[i], torch_jvp_result[i]):
                        passed = False
                        break
            
            # 计算VJP部分 - 精确匹配v参数形状与函数输出形状
            # 先计算函数实际输出，以确定准确的形状
            np_output1, np_output2 = f(np_x)
            np_v1 = np.ones_like(np_output1)  # 与第一个输出形状精确匹配
            np_v2 = np.ones_like(np_output2)  # 与第二个输出形状精确匹配
            rm_v1 = rm.tensor(np_v1)
            rm_v2 = rm.tensor(np_v2)
            if TORCH_AVAILABLE:
                torch_v1 = torch.tensor(np_v1)
                torch_v2 = torch.tensor(np_v2)
            else:
                torch_v1, torch_v2 = None, None
            
            # 初始化变量以避免UnboundLocalError
            torch_vjp_result = None
            
            # 计算Riemann的VJP
            rm_outputs_vjp, rm_vjp = vjp(f, rm_x, (rm_v1, rm_v2))
            
            # 计算PyTorch的VJP
            if TORCH_AVAILABLE:
                torch_outputs_vjp, torch_vjp_result = torch_vjp(pt_f, torch_x, (torch_v1, torch_v2))
            
            # 比较VJP结果 - 分别比较每个输入的VJP
            vjp_passed = True
            if TORCH_AVAILABLE and torch_vjp_result is not None:
                for i in range(len(rm_vjp)):
                    if not compare_values(rm_vjp[i], torch_vjp_result[i]):
                        vjp_passed = False
                        break
            
            final_passed = passed and vjp_passed
            
            # 详细错误信息
            if IS_RUNNING_AS_SCRIPT and not final_passed:
                jvp_status = "通过" if (not TORCH_AVAILABLE or torch_jvp_result is None or all(compare_values(rm_jvp[i], torch_jvp_result[i]) for i in range(len(rm_jvp)))) else "失败"
                vjp_status = "通过" if (not TORCH_AVAILABLE or torch_vjp_result is None or all(compare_values(rm_vjp[i], torch_vjp_result[i]) for i in range(len(rm_vjp)))) else "失败"
                print(f"  JVP比较: {jvp_status}")
                print(f"  VJP比较: {vjp_status}")
            
            return final_passed
        
        self._run_test_case(case_name, test_func, error_message=f"JVP/VJP计算结果不匹配: {case_name}")
    
    def test_multiple_inputs_single_output(self):
        """测试场景3: 多输入单输出函数"""
        case_name = "多输入单输出函数 JVP/VJP"
        
        def test_func():
            # 定义测试函数
            def f(x, y):
                return x @ y + x.sum() + y.sum()
            
            def pt_f(x, y):
                return x @ y + x.sum() + y.sum()
            
            # 创建测试数据
            np_x = np.random.randn(2, 3)
            np_y = np.random.randn(3, 4)
            rm_x = rm.tensor(np_x, requires_grad=True)
            rm_y = rm.tensor(np_y, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(np_x, requires_grad=True)
                torch_y = torch.tensor(np_y, requires_grad=True)
            else:
                torch_x, torch_y = None, None
            
            np_vx = np.random.randn(2, 3)
            np_vy = np.random.randn(3, 4)
            rm_vx = rm.tensor(np_vx)
            rm_vy = rm.tensor(np_vy)
            if TORCH_AVAILABLE:
                torch_vx = torch.tensor(np_vx)
                torch_vy = torch.tensor(np_vy)
            else:
                torch_vx, torch_vy = None, None
            
            # 初始化变量以避免UnboundLocalError
            torch_jvp_result = None
            
            # 计算Riemann的JVP
            rm_output, rm_jvp = jvp(f, (rm_x, rm_y), (rm_vx, rm_vy))
            
            # 计算PyTorch的JVP
            if TORCH_AVAILABLE:
                torch_output, torch_jvp_result = torch_jvp(pt_f, (torch_x, torch_y), (torch_vx, torch_vy))
            
            # 比较结果
            passed = compare_values(rm_jvp, torch_jvp_result)
            
            # 计算VJP部分 - 精确匹配v参数形状与函数输出形状
            # 函数输出是(2,4)，因为x @ y的形状是(2,4)，标量加法会广播到整个矩阵
            np_v_output = np.ones((2, 4))  # 与函数输出形状(2,4)精确匹配
            rm_v_output = rm.tensor(np_v_output)
            if TORCH_AVAILABLE:
                torch_v_output = torch.tensor(np_v_output)
            else:
                torch_v_output = None
            
            # 初始化变量以避免UnboundLocalError
            torch_vjp_result = None
            
            # 计算Riemann的VJP
            rm_output_vjp, rm_vjp = vjp(f, (rm_x, rm_y), rm_v_output)
            
            # 计算PyTorch的VJP
            if TORCH_AVAILABLE:
                torch_output_vjp, torch_vjp_result = torch_vjp(pt_f, (torch_x, torch_y), torch_v_output)
            
            # 比较VJP结果 - 分别比较每个输入的VJP
            vjp_passed = True
            if TORCH_AVAILABLE and torch_vjp_result is not None:
                for i in range(len(rm_vjp)):
                    if not compare_values(rm_vjp[i], torch_vjp_result[i]):
                        vjp_passed = False
                        break
            
            final_passed = passed and vjp_passed
            
            # 详细错误信息
            if IS_RUNNING_AS_SCRIPT and not final_passed:
                jvp_status = "通过" if compare_values(rm_jvp, torch_jvp_result) else "失败"
                vjp_status = "通过" if (not TORCH_AVAILABLE or torch_vjp_result is None or all(compare_values(rm_vjp[i], torch_vjp_result[i]) for i in range(len(rm_vjp)))) else "失败"
                print(f"  JVP比较: {jvp_status}")
                print(f"  VJP比较: {vjp_status}")
            
            return final_passed
        
        self._run_test_case(case_name, test_func, error_message=f"JVP/VJP计算结果不匹配: {case_name}")
    
    def test_multiple_inputs_multiple_outputs(self):
        """测试场景4: 多输入多输出函数"""
        case_name = "多输入多输出函数 JVP/VJP"
        
        def test_func():
            # 定义测试函数
            def f(x, y):
                return x @ y, x.sum() + y.sum()
            
            def pt_f(x, y):
                return x @ y, x.sum() + y.sum()
            
            # 创建测试数据
            np_x = np.random.randn(2, 3)
            np_y = np.random.randn(3, 4)
            rm_x = rm.tensor(np_x, requires_grad=True)
            rm_y = rm.tensor(np_y, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(np_x, requires_grad=True)
                torch_y = torch.tensor(np_y, requires_grad=True)
            else:
                torch_x, torch_y = None, None
            
            np_vx = np.random.randn(2, 3)
            np_vy = np.random.randn(3, 4)
            rm_vx = rm.tensor(np_vx)
            rm_vy = rm.tensor(np_vy)
            if TORCH_AVAILABLE:
                torch_vx = torch.tensor(np_vx)
                torch_vy = torch.tensor(np_vy)
            else:
                torch_vx, torch_vy = None, None
            
            # 初始化变量以避免UnboundLocalError
            torch_jvp_result = None
            
            # 计算Riemann的JVP
            rm_outputs, rm_jvp = jvp(f, (rm_x, rm_y), (rm_vx, rm_vy))
            
            # 计算PyTorch的JVP
            if TORCH_AVAILABLE:
                torch_outputs, torch_jvp_result = torch_jvp(pt_f, (torch_x, torch_y), (torch_vx, torch_vy))
            
            # 比较JVP结果 - 分别比较每个输出的JVP
            jvp_passed = True
            if TORCH_AVAILABLE and torch_jvp_result is not None:
                for i in range(len(rm_jvp)):
                    if not compare_values(rm_jvp[i], torch_jvp_result[i]):
                        jvp_passed = False
                        break
            
            # 计算VJP部分 - 精确匹配v参数形状与输出匹配
            np_v1 = np.ones((2, 4))  # 与x@y的输出形状匹配
            np_v2 = np.array(1.0)[()]  # 标量，与sum输出匹配
            rm_v1 = rm.tensor(np_v1)
            rm_v2 = rm.tensor(np_v2)
            if TORCH_AVAILABLE:
                torch_v1 = torch.tensor(np_v1)
                torch_v2 = torch.tensor(np_v2)
            else:
                torch_v1, torch_v2 = None, None
            
            # 初始化变量以避免UnboundLocalError
            torch_vjp_result = None
            
            # 计算Riemann的VJP
            rm_outputs_vjp, rm_vjp = vjp(f, (rm_x.copy(), rm_y.copy()), (rm_v1, rm_v2))
            
            # 计算PyTorch的VJP
            if TORCH_AVAILABLE:
                torch_outputs_vjp, torch_vjp_result = torch_vjp(pt_f, (torch_x, torch_y), (torch_v1, torch_v2))
            
            # 比较VJP结果 - 分别比较每个输入的VJP
            vjp_passed = True
            if TORCH_AVAILABLE and torch_vjp_result is not None:
                for i in range(len(rm_vjp)):
                    if not compare_values(rm_vjp[i], torch_vjp_result[i]):
                        vjp_passed = False
                        # 添加详细的调试输出
                        print(f"\nVJP比较失败 - 输入索引 {i}:")
                        print(f"Riemann VJP结果: {rm_vjp[i].numpy()}")
                        print(f"PyTorch VJP结果: {torch_vjp_result[i].detach().numpy()}")
            
            final_passed = jvp_passed and vjp_passed
            
            # 详细错误信息
            if IS_RUNNING_AS_SCRIPT and not final_passed:
                jvp_status = "通过" if (not TORCH_AVAILABLE or torch_jvp_result is None or all(compare_values(rm_jvp[i], torch_jvp_result[i]) for i in range(len(rm_jvp)))) else "失败"
                vjp_status = "通过" if (not TORCH_AVAILABLE or torch_vjp_result is None or all(compare_values(rm_vjp[i], torch_vjp_result[i]) for i in range(len(rm_vjp)))) else "失败"
                print(f"  JVP比较: {jvp_status}")
                print(f"  VJP比较: {vjp_status}")
            
            return final_passed
        
        self._run_test_case(case_name, test_func, error_message=f"JVP/VJP计算结果不匹配: {case_name}")
    
    def test_create_graph_and_strict(self):
        """测试场景6: create_graph和strict参数"""
        test_cases = [
            {"name": "JVP create_graph=True", "type": "jvp", "param": "create_graph", "value": True, "should_pass": True},
            {"name": "VJP create_graph=True", "type": "vjp", "param": "create_graph", "value": True, "should_pass": True},
            {"name": "JVP strict=False（忽略部分输入）", "type": "jvp", "param": "strict", "value": False, "should_pass": True},
            {"name": "JVP strict=True（忽略部分输入）", "type": "jvp", "param": "strict", "value": True, "should_pass": False},  # 修复：根据实际行为设为False
            {"name": "VJP strict=False（忽略部分输入）", "type": "vjp", "param": "strict", "value": False, "should_pass": True},
            {"name": "VJP strict=True（忽略部分输入）", "type": "vjp", "param": "strict", "value": True, "should_pass": False}  # 修复：根据实际行为设为False
        ]

        for case in test_cases:
            def test_func():
                if case["param"] == "create_graph" and case["type"] == "jvp":
                    # 定义测试函数
                    def f(x):
                        return x ** 2.
                    
                    # 创建输入
                    rm_x = rm.tensor([[2.0, 3.0]], requires_grad=True)
                    rm_v = rm.tensor([[1.0, 1.0]])
                    
                    # 计算JVP
                    rm_outputs, rm_jvp = jvp(f, rm_x, rm_v, create_graph=True)
                    
                    # 验证requires_grad
                    passed = hasattr(rm_jvp, 'requires_grad') and rm_jvp.requires_grad
                    return case["should_pass"] == passed
                    
                elif case["param"] == "create_graph" and case["type"] == "vjp":
                    # 定义测试函数
                    def f(x):
                        return x ** 2.
                    
                    # 创建输入
                    rm_x = rm.tensor([[2.0, 3.0]], requires_grad=True)
                    rm_v = rm.tensor([[1.0, 1.0]])
                    
                    # 计算VJP
                    rm_outputs, rm_vjp = vjp(f, rm_x, rm_v, create_graph=True)
                    
                    # 验证requires_grad
                    passed = hasattr(rm_vjp, 'requires_grad') and rm_vjp.requires_grad
                    return case["should_pass"] == passed
                    
                elif case["param"] == "strict" and case["type"] == "jvp":
                    # 定义忽略部分输入的函数
                    def f_independent(x, y):
                        return x ** 2.
                    
                    # 创建输入
                    rm_x = rm.tensor(2.0, requires_grad=True)
                    rm_y = rm.tensor(3.0, requires_grad=True)
                    rm_vx = rm.tensor(1.0)
                    rm_vy = rm.tensor(1.0)
                    
                    try:
                        # 尝试计算JVP
                        rm_outputs_indep, rm_jvp_indep = jvp(f_independent, (rm_x, rm_y), (rm_vx, rm_vy), strict=case["value"])
                        # 如果成功执行，设置passed为True
                        passed = True
                    except Exception as e:
                        # 如果抛出异常，设置passed为False
                        passed = False
                    
                    return case["should_pass"] == passed
                    
                elif case["param"] == "strict" and case["type"] == "vjp":
                    # 定义忽略部分输入的函数
                    def f_independent(x, y):
                        return x ** 2.
                    
                    # 创建输入
                    rm_x = rm.tensor(2.0, requires_grad=True)
                    rm_y = rm.tensor(3.0, requires_grad=True)
                    rm_v = rm.tensor(1.0)
                    
                    try:
                        # 尝试计算VJP
                        rm_outputs_indep, rm_vjp_indep = vjp(f_independent, (rm_x, rm_y), rm_v, strict=case["value"])
                        # 如果成功执行，设置passed为True
                        passed = True
                    except Exception as e:
                        # 如果抛出异常，设置passed为False
                        passed = False
                    
                    return case["should_pass"] == passed
                
                return False  # 默认返回失败，除非上面的条件分支执行
            
            self._run_test_case(case["name"], test_func, error_message=f"测试用例失败: {case['name']}")
    
    def test_scalar_output_vjp_optional_v(self):
        """测试场景7: 标量输出时VJP的可选v参数"""
        case_name = "标量输出时VJP的可选v参数"
        
        def test_func():
            # 定义测试函数
            def f(x):
                return x.sum()
            
            def pt_f(x):
                return x.sum()
            
            # 创建测试数据
            np_x = np.random.randn(3, 4)
            rm_x = rm.tensor(np_x, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(np_x, requires_grad=True)
            else:
                torch_x = None
            
            # 初始化变量以避免UnboundLocalError
            torch_vjp_result = None
            
            # 计算Riemann的VJP（不指定v参数）
            rm_output, rm_vjp = vjp(f, rm_x)
            
            # 计算PyTorch的VJP（同样不指定v参数）
            if TORCH_AVAILABLE:
                torch_output, torch_vjp_result = torch_vjp(pt_f, torch_x)
            
            # 比较结果
            final_passed = compare_values(rm_vjp, torch_vjp_result)
            
            # 详细错误信息
            if IS_RUNNING_AS_SCRIPT and not final_passed:
                print(f"  VJP比较: {'通过' if final_passed else '失败'}")
            
            return final_passed
        
        self._run_test_case(case_name, test_func, error_message=f"标量输出时VJP的可选v参数测试失败: {case_name}")

# 主函数
if __name__ == "__main__":
    IS_RUNNING_AS_SCRIPT = True
    clear_screen()
    
    print("开始测试JVP和VJP函数...")
    print(f"PyTorch 可用: {TORCH_AVAILABLE}")
    print("=" * 60)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestJvpVjpFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出
    result = runner.run(suite)
    
    # 打印统计摘要
    stats.print_summary()
    
    # 设置退出码
    sys.exit(not result.wasSuccessful())