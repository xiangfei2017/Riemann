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
    
    def run_clamp_test(self, test_case, is_inplace=False):
        """运行单个clamp测试用例"""
        name = test_case['name']
        x_data = test_case['x_data']
        min_val = test_case.get('min', None)
        max_val = test_case.get('max', None)
        
        try:
            # 准备输入数据
            rm_x = rm.tensor(x_data)
            
            # 执行clamp操作
            if is_inplace:
                rm_result = rm_x.clamp_(min_val, max_val) if min_val is not None and max_val is not None else \
                           rm_x.clamp_(min=min_val) if min_val is not None else \
                           rm_x.clamp_(max=max_val)
                # 对于原地操作，结果就是输入张量
                rm_result_data = rm_x
            else:
                # 处理out参数
                if 'out' in test_case:
                    out_shape = x_data.shape
                    rm_out = rm.zeros(out_shape)
                    rm_result = rm.clamp(rm_x, min_val, max_val, out=rm_out) if min_val is not None and max_val is not None else \
                               rm.clamp(rm_x, min=min_val, out=rm_out) if min_val is not None else \
                               rm.clamp(rm_x, max=max_val, out=rm_out)
                    rm_result_data = rm_out
                else:
                    rm_result = rm.clamp(rm_x, min_val, max_val) if min_val is not None and max_val is not None else \
                               rm.clamp(rm_x, min=min_val) if min_val is not None else \
                               rm.clamp(rm_x, max=max_val)
                    rm_result_data = rm_result
            
            # 使用PyTorch进行比较
            if TORCH_AVAILABLE:
                t_x = torch.tensor(x_data)
                if is_inplace:
                    t_result = t_x.clamp_(min_val, max_val) if min_val is not None and max_val is not None else \
                               t_x.clamp_(min=min_val) if min_val is not None else \
                               t_x.clamp_(max=max_val)
                    t_result_data = t_x
                else:
                    if 'out' in test_case:
                        out_shape = x_data.shape
                        t_out = torch.zeros(out_shape)
                        t_result = torch.clamp(t_x, min_val, max_val, out=t_out) if min_val is not None and max_val is not None else \
                                   torch.clamp(t_x, min=min_val, out=t_out) if min_val is not None else \
                                   torch.clamp(t_x, max=max_val, out=t_out)
                        t_result_data = t_out
                    else:
                        t_result = torch.clamp(t_x, min_val, max_val) if min_val is not None and max_val is not None else \
                                   torch.clamp(t_x, min=min_val) if min_val is not None else \
                                   torch.clamp(t_x, max=max_val)
                        t_result_data = t_result
            else:
                t_result_data = None
            
            # 比较结果
            passed = compare_values(rm_result_data, t_result_data)
            
            # 梯度测试
            grad_passed = True
            if TORCH_AVAILABLE and test_case.get('grad_test', True):
                # 确保数据类型是浮点型
                if x_data.dtype not in [np.float32, np.float64]:
                    x_data_float = x_data.astype(np.float32)
                else:
                    x_data_float = x_data
                
                # Riemann梯度计算
                if is_inplace:
                    # 原地操作不能对需要梯度的叶子节点张量执行
                    rm_x_leaf = rm.tensor(x_data_float, requires_grad=True)
                    rm_x_non_leaf = rm_x_leaf * 1.0  # 生成非叶子节点张量
                    if min_val is not None and max_val is not None:
                        rm_x_non_leaf.clamp_(min_val, max_val)
                    elif min_val is not None:
                        rm_x_non_leaf.clamp_(min=min_val)
                    else:
                        rm_x_non_leaf.clamp_(max=max_val)
                    rm_sum = rm.sum(rm_x_non_leaf)
                    rm_sum.backward()
                    rm_grad = rm_x_leaf.grad
                else:
                    rm_x_grad = rm.tensor(x_data_float, requires_grad=True)
                    if min_val is not None and max_val is not None:
                        rm_result_grad = rm.clamp(rm_x_grad, min_val, max_val)
                    elif min_val is not None:
                        rm_result_grad = rm.clamp(rm_x_grad, min=min_val)
                    else:
                        rm_result_grad = rm.clamp(rm_x_grad, max=max_val)
                    rm_sum = rm.sum(rm_result_grad)
                    rm_sum.backward()
                    rm_grad = rm_x_grad.grad
                
                # PyTorch梯度计算
                if is_inplace:
                    t_x_leaf = torch.tensor(x_data_float, requires_grad=True)
                    t_x_non_leaf = t_x_leaf * 1.0  # 生成非叶子节点张量
                    if min_val is not None and max_val is not None:
                        t_x_non_leaf.clamp_(min_val, max_val)
                    elif min_val is not None:
                        t_x_non_leaf.clamp_(min=min_val)
                    else:
                        t_x_non_leaf.clamp_(max=max_val)
                    t_sum = torch.sum(t_x_non_leaf)
                    t_sum.backward()
                    t_grad = t_x_leaf.grad
                else:
                    t_x_grad = torch.tensor(x_data_float, requires_grad=True)
                    if min_val is not None and max_val is not None:
                        t_result_grad = torch.clamp(t_x_grad, min_val, max_val)
                    elif min_val is not None:
                        t_result_grad = torch.clamp(t_x_grad, min=min_val)
                    else:
                        t_result_grad = torch.clamp(t_x_grad, max=max_val)
                    t_sum = torch.sum(t_result_grad)
                    t_sum.backward()
                    t_grad = t_x_grad.grad
                
                grad_passed = compare_values(rm_grad, t_grad)
            
            # 特殊测试：out参数返回值验证
            if not is_inplace and 'out' in test_case:
                out_passed = (rm_result is rm_out)
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  子用例: {name} - out返回值验证 - {Colors.OKGREEN if out_passed else Colors.FAIL}{'通过' if out_passed else '失败'}{Colors.ENDC}")
                return passed and out_passed, grad_passed
            
            # 特殊测试：原地操作返回值验证
            if is_inplace:
                return_passed = (rm_result is rm_x)
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  子用例: {name} - 返回值验证 - {Colors.OKGREEN if return_passed else Colors.FAIL}{'通过' if return_passed else '失败'}{Colors.ENDC}")
                return passed and return_passed, grad_passed
            
            return passed, grad_passed
            
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                print(f"  子用例: {name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
            return False, False

    def test_clamp_non_inplace(self):
        """测试非原地函数clamp()的所有场景"""
        case_name = "clamp非原地函数测试组"
        start_time = time.time()
        try:
            # 定义测试用例列表
            test_cases = [
                {
                    "name": "基本clamp功能",
                    "x_data": np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32),
                    "min": 2.0,
                    "max": 8.0
                },
                {
                    "name": "只设置min参数",
                    "x_data": np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32),
                    "min": 2.0
                },
                {
                    "name": "只设置max参数",
                    "x_data": np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32),
                    "max": 6.0
                },
                {
                    "name": "边界值情况 - 所有值在范围内",
                    "x_data": np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                    "min": 2.0,
                    "max": 7.0
                },
                {
                    "name": "边界值情况 - 所有值小于min",
                    "x_data": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    "min": 5.0
                },
                {
                    "name": "边界值情况 - 所有值大于max",
                    "x_data": np.array([[8.0, 9.0], [10.0, 11.0]], dtype=np.float32),
                    "max": 7.0
                },
                {
                    "name": "标量输入",
                    "x_data": np.array(5.0, dtype=np.float32),
                    "min": 2.0,
                    "max": 8.0
                },
                {
                    "name": "不同形状输入",
                    "x_data": np.random.randn(2, 3, 4).astype(np.float32),
                    "min": -1.0,
                    "max": 1.0
                },
                {
                    "name": "out参数测试",
                    "x_data": np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32),
                    "min": 2.0,
                    "max": 8.0,
                    "out": True
                },
                {
                    "name": "混合数据类型",
                    "x_data": np.array([[-3, 0, 3], [5, 7, 9]], dtype=np.int32),
                    "min": 2,
                    "max": 8,
                    "grad_test": False  # 整数类型不支持梯度测试
                }
            ]
            
            passed_cases = []
            grad_passed_cases = []
            
            # 执行所有测试用例
            for test_case in test_cases:
                passed, grad_passed = self.run_clamp_test(test_case, is_inplace=False)
                passed_cases.append(passed)
                if grad_passed is not None:
                    grad_passed_cases.append(grad_passed)
                
                # 记录测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(test_case['name'], passed)
                    if TORCH_AVAILABLE and test_case.get('grad_test', True):
                        stats.add_result(f"{test_case['name']} - 梯度跟踪", grad_passed)
                    print(f"  子用例: {test_case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{'通过' if passed else '失败'}{Colors.ENDC}")
                    if TORCH_AVAILABLE and test_case.get('grad_test', True):
                        print(f"  子用例: {test_case['name']} - 梯度跟踪 - {Colors.OKGREEN if grad_passed else Colors.FAIL}{'通过' if grad_passed else '失败'}{Colors.ENDC}")
            
            # 计算总结果
            passed_forward = all(passed_cases)
            passed_grad = all(grad_passed_cases) if grad_passed_cases else True
            passed = passed_forward and passed_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                print(f"测试用例组: {case_name} - {Colors.OKGREEN if passed_forward else Colors.FAIL}{'通过' if passed_forward else '失败'}{Colors.ENDC}")
                if TORCH_AVAILABLE:
                    print(f"梯度测试 - {Colors.OKGREEN if passed_grad else Colors.FAIL}{'通过' if passed_grad else '失败'}{Colors.ENDC}")
                print(f" ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed_forward, f"clamp非原地函数测试组失败")
            self.assertTrue(passed_grad, f"clamp非原地函数梯度测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例组: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise

    def test_clamp_inplace(self):
        """测试原地修剪函数clamp_()"""
        case_name = "clamp原地函数测试组"
        start_time = time.time()
        try:
            # 定义测试用例列表
            test_cases = [
                {
                    "name": "基本clamp_功能",
                    "x_data": np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32),
                    "min": 2.0,
                    "max": 8.0
                },
                {
                    "name": "只设置min参数",
                    "x_data": np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32),
                    "min": 2.0
                },
                {
                    "name": "只设置max参数",
                    "x_data": np.array([[-3.0, 0.0, 3.0], [5.0, 7.0, 9.0]], dtype=np.float32),
                    "max": 6.0
                },
                {
                    "name": "边界值情况 - 所有值在范围内",
                    "x_data": np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                    "min": 2.0,
                    "max": 7.0
                },
                {
                    "name": "边界值情况 - 所有值小于min",
                    "x_data": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    "min": 5.0
                },
                {
                    "name": "边界值情况 - 所有值大于max",
                    "x_data": np.array([[8.0, 9.0], [10.0, 11.0]], dtype=np.float32),
                    "max": 7.0
                }
            ]
            
            passed_cases = []
            grad_passed_cases = []
            
            # 执行所有测试用例
            for test_case in test_cases:
                passed, grad_passed = self.run_clamp_test(test_case, is_inplace=True)
                passed_cases.append(passed)
                if grad_passed is not None:
                    grad_passed_cases.append(grad_passed)
                
                # 记录测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(test_case['name'], passed)
                    if TORCH_AVAILABLE and test_case.get('grad_test', True):
                        stats.add_result(f"{test_case['name']} - 梯度跟踪", grad_passed)
                    print(f"  子用例: {test_case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{'通过' if passed else '失败'}{Colors.ENDC}")
                    if TORCH_AVAILABLE and test_case.get('grad_test', True):
                        print(f"  子用例: {test_case['name']} - 梯度跟踪 - {Colors.OKGREEN if grad_passed else Colors.FAIL}{'通过' if grad_passed else '失败'}{Colors.ENDC}")
            
            # 计算总结果
            passed_forward = all(passed_cases)
            passed_grad = all(grad_passed_cases) if grad_passed_cases else True
            passed = passed_forward and passed_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                print(f"测试用例组: {case_name} - {Colors.OKGREEN if passed_forward else Colors.FAIL}{'通过' if passed_forward else '失败'}{Colors.ENDC}")
                if TORCH_AVAILABLE:
                    print(f"梯度测试 - {Colors.OKGREEN if passed_grad else Colors.FAIL}{'通过' if passed_grad else '失败'}{Colors.ENDC}")
                print(f" ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed_forward, f"clamp原地函数测试组失败")
            self.assertTrue(passed_grad, f"clamp原地函数梯度测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例组: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
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
