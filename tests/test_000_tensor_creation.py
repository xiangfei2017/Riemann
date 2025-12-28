import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的张量创建函数")
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
        
        # 打印表头
        header_line = ""
        for i, header in enumerate(headers):
            header_width = self._get_display_width(header)
            padding = col_widths[i] - header_width
            header_line += header + " " * padding
        print(header_line)
        print("-"*total_width)
        
        # 打印数据行
        for func_name, stats in self.function_stats.items():
            pass_rate = stats["passed"]/stats["total"]*100 if stats["total"] > 0 else 0
            status_color = Colors.OKGREEN if pass_rate == 100 else Colors.FAIL
            
            func_name_display = func_name
            func_name_width = self._get_display_width(func_name_display)
            func_name_padding = col_widths[0] - func_name_width
            
            pass_total_display = f"{stats['passed']}/{stats['total']}"
            pass_total_width = self._get_display_width(pass_total_display)
            pass_total_padding = col_widths[1] - pass_total_width
            
            pass_rate_display = f"{pass_rate:.1f}%"
            pass_rate_width = self._get_display_width(pass_rate_display)
            pass_rate_padding = col_widths[2] - pass_rate_width
            
            time_display = f"{stats['time']:.4f}"
            time_width = self._get_display_width(time_display)
            time_padding = col_widths[3] - time_width
            
            # 组合数据行
            data_line = (func_name_display + " " * func_name_padding +
                         pass_total_display + " " * pass_total_padding +
                         status_color + pass_rate_display + Colors.ENDC + " " * pass_rate_padding +
                         time_display + " " * time_padding)
            print(data_line)
        
        print("="*total_width)

# 定义测试统计实例
stats = StatisticsCollector()

# 是否作为独立脚本运行
IS_RUNNING_AS_SCRIPT = False

# 比较函数：比较两个值是否相等
def compare_values(rm_val, torch_val, atol=1e-6, rtol=1e-6, check_dtype=False):
    """比较riemann和pytorch的值是否相等"""
    # 获取实际的数据数组
    rm_data = rm_val.numpy() if hasattr(rm_val, 'numpy') else rm_val.data if hasattr(rm_val, 'data') else rm_val
    torch_data = torch_val.detach().numpy() if hasattr(torch_val, 'detach') else torch_val
    
    # 检查形状是否相同
    if np.shape(rm_data) != np.shape(torch_data):
        return False, f"形状不匹配: {np.shape(rm_data)} vs {np.shape(torch_data)}"
    
    # 检查值是否在容差范围内相等
    try:
        if not np.allclose(rm_data, torch_data, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(rm_data - torch_data))
            return False, f"值不匹配，最大差异: {max_diff}"
    except Exception as e:
        return False, f"比较时出错: {str(e)}"
    
    # 检查数据类型是否相同（仅在check_dtype为True时执行）
    if check_dtype and hasattr(rm_data, 'dtype') and hasattr(torch_data, 'dtype'):
        if str(rm_data.dtype) != str(torch_data.dtype):
            return False, f"数据类型不匹配: {rm_data.dtype} vs {torch_data.dtype}"
    
    return True, ""

class TestTensorCreationFunctions(unittest.TestCase):
    def setUp(self):
        if IS_RUNNING_AS_SCRIPT:
            # 如果作为脚本运行，设置当前测试函数
            test_method_name = self._testMethodName
            self.current_test_name = test_method_name
            stats.start_function(test_method_name)
    
    def tearDown(self):
        if IS_RUNNING_AS_SCRIPT:
            # 如果作为脚本运行，结束当前测试函数的计时
            stats.end_function()
    
    # 修复test_zeros方法
    # 将原来的:
    def test_zeros(self):
        """测试zeros函数"""
        test_cases = [
            {"name": "标量形状", "shape": (), "dtype": None},
            {"name": "一维张量", "shape": (5,), "dtype": None},
            {"name": "二维张量", "shape": (3, 4), "dtype": None},
            {"name": "三维张量", "shape": (2, 3, 4), "dtype": None},
            {"name": "指定float32类型", "shape": (3, 4), "dtype": rm.float32},
            {"name": "指定int32类型", "shape": (3, 4), "dtype": rm.int32},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_zeros.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建张量
                rm_tensor = rm.zeros(case["shape"], dtype=case["dtype"])
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 检查所有元素是否为0
                    all_zeros = np.allclose(rm_tensor.numpy(), np.zeros(case["shape"]))
                    if not all_zeros:
                        passed = False
                        error_msg = "不是所有元素都是0"
                    else:
                        # 如果PyTorch可用，与PyTorch进行比较
                        if TORCH_AVAILABLE:
                            torch_dtype = None
                            if case["dtype"] == rm.float32:
                                torch_dtype = torch.float32
                            elif case["dtype"] == rm.int32:
                                torch_dtype = torch.int32
                            
                            torch_tensor = torch.zeros(case["shape"], dtype=torch_dtype)
                            passed, error_msg = compare_values(rm_tensor, torch_tensor)
                        else:
                            passed = True
                            error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_zeros_like(self):
        """测试zeros_like函数"""
        test_cases = [
            {"name": "标量输入", "shape": (), "dtype": None},
            {"name": "一维张量输入", "shape": (5,), "dtype": None},
            {"name": "二维张量输入", "shape": (3, 4), "dtype": None},
            {"name": "float32类型输入", "shape": (3, 4), "dtype": rm.float32},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_zeros_like.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建输入张量
                input_tensor = rm.ones(case["shape"], dtype=case["dtype"])
                
                # 使用Riemann创建zeros_like张量
                rm_tensor = rm.zeros_like(input_tensor)
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 检查所有元素是否为0
                    all_zeros = np.allclose(rm_tensor.numpy(), np.zeros(case["shape"]))
                    if not all_zeros:
                        passed = False
                        error_msg = "不是所有元素都是0"
                    else:
                        # 如果PyTorch可用，与PyTorch进行比较
                        if TORCH_AVAILABLE:
                            torch_dtype = None
                            if case["dtype"] == rm.float32:
                                torch_dtype = torch.float32
                            
                            torch_input = torch.ones(case["shape"], dtype=torch_dtype)
                            torch_tensor = torch.zeros_like(torch_input)
                            passed, error_msg = compare_values(rm_tensor, torch_tensor)
                        else:
                            passed = True
                            error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_ones(self):
        """测试ones函数"""
        test_cases = [
            {"name": "标量形状", "shape": (), "dtype": None},
            {"name": "一维张量", "shape": (5,), "dtype": None},
            {"name": "二维张量", "shape": (3, 4), "dtype": None},
            {"name": "三维张量", "shape": (2, 3, 4), "dtype": None},
            {"name": "指定float32类型", "shape": (3, 4), "dtype": rm.float32},
            {"name": "指定int32类型", "shape": (3, 4), "dtype": rm.int32},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_ones.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建张量
                rm_tensor = rm.ones(case["shape"], dtype=case["dtype"])
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 检查所有元素是否为1
                    all_ones = np.allclose(rm_tensor.numpy(), np.ones(case["shape"]))
                    if not all_ones:
                        passed = False
                        error_msg = "不是所有元素都是1"
                    else:
                        # 如果PyTorch可用，与PyTorch进行比较
                        if TORCH_AVAILABLE:
                            torch_dtype = None
                            if case["dtype"] == rm.float32:
                                torch_dtype = torch.float32
                            elif case["dtype"] == rm.int32:
                                torch_dtype = torch.int32
                            
                            torch_tensor = torch.ones(case["shape"], dtype=torch_dtype)
                            passed, error_msg = compare_values(rm_tensor, torch_tensor)
                        else:
                            passed = True
                            error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_ones_like(self):
        """测试ones_like函数"""
        test_cases = [
            {"name": "标量输入", "shape": (), "dtype": None},
            {"name": "一维张量输入", "shape": (5,), "dtype": None},
            {"name": "二维张量输入", "shape": (3, 4), "dtype": None},
            {"name": "float32类型输入", "shape": (3, 4), "dtype": rm.float32},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_ones_like.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建输入张量
                input_tensor = rm.zeros(case["shape"], dtype=case["dtype"])
                
                # 使用Riemann创建ones_like张量
                rm_tensor = rm.ones_like(input_tensor)
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 检查所有元素是否为1
                    all_ones = np.allclose(rm_tensor.numpy(), np.ones(case["shape"]))
                    if not all_ones:
                        passed = False
                        error_msg = "不是所有元素都是1"
                    else:
                        # 如果PyTorch可用，与PyTorch进行比较
                        if TORCH_AVAILABLE:
                            torch_dtype = None
                            if case["dtype"] == rm.float32:
                                torch_dtype = torch.float32
                            
                            torch_input = torch.zeros(case["shape"], dtype=torch_dtype)
                            torch_tensor = torch.ones_like(torch_input)
                            passed, error_msg = compare_values(rm_tensor, torch_tensor)
                        else:
                            passed = True
                            error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_empty(self):
        """测试empty函数"""
        test_cases = [
            {"name": "一维张量", "shape": (5,)},
            {"name": "二维张量", "shape": (3, 4)},
            {"name": "三维张量", "shape": (2, 3, 4)},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_empty.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建empty张量
                rm_tensor = rm.empty(case["shape"])
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 对于empty函数，我们只检查形状，不检查值
                    passed = True
                    error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_empty_like(self):
        """测试empty_like函数"""
        test_cases = [
            {"name": "一维张量输入", "shape": (5,)},
            {"name": "二维张量输入", "shape": (3, 4)},
            {"name": "三维张量输入", "shape": (2, 3, 4)},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_empty_like.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建输入张量
                input_tensor = rm.ones(case["shape"])
                
                # 使用Riemann创建empty_like张量
                rm_tensor = rm.empty_like(input_tensor)
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 对于empty_like函数，我们只检查形状，不检查值
                    passed = True
                    error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_full(self):
        """测试full函数"""
        test_cases = [
            {"name": "标量形状整数填充", "shape": (), "fill_value": 5, "dtype": None},
            {"name": "一维张量浮点数填充", "shape": (5,), "fill_value": 3.14, "dtype": None},
            {"name": "二维张量整数填充", "shape": (3, 4), "fill_value": 7, "dtype": None},
            {"name": "指定float32类型", "shape": (3, 4), "fill_value": 3.14, "dtype": rm.float32},
            {"name": "指定int32类型", "shape": (3, 4), "fill_value": 7, "dtype": rm.int32},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_full.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建张量
                rm_tensor = rm.full(case["shape"], fill_value=case["fill_value"], dtype=case["dtype"])
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 检查所有元素是否为指定值
                    expected_values = np.full(case["shape"], case["fill_value"])
                    all_match = np.allclose(rm_tensor.numpy(), expected_values)
                    if not all_match:
                        passed = False
                        error_msg = f"值不匹配，期望{case['fill_value']}"
                    else:
                        # 如果PyTorch可用，与PyTorch进行比较
                        if TORCH_AVAILABLE:
                            torch_dtype = None
                            if case["dtype"] == rm.float32:
                                torch_dtype = torch.float32
                            elif case["dtype"] == rm.int32:
                                torch_dtype = torch.int32
                            
                            torch_tensor = torch.full(case["shape"], case["fill_value"], dtype=torch_dtype)
                            passed, error_msg = compare_values(rm_tensor, torch_tensor)
                        else:
                            passed = True
                            error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_full_like(self):
        """测试full_like函数"""
        test_cases = [
            {"name": "标量输入整数填充", "shape": (), "fill_value": 5, "dtype": None},
            {"name": "一维张量输入浮点数填充", "shape": (5,), "fill_value": 3.14, "dtype": None},
            {"name": "二维张量输入整数填充", "shape": (3, 4), "fill_value": 7, "dtype": None},
            {"name": "float32类型输入", "shape": (3, 4), "fill_value": 3.14, "dtype": rm.float32},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_full_like.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建输入张量
                input_tensor = rm.ones(case["shape"], dtype=case["dtype"])
                
                # 使用Riemann创建full_like张量
                rm_tensor = rm.full_like(input_tensor, case["fill_value"])
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 检查所有元素是否为指定值
                    expected_values = np.full(case["shape"], case["fill_value"])
                    all_match = np.allclose(rm_tensor.numpy(), expected_values)
                    if not all_match:
                        passed = False
                        error_msg = f"值不匹配，期望{case['fill_value']}"
                    else:
                        # 如果PyTorch可用，与PyTorch进行比较
                        if TORCH_AVAILABLE:
                            torch_dtype = None
                            if case["dtype"] == rm.float32:
                                torch_dtype = torch.float32
                            
                            torch_input = torch.ones(case["shape"], dtype=torch_dtype)
                            torch_tensor = torch.full_like(torch_input, case["fill_value"])
                            passed, error_msg = compare_values(rm_tensor, torch_tensor)
                        else:
                            passed = True
                            error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_eye(self):
        """测试eye函数"""
        test_cases = [
            {"name": "方阵n=3", "n": 3, "m": None, "dtype": None},
            {"name": "矩形矩阵n=2,m=4", "n": 2, "m": 4, "dtype": None},
            {"name": "矩形矩阵n=4,m=2", "n": 4, "m": 2, "dtype": None},
            {"name": "指定float32类型", "n": 3, "m": None, "dtype": rm.float32},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_eye.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建单位矩阵
                if case["m"] is None:
                    rm_tensor = rm.eye(case["n"], dtype=case["dtype"])
                    expected_shape = (case["n"], case["n"])
                else:
                    rm_tensor = rm.eye(case["n"], case["m"], dtype=case["dtype"])
                    expected_shape = (case["n"], case["m"])
                
                # 检查形状
                shape_match = rm_tensor.shape == expected_shape
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{expected_shape}, 得到{rm_tensor.shape}"
                else:
                    # 创建期望的单位矩阵
                    expected_matrix = np.eye(case["n"], case["m"])
                    
                    # 检查对角线是否为1，其他位置是否为0
                    all_match = np.allclose(rm_tensor.numpy(), expected_matrix)
                    if not all_match:
                        passed = False
                        error_msg = "对角线元素不为1或非对角线元素不为0"
                    else:
                        # 如果PyTorch可用，与PyTorch进行比较
                        if TORCH_AVAILABLE:
                            torch_dtype = None
                            if case["dtype"] == rm.float32:
                                torch_dtype = torch.float32
                            
                            if case["m"] is None:
                                torch_tensor = torch.eye(case["n"], dtype=torch_dtype)
                            else:
                                torch_tensor = torch.eye(case["n"], case["m"], dtype=torch_dtype)
                            passed, error_msg = compare_values(rm_tensor, torch_tensor)
                        else:
                            passed = True
                            error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_arange(self):
        """测试arange函数"""
        test_cases = [
            {"name": "只有end参数", "start": None, "end": 5, "step": None},
            {"name": "有start和end参数", "start": 2, "end": 10, "step": None},
            {"name": "有start、end和step参数", "start": 1, "end": 10, "step": 2},
            {"name": "负step参数", "start": 10, "end": 1, "step": -2},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_arange.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建arange张量
                if case["start"] is None:
                    rm_tensor = rm.arange(case["end"])
                elif case["step"] is None:
                    rm_tensor = rm.arange(case["start"], case["end"])
                else:
                    rm_tensor = rm.arange(case["start"], case["end"], case["step"])
                
                # 创建期望的数组
                if case["start"] is None:
                    expected_array = np.arange(case["end"])
                elif case["step"] is None:
                    expected_array = np.arange(case["start"], case["end"])
                else:
                    expected_array = np.arange(case["start"], case["end"], case["step"])
                
                # 检查形状
                shape_match = rm_tensor.shape == expected_array.shape
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{expected_array.shape}, 得到{rm_tensor.shape}"
                else:
                    # 检查值是否匹配
                    all_match = np.array_equal(rm_tensor.numpy(), expected_array)
                    if not all_match:
                        passed = False
                        error_msg = "值不匹配"
                    else:
                        # 如果PyTorch可用，与PyTorch进行比较
                        if TORCH_AVAILABLE:
                            if case["start"] is None:
                                torch_tensor = torch.arange(case["end"])
                            elif case["step"] is None:
                                torch_tensor = torch.arange(case["start"], case["end"])
                            else:
                                torch_tensor = torch.arange(case["start"], case["end"], case["step"])
                            passed, error_msg = compare_values(rm_tensor, torch_tensor)
                        else:
                            passed = True
                            error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_linspace(self):
        """测试linspace函数"""
        test_cases = [
            {"name": "基本用法", "start": 0, "end": 1, "num": 5},
            {"name": "更多点", "start": 0, "end": 10, "num": 11},
            {"name": "降序", "start": 10, "end": 0, "num": 6},
            {"name": "单点", "start": 5, "end": 5, "num": 1},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_linspace.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建linspace张量
                rm_tensor = rm.linspace(case["start"], case["end"], case["num"])
                
                # 创建期望的数组
                expected_array = np.linspace(case["start"], case["end"], case["num"])
                
                # 检查形状
                shape_match = rm_tensor.shape == expected_array.shape
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{expected_array.shape}, 得到{rm_tensor.shape}"
                else:
                    # 检查值是否匹配
                    all_match = np.allclose(rm_tensor.numpy(), expected_array)
                    if not all_match:
                        passed = False
                        error_msg = "值不匹配"
                    else:
                        # 如果PyTorch可用，与PyTorch进行比较
                        if TORCH_AVAILABLE:
                            torch_tensor = torch.linspace(case["start"], case["end"], case["num"])
                            passed, error_msg = compare_values(rm_tensor, torch_tensor)
                        else:
                            passed = True
                            error_msg = ""
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  错误: {error_msg}")
                
                self.assertTrue(passed, f"测试用例'{case_name}'失败: {error_msg}")
                
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
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行张量创建函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTensorCreationFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)