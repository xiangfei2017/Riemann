import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    # 从rm.cuda获取cupy引用和CUDA可用性
    CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
    cp = rm.cuda.cp
except ImportError as e:
    print(f"无法导入riemann模块: {e}")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的张量值判断函数")
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
def compare_values(rm_result, expected_result, atol=1e-6, rtol=1e-6):
    """比较Riemann结果和预期结果是否接近或相等"""
    # 处理None值的情况
    if rm_result is None and expected_result is None:
        return True
    if rm_result is None or expected_result is None:
        return False
    
    # 处理布尔值的情况
    if isinstance(rm_result, bool) and isinstance(expected_result, bool):
        return rm_result == expected_result
    
    # 处理元组/列表的情况
    if isinstance(rm_result, (list, tuple)) and isinstance(expected_result, (list, tuple)):
        if len(rm_result) != len(expected_result):
            return False
        
        all_passed = True
        for i, (r, e) in enumerate(zip(rm_result, expected_result)):
            if not compare_values(r, e, atol, rtol):
                all_passed = False
                break
        
        return all_passed
    
    # 处理张量和numpy数组的情况
    try:
        # 提取Riemann结果数据
        if hasattr(rm_result, 'is_cuda') and rm_result.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            rm_data = rm_result.detach().cpu().numpy()
        elif hasattr(rm_result, 'detach'):
            rm_data = rm_result.detach().numpy()
        elif hasattr(rm_result, 'data'):
            rm_data = rm_result.data
            # 处理CuPy数组
            if hasattr(rm_data, 'get'):
                rm_data = rm_data.get()
        elif hasattr(rm_result, 'numpy'):
            rm_data = rm_result.numpy()
        else:
            rm_data = rm_result
        
        # 转换预期结果为numpy数组
        if isinstance(expected_result, (list, tuple)):
            expected_data = np.array(expected_result)
        elif hasattr(expected_result, 'is_cuda') and expected_result.is_cuda:
            expected_data = expected_result.detach().cpu().numpy()
        elif hasattr(expected_result, 'detach'):
            expected_data = expected_result.detach().numpy()
        elif hasattr(expected_result, 'data'):
            expected_data = expected_result.data
            # 处理CuPy数组
            if hasattr(expected_data, 'get'):
                expected_data = expected_data.get()
        elif hasattr(expected_result, 'numpy'):
            expected_data = expected_result.numpy()
        else:
            expected_data = np.array(expected_result)
        
        # 比较形状
        if np.shape(rm_data) != np.shape(expected_data):
            return False
        
        # 比较数据
        if np.issubdtype(np.array(rm_data).dtype, np.floating):
            return np.allclose(rm_data, expected_data, rtol=rtol, atol=atol)
        else:
            return np.array_equal(rm_data, expected_data)
    except Exception as e:
        print(f"比较值转换错误: {e}")
        return False

# 测试张量值判断函数类
class TestTensorValueFunctions(unittest.TestCase):
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
    
    def test_all(self):
        """测试all函数"""
        test_cases = [
            {"name": "全True张量", "data": np.ones((3, 4), dtype=bool), "dim": None, "keepdim": False, "expected": True},
            {"name": "包含False的张量", "data": np.array([[True, True], [True, False]]), "dim": None, "keepdim": False, "expected": False},
            {"name": "沿指定维度的all", "data": np.array([[True, True], [True, True]]), "dim": 1, "keepdim": False, "expected": [True, True]},
            {"name": "沿指定维度的all(包含False)", "data": np.array([[True, True], [True, False]]), "dim": 1, "keepdim": False, "expected": [True, False]},
            {"name": "保持维度的all", "data": np.ones((3, 4), dtype=bool), "dim": 1, "keepdim": True, "expected": np.ones((3, 1), dtype=bool)},
            {"name": "空维度元组", "data": np.ones((3, 4), dtype=bool), "dim": (), "keepdim": False, "expected": True},
            {"name": "多维度all", "data": np.ones((2, 3, 4), dtype=bool), "dim": (0, 1), "keepdim": False, "expected": np.ones(4, dtype=bool)},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"all - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据 - 只有浮点型张量才设置requires_grad=True
                    requires_grad = isinstance(case["data"], np.ndarray) and np.issubdtype(case["data"].dtype, np.floating)
                    rm_x = rm.tensor(case["data"], requires_grad=requires_grad, device=device)
                    
                    # 测试all函数
                    rm_result = rm_x.all(dim=case["dim"], keepdim=case["keepdim"])
                    
                    # 如果没有指定维度，结果应该是标量
                    if case["dim"] is None or case["dim"] == ():
                        rm_result_value = rm_result.item()
                    else:
                        rm_result_value = rm_result
                    
                    # 比较结果
                    passed = compare_values(rm_result_value, case["expected"])
                    
                    # 检查数据类型
                    type_passed = isinstance(rm_result, rm.TN) and rm_result.dtype == np.bool_
                    passed = passed and type_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  预期结果: {case['expected']}")
                            print(f"  实际结果: {rm_result_value}")
                            print(f"  数据类型: {rm_result.dtype}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"all测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False)
                        print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time_taken:.4f}秒)")
                        print(f"  错误信息: {str(e)}")
                    self.fail(f"all测试异常: {case_name}, 错误: {str(e)}")
    
    def test_any(self):
        """测试any函数"""
        test_cases = [
            {"name": "全False张量", "data": np.zeros((3, 4), dtype=bool), "dim": None, "keepdim": False, "expected": False},
            {"name": "包含True的张量", "data": np.array([[False, False], [False, True]]), "dim": None, "keepdim": False, "expected": True},
            {"name": "沿指定维度的any", "data": np.array([[False, False], [True, False]]), "dim": 1, "keepdim": False, "expected": [False, True]},
            {"name": "沿指定维度的any(全False)", "data": np.zeros((3, 4), dtype=bool), "dim": 1, "keepdim": False, "expected": np.zeros(3, dtype=bool)},
            {"name": "保持维度的any", "data": np.array([[False, True], [False, False]]), "dim": 1, "keepdim": True, "expected": np.array([[True], [False]])},
            {"name": "空维度元组", "data": np.array([[False, True]]), "dim": (), "keepdim": False, "expected": True},
            {"name": "多维度any", "data": np.array([[[False, True], [False, False]], [[False, False], [False, False]]]), "dim": (0, 1), "keepdim": False, "expected": np.array([False, True])},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"any - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据 - 只有浮点型张量才设置requires_grad=True
                    requires_grad = isinstance(case["data"], np.ndarray) and np.issubdtype(case["data"].dtype, np.floating)
                    rm_x = rm.tensor(case["data"], requires_grad=requires_grad, device=device)
                    
                    # 测试any函数
                    rm_result = rm_x.any(dim=case["dim"], keepdim=case["keepdim"])
                    
                    # 如果没有指定维度，结果应该是标量
                    if case["dim"] is None or case["dim"] == ():
                        rm_result_value = rm_result.item()
                    else:
                        rm_result_value = rm_result
                    
                    # 比较结果
                    passed = compare_values(rm_result_value, case["expected"])
                    
                    # 检查数据类型
                    type_passed = isinstance(rm_result, rm.TN) and rm_result.dtype == np.bool_
                    passed = passed and type_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  预期结果: {case['expected']}")
                            print(f"  实际结果: {rm_result_value}")
                            print(f"  数据类型: {rm_result.dtype}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"any测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False)
                        print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time_taken:.4f}秒)")
                        print(f"  错误信息: {str(e)}")
                    self.fail(f"any测试异常: {case_name}, 错误: {str(e)}")
    
    def test_equal(self):
        """测试equal函数"""
        test_cases = [
            {"name": "相同张量", "data1": np.array([1, 2, 3]), "data2": np.array([1, 2, 3]), "expected": True},
            {"name": "不同元素的张量", "data1": np.array([1, 2, 3]), "data2": np.array([1, 2, 4]), "expected": False},
            {"name": "不同形状的张量", "data1": np.array([1, 2, 3]), "data2": np.array([[1, 2], [3, 4]]), "expected": False},
            {"name": "空张量", "data1": np.array([]), "data2": np.array([]), "expected": True},
            {"name": "浮点张量", "data1": np.array([1.0, 2.0, 3.0]), "data2": np.array([1.0, 2.0, 3.0]), "expected": True},
            {"name": "布尔张量", "data1": np.array([True, False, True]), "data2": np.array([True, False, True]), "expected": True},
            {"name": "多维张量", "data1": np.ones((2, 3, 4)), "data2": np.ones((2, 3, 4)), "expected": True},
        ]
        
        for case in test_cases:
            case_name = f"equal - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据 - 只有浮点型张量才设置requires_grad=True
                requires_grad1 = isinstance(case["data1"], np.ndarray) and np.issubdtype(case["data1"].dtype, np.floating)
                requires_grad2 = isinstance(case["data2"], np.ndarray) and np.issubdtype(case["data2"].dtype, np.floating)
                rm_x1 = rm.tensor(case["data1"], requires_grad=requires_grad1)
                rm_x2 = rm.tensor(case["data2"], requires_grad=requires_grad2)
                
                # 测试equal函数
                rm_result = rm.equal(rm_x1, rm_x2)
                
                # 比较结果
                passed = compare_values(rm_result, case["expected"])
                
                # 检查结果是否为布尔类型
                type_passed = isinstance(rm_result, bool)
                passed = passed and type_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  预期结果: {case['expected']}")
                        print(f"  实际结果: {rm_result}")
                        print(f"  数据类型: {type(rm_result).__name__}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"equal测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False)
                    print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time_taken:.4f}秒)")
                    print(f"  错误信息: {str(e)}")
                self.fail(f"equal测试异常: {case_name}, 错误: {str(e)}")
    
    def test_not_equal(self):
        """测试not_equal函数"""
        test_cases = [
            {"name": "相同张量", "data1": np.array([1, 2, 3]), "data2": np.array([1, 2, 3]), "expected": False},
            {"name": "不同元素的张量", "data1": np.array([1, 2, 3]), "data2": np.array([1, 2, 4]), "expected": True},
            {"name": "不同形状的张量", "data1": np.array([1, 2, 3]), "data2": np.array([[1, 2], [3, 4]]), "expected": True},
            {"name": "空张量", "data1": np.array([]), "data2": np.array([]), "expected": False},
            {"name": "浮点张量", "data1": np.array([1.0, 2.0, 3.0]), "data2": np.array([1.0, 2.0, 4.0]), "expected": True},
            {"name": "布尔张量", "data1": np.array([True, False, True]), "data2": np.array([True, True, True]), "expected": True},
            {"name": "多维张量", "data1": np.ones((2, 3, 4)), "data2": np.zeros((2, 3, 4)), "expected": True},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"not_equal - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据 - 仅对浮点型张量设置requires_grad
                    requires_grad1 = isinstance(case["data1"].flatten()[0], (np.float16, np.float32, np.float64)) if len(case["data1"].flatten()) > 0 else False
                    requires_grad2 = isinstance(case["data2"].flatten()[0], (np.float16, np.float32, np.float64)) if len(case["data2"].flatten()) > 0 else False
                    rm_x1 = rm.tensor(case["data1"], requires_grad=requires_grad1, device=device)
                    rm_x2 = rm.tensor(case["data2"], requires_grad=requires_grad2, device=device)
                    
                    # 测试not_equal函数
                    rm_result = rm.not_equal(rm_x1, rm_x2)
                    
                    # 比较结果
                    passed = compare_values(rm_result, case["expected"])
                    
                    # 检查结果是否为布尔类型
                    type_passed = isinstance(rm_result, bool)
                    passed = passed and type_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  预期结果: {case['expected']}")
                            print(f"  实际结果: {rm_result}")
                            print(f"  数据类型: {type(rm_result).__name__}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"not_equal测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False)
                        print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time_taken:.4f}秒)")
                        print(f"  错误信息: {str(e)}")
                    self.fail(f"not_equal测试异常: {case_name}, 错误: {str(e)}")
    
    def test_allclose(self):
        """测试allclose函数"""
        test_cases = [
            {"name": "完全相同的张量", "data1": np.array([1, 2, 3]), "data2": np.array([1, 2, 3]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": False, "expected": True},
            {"name": "接近相等的浮点张量", "data1": np.array([1.0, 2.0, 3.0]), "data2": np.array([1.0, 2.0, 3.0000001]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": False, "expected": True},
            {"name": "不接近的张量", "data1": np.array([1.0, 2.0, 3.0]), "data2": np.array([1.0, 2.0, 3.1]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": False, "expected": False},
            {"name": "不同形状的张量", "data1": np.array([1, 2, 3]), "data2": np.array([[1, 2], [3, 4]]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": False, "expected": False},
            {"name": "大容差测试", "data1": np.array([1.0, 2.0, 3.0]), "data2": np.array([1.0, 2.0, 3.01]), "rtol": 1e-2, "atol": 1e-2, "equal_nan": False, "expected": True},
            {"name": "零张量", "data1": np.zeros(3), "data2": np.zeros(3), "rtol": 1e-5, "atol": 1e-8, "equal_nan": False, "expected": True},
            {"name": "接近零的张量", "data1": np.zeros(3), "data2": np.array([1e-9, 1e-9, 1e-9]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": False, "expected": True},
            # 添加equal_nan相关测试用例
            {"name": "包含相同位置NaN (equal_nan=True)", "data1": np.array([1.0, np.nan, 3.0]), "data2": np.array([1.0, np.nan, 3.0]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": True, "expected": True},
            {"name": "包含相同位置NaN (equal_nan=False)", "data1": np.array([1.0, np.nan, 3.0]), "data2": np.array([1.0, np.nan, 3.0]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": False, "expected": False},
            {"name": "包含不同位置NaN (equal_nan=True)", "data1": np.array([1.0, np.nan, 3.0]), "data2": np.array([np.nan, 2.0, 3.0]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": True, "expected": False},
            {"name": "混合值与NaN (equal_nan=True)", "data1": np.array([1.0, np.nan, 3.0]), "data2": np.array([1.0, np.nan, 3.0000001]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": True, "expected": True},
            {"name": "混合值与NaN (equal_nan=False)", "data1": np.array([1.0, np.nan, 3.0]), "data2": np.array([1.0, np.nan, 3.0]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": False, "expected": False},
            {"name": "全部NaN (equal_nan=True)", "data1": np.array([np.nan, np.nan]), "data2": np.array([np.nan, np.nan]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": True, "expected": True},
            {"name": "全部NaN (equal_nan=False)", "data1": np.array([np.nan, np.nan]), "data2": np.array([np.nan, np.nan]), "rtol": 1e-5, "atol": 1e-8, "equal_nan": False, "expected": False},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"allclose - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据 - 仅对浮点型张量设置requires_grad
                    requires_grad1 = isinstance(case["data1"].flatten()[0], (np.float16, np.float32, np.float64)) if len(case["data1"].flatten()) > 0 else False
                    requires_grad2 = isinstance(case["data2"].flatten()[0], (np.float16, np.float32, np.float64)) if len(case["data2"].flatten()) > 0 else False
                    rm_x1 = rm.tensor(case["data1"], requires_grad=requires_grad1, device=device)
                    rm_x2 = rm.tensor(case["data2"], requires_grad=requires_grad2, device=device)
                    
                    # 测试allclose函数
                    rm_result = rm.allclose(rm_x1, rm_x2, rtol=case["rtol"], atol=case["atol"], equal_nan=case["equal_nan"])
                    
                    # 比较结果
                    passed = compare_values(rm_result, case["expected"])
                    
                    # 检查结果是否为布尔类型
                    type_passed = isinstance(rm_result, bool)
                    passed = passed and type_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  预期结果: {case['expected']}")
                            print(f"  实际结果: {rm_result}")
                            print(f"  数据类型: {type(rm_result).__name__}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"allclose测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False)
                        print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time_taken:.4f}秒)")
                        print(f"  错误信息: {str(e)}")
                    self.fail(f"allclose测试异常: {case_name}, 错误: {str(e)}")
    
    def test_unique(self):
        """测试unique函数"""
        test_cases = [
            {"name": "基本unique", "data": np.array([1, 2, 3, 2, 1, 4]), "sorted": True, "return_inverse": False, "return_counts": False, "return_indices": False, "expected": [1, 2, 3, 4]},
            {"name": "带逆索引", "data": np.array([1, 2, 3, 2, 1, 4]), "sorted": True, "return_inverse": True, "return_counts": False, "return_indices": False, "expected": ([1, 2, 3, 4], [0, 1, 2, 1, 0, 3])},
            {"name": "带计数", "data": np.array([1, 2, 3, 2, 1, 4]), "sorted": True, "return_inverse": False, "return_counts": True, "return_indices": False, "expected": ([1, 2, 3, 4], [2, 2, 1, 1])},
            {"name": "带首次出现索引", "data": np.array([1, 2, 3, 2, 1, 4]), "sorted": True, "return_inverse": False, "return_counts": False, "return_indices": True, "expected": ([1, 2, 3, 4], [0, 1, 2, 5])},
            {"name": "全部返回", "data": np.array([1, 2, 3, 2, 1, 4]), "sorted": True, "return_inverse": True, "return_counts": True, "return_indices": True, "expected": ([1, 2, 3, 4], [0, 1, 2, 1, 0, 3], [0, 1, 2, 5], [2, 2, 1, 1])},
            {"name": "非排序", "data": np.array([3, 1, 2, 2, 1, 4]), "sorted": False, "return_inverse": False, "return_counts": False, "return_indices": False, "expected": np.unique(np.array([3, 1, 2, 2, 1, 4]), return_index=True)[0]},
            {"name": "多维张量", "data": np.array([[1, 2], [3, 2], [1, 4]]), "sorted": True, "return_inverse": False, "return_counts": False, "return_indices": False, "expected": [1, 2, 3, 4]},
            {"name": "浮点张量", "data": np.array([1.0, 2.0, 3.0, 2.0, 1.0]), "sorted": True, "return_inverse": False, "return_counts": False, "return_indices": False, "expected": [1.0, 2.0, 3.0]},
            {"name": "布尔张量", "data": np.array([True, False, True, False]), "sorted": True, "return_inverse": False, "return_counts": False, "return_indices": False, "expected": [False, True]},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"unique - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据 - 仅对浮点型张量设置requires_grad
                    requires_grad = isinstance(case["data"].flatten()[0], (np.float16, np.float32, np.float64)) if len(case["data"].flatten()) > 0 else False
                    rm_x = rm.tensor(case["data"], requires_grad=requires_grad, device=device)
                
                    # 测试unique函数
                    rm_result = rm.unique(
                        rm_x, 
                        sorted=case["sorted"],
                        return_inverse=case["return_inverse"],
                        return_counts=case["return_counts"],
                        return_indices=case["return_indices"]
                    )
                    
                    # 比较结果
                    passed = compare_values(rm_result, case["expected"])
                    
                    # 检查返回类型
                    if case["return_inverse"] or case["return_counts"] or case["return_indices"]:
                        # 应该返回元组
                        type_passed = isinstance(rm_result, tuple)
                        # 检查元组中每个元素是否为TN类型
                        for item in rm_result:
                            type_passed = type_passed and isinstance(item, rm.TN)
                    else:
                        # 应该返回单个TN对象
                        type_passed = isinstance(rm_result, rm.TN)
                    
                    passed = passed and type_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  预期结果类型: {type(case['expected']).__name__}")
                            print(f"  实际结果类型: {type(rm_result).__name__}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"unique测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False)
                        print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time_taken:.4f}秒)")
                        print(f"  错误信息: {str(e)}")
                    self.fail(f"unique测试异常: {case_name}, 错误: {str(e)}")
    
    def test_nonzero(self):
        """测试nonzero函数"""
        test_cases = [
            {"name": "基本二维张量", "data": np.array([[1, 0, 2], [0, 3, 0]]), "as_tuple": False, "expected": [[0, 0], [0, 2], [1, 1]]},
            {"name": "基本二维张量(as_tuple=True)", "data": np.array([[1, 0, 2], [0, 3, 0]]), "as_tuple": True, "expected": ([0, 0, 1], [0, 2, 1])},
            {"name": "全零张量", "data": np.zeros((3, 4)), "as_tuple": False, "expected": np.array([], dtype=np.int64).reshape(0, 2)},
            {"name": "全零张量(as_tuple=True)", "data": np.zeros((3, 4)), "as_tuple": True, "expected": (np.array([]), np.array([]))},
            {"name": "一维张量", "data": np.array([1, 0, 3, 0, 5]), "as_tuple": False, "expected": [[0], [2], [4]]},
            {"name": "一维张量(as_tuple=True)", "data": np.array([1, 0, 3, 0, 5]), "as_tuple": True, "expected": ([0, 2, 4],)},
            {"name": "三维张量", "data": np.array([[[1, 0], [0, 2]], [[3, 0], [0, 4]]]), "as_tuple": False, "expected": [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]]},
            {"name": "三维张量(as_tuple=True)", "data": np.array([[[1, 0], [0, 2]], [[3, 0], [0, 4]]]), "as_tuple": True, "expected": ([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1])},
            {"name": "复数张量", "data": np.array([1+1j, 0+0j, 3+2j, 0+0j]), "as_tuple": False, "expected": [[0], [2]]},
            {"name": "复数张量(as_tuple=True)", "data": np.array([1+1j, 0+0j, 3+2j, 0+0j]), "as_tuple": True, "expected": ([0, 2],)},
            {"name": "浮点张量", "data": np.array([1.5, 0.0, 2.7, 0.0, 3.1]), "as_tuple": False, "expected": [[0], [2], [4]]},
            {"name": "布尔张量", "data": np.array([True, False, True, False]), "as_tuple": False, "expected": [[0], [2]]},
            {"name": "整数张量", "data": np.array([1, 0, 3, 0, 5]), "as_tuple": False, "expected": [[0], [2], [4]]},
            {"name": "单元素非零张量", "data": np.array([[0, 0, 0], [0, 5, 0]]), "as_tuple": False, "expected": [[1, 1]]},
            {"name": "单元素非零张量(as_tuple=True)", "data": np.array([[0, 0, 0], [0, 5, 0]]), "as_tuple": True, "expected": ([1], [1])},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"nonzero - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据 - 只有浮点型张量才设置requires_grad=True
                    requires_grad = isinstance(case["data"], np.ndarray) and np.issubdtype(case["data"].dtype, np.floating)
                    rm_x = rm.tensor(case["data"], requires_grad=requires_grad, device=device)
                    
                    # 测试nonzero函数
                    rm_result = rm.nonzero(rm_x, as_tuple=case["as_tuple"])
                    
                    # 比较结果
                    passed = compare_values(rm_result, case["expected"])
                    
                    # 检查返回类型
                    if case["as_tuple"]:
                        # as_tuple=True时应该返回元组
                        type_passed = isinstance(rm_result, tuple)
                        # 检查元组中每个元素是否为TN类型
                        for item in rm_result:
                            type_passed = type_passed and isinstance(item, rm.TN)
                    else:
                        # as_tuple=False时应该返回TN类型
                        type_passed = isinstance(rm_result, rm.TN)
                    
                    passed = passed and type_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  预期结果: {case['expected']}")
                            print(f"  实际结果: {rm_result}")
                            print(f"  返回类型: {type(rm_result).__name__}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"nonzero测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False)
                        print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time_taken:.4f}秒)")
                        print(f"  错误信息: {str(e)}")
                    self.fail(f"nonzero测试异常: {case_name}, 错误: {str(e)}")

if __name__ == '__main__':
    # 标记为独立脚本运行
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    # 打印测试信息
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行张量值判断函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 运行测试
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTensorValueFunctions)
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计
    stats.print_summary()
    
    # 退出程序，根据测试结果返回相应的退出码
    sys.exit(0 if result.wasSuccessful() else 1)