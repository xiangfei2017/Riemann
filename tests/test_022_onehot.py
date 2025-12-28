import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.nn.functional import one_hot
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的one_hot函数")
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
    
    # 转换为numpy数组
    rm_data = rm_result.data if hasattr(rm_result, 'data') else rm_result
    torch_data = torch_result.detach().cpu().numpy()
    
    # 处理形状不匹配的情况
    try:
        # 增加容差参数
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

class TestOneHotFunctions(unittest.TestCase):
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
    
    def test_basic_1d_input(self):
        """测试基本的一维输入"""
        case_name = "一维输入one_hot测试"
        start_time = time.time()
        try:
            # 创建测试数据
            np_target = np.array([0, 1, 2, 0], dtype=np.int64)
            num_classes = 3
            
            # Riemann实现
            rm_target = rm.tensor(np_target)
            rm_result = one_hot(rm_target, num_classes)
            
            # PyTorch实现
            torch_result = None
            if TORCH_AVAILABLE:
                torch_target = torch.tensor(np_target)
                torch_result = torch.nn.functional.one_hot(torch_target, num_classes).float()
            
            # 比较结果
            passed = True
            if TORCH_AVAILABLE:
                passed = compare_values(rm_result, torch_result)
            
            # 验证形状
            expected_shape = (np_target.shape[0], num_classes)
            self.assertEqual(rm_result.shape, expected_shape)
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
            # 断言确保测试通过
            self.assertTrue(passed, f"一维输入one_hot测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_2d_input(self):
        """测试二维输入"""
        case_name = "二维输入one_hot测试"
        start_time = time.time()
        try:
            # 创建测试数据
            np_target = np.array([[0, 1], [2, 0]], dtype=np.int64)
            num_classes = 3
            
            # Riemann实现
            rm_target = rm.tensor(np_target)
            rm_result = one_hot(rm_target, num_classes)
            
            # PyTorch实现
            torch_result = None
            if TORCH_AVAILABLE:
                torch_target = torch.tensor(np_target)
                torch_result = torch.nn.functional.one_hot(torch_target, num_classes).float()
            
            # 比较结果
            passed = True
            if TORCH_AVAILABLE:
                passed = compare_values(rm_result, torch_result)
            
            # 验证形状
            expected_shape = np_target.shape + (num_classes,)
            self.assertEqual(rm_result.shape, expected_shape)
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
            # 断言确保测试通过
            self.assertTrue(passed, f"二维输入one_hot测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_3d_input(self):
        """测试三维输入"""
        case_name = "三维输入one_hot测试"
        start_time = time.time()
        try:
            # 创建测试数据
            np_target = np.array([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=np.int64)
            num_classes = 3
            
            # Riemann实现
            rm_target = rm.tensor(np_target)
            rm_result = one_hot(rm_target, num_classes)
            
            # PyTorch实现
            torch_result = None
            if TORCH_AVAILABLE:
                torch_target = torch.tensor(np_target)
                torch_result = torch.nn.functional.one_hot(torch_target, num_classes).float()
            
            # 比较结果
            passed = True
            if TORCH_AVAILABLE:
                passed = compare_values(rm_result, torch_result)
            
            # 验证形状
            expected_shape = np_target.shape + (num_classes,)
            self.assertEqual(rm_result.shape, expected_shape)
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
            # 断言确保测试通过
            self.assertTrue(passed, f"三维输入one_hot测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_more_classes_than_indices(self):
        """测试类别数大于实际索引值的情况"""
        case_name = "类别数大于索引值one_hot测试"
        start_time = time.time()
        try:
            # 创建测试数据
            np_target = np.array([0, 1, 0], dtype=np.int64)
            num_classes = 5  # 类别数大于实际索引值
            
            # Riemann实现
            rm_target = rm.tensor(np_target)
            rm_result = one_hot(rm_target, num_classes)
            
            # PyTorch实现
            torch_result = None
            if TORCH_AVAILABLE:
                torch_target = torch.tensor(np_target)
                torch_result = torch.nn.functional.one_hot(torch_target, num_classes).float()
            
            # 比较结果
            passed = True
            if TORCH_AVAILABLE:
                passed = compare_values(rm_result, torch_result)
            
            # 验证形状
            expected_shape = (np_target.shape[0], num_classes)
            self.assertEqual(rm_result.shape, expected_shape)
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
            # 断言确保测试通过
            self.assertTrue(passed, f"类别数大于索引值one_hot测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_edge_case_empty_input(self):
        """测试空输入的边缘情况"""
        case_name = "空输入one_hot测试"
        start_time = time.time()
        try:
            # 创建测试数据
            np_target = np.array([], dtype=np.int64)
            num_classes = 3
            
            # Riemann实现
            rm_target = rm.tensor(np_target)
            rm_result = one_hot(rm_target, num_classes)
            
            # PyTorch实现
            torch_result = None
            if TORCH_AVAILABLE:
                torch_target = torch.tensor(np_target)
                torch_result = torch.nn.functional.one_hot(torch_target, num_classes).float()
            
            # 比较结果
            passed = True
            if TORCH_AVAILABLE:
                passed = compare_values(rm_result, torch_result)
            
            # 验证形状
            expected_shape = (0, num_classes)
            self.assertEqual(rm_result.shape, expected_shape)
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
            # 断言确保测试通过
            self.assertTrue(passed, f"空输入one_hot测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_edge_case_single_element(self):
        """测试单元素输入的边缘情况"""
        case_name = "单元素输入one_hot测试"
        start_time = time.time()
        try:
            # 创建测试数据
            np_target = np.array([2], dtype=np.int64)
            num_classes = 5
            
            # Riemann实现
            rm_target = rm.tensor(np_target)
            rm_result = one_hot(rm_target, num_classes)
            
            # PyTorch实现
            torch_result = None
            if TORCH_AVAILABLE:
                torch_target = torch.tensor(np_target)
                torch_result = torch.nn.functional.one_hot(torch_target, num_classes).float()
            
            # 比较结果
            passed = True
            if TORCH_AVAILABLE:
                passed = compare_values(rm_result, torch_result)
            
            # 验证形状
            expected_shape = (1, num_classes)
            self.assertEqual(rm_result.shape, expected_shape)
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
            # 断言确保测试通过
            self.assertTrue(passed, f"单元素输入one_hot测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_large_class_count(self):
        """测试大类别数的情况"""
        case_name = "大类别数one_hot测试"
        start_time = time.time()
        try:
            # 创建测试数据
            np_target = np.random.randint(0, 100, size=(10,), dtype=np.int64)
            num_classes = 1000
            
            # Riemann实现
            rm_target = rm.tensor(np_target)
            rm_result = one_hot(rm_target, num_classes)
            
            # PyTorch实现
            torch_result = None
            if TORCH_AVAILABLE:
                torch_target = torch.tensor(np_target)
                torch_result = torch.nn.functional.one_hot(torch_target, num_classes).float()
            
            # 比较结果
            passed = True
            if TORCH_AVAILABLE:
                passed = compare_values(rm_result, torch_result)
            
            # 验证形状
            expected_shape = (np_target.shape[0], num_classes)
            self.assertEqual(rm_result.shape, expected_shape)
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
            # 断言确保测试通过
            self.assertTrue(passed, f"大类别数one_hot测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_negative_indices(self):
        """测试负索引的情况（应抛出异常）"""
        case_name = "负索引one_hot测试"
        start_time = time.time()
        try:
            # 创建测试数据
            np_target = np.array([-1, 0, 1], dtype=np.int64)
            num_classes = 3
            
            # Riemann实现 - 应该抛出异常
            rm_target = rm.tensor(np_target)
            
            # 检查Riemann是否抛出异常
            with self.assertRaises(RuntimeError) as context_rm:
                one_hot(rm_target, num_classes)
            
            # 验证异常消息
            self.assertIn("Class values must be non-negative", str(context_rm.exception))
            
            # 如果PyTorch可用，也测试PyTorch的行为
            if TORCH_AVAILABLE:
                torch_target = torch.tensor(np_target)
                
                # 检查PyTorch是否抛出异常
                with self.assertRaises(RuntimeError) as context_torch:
                    torch.nn.functional.one_hot(torch_target, num_classes).float()
                
                # 验证异常消息
                self.assertIn("Class values must be non-negative", str(context_torch.exception))
            
            passed = True
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
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
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行one_hot函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestOneHotFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)