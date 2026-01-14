import sys
import time
import numpy as np
import unittest
import riemann as rm
import riemann.cuda as cuda

# 检查CUDA是否可用，如果不可用则设置标志
CUDA_AVAILABLE = cuda.is_available()

# 定义色彩常量
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

def clear_screen():
    """清屏函数，兼容Windows和Unix系统"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

# 全局统计实例
stats = StatisticsCollector()

# 判断是否作为独立脚本运行
IS_RUNNING_AS_SCRIPT = __name__ == "__main__"


class CUDAUnitTest(unittest.TestCase):
    """使用 unittest 框架的 CUDA 测试类，用于 pytest 发现"""
    
    def setUp(self):
        """测试前的准备工作"""
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA is not available")
        # 获取当前测试方法名
        self.current_test_name = self._testMethodName
        # 如果是独立脚本运行，则开始记录函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")
    
    def tearDown(self):
        """测试后的清理工作"""
        # 如果是独立脚本运行，则结束函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_context_management(self):
        """测试上下文管理功能"""
        try:
            # 定义上下文管理的子用例
            context_test_cases = [
                {
                    "name": "test_nested_context",
                    "description": "测试嵌套上下文管理",
                    "test_func": lambda: self._test_nested_context()
                },
                {
                    "name": "test_multi_device_context",
                    "description": "测试多设备上下文管理",
                    "test_func": lambda: self._test_multi_device_context()
                },
                {
                    "name": "test_context_exception_handling",
                    "description": "测试上下文异常处理",
                    "test_func": lambda: self._test_context_exception_handling()
                },
                {
                    "name": "test_context_device_priority",
                    "description": "测试上下文优先级高于默认设备设置",
                    "test_func": lambda: self._test_context_device_priority()
                }
            ]
            
            # 运行所有上下文管理子用例
            for case in context_test_cases:
                start_time = time.time()
                try:
                    case["test_func"]()
                    passed = True
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case["name"], passed)
                        print(f"测试用例: {case['description']} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                except Exception as e:
                    passed = False
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case["name"], passed, [str(e)])
                        print(f"测试用例: {case['description']} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                    raise
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("test_context_management", True)
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("test_context_management", False, [str(e)])
            raise
    
    def _test_nested_context(self):
        """测试嵌套上下文管理（子用例）"""
        with cuda.Device('cuda:0'):
            self.assertTrue(cuda.is_in_cuda_context())
            
            with cuda.Device('cpu'):
                self.assertFalse(cuda.is_in_cuda_context())
            
            self.assertTrue(cuda.is_in_cuda_context())
    
    def _test_multi_device_context(self):
        """测试多设备上下文管理（子用例）"""
        num_devices = cuda.device_count()
        if num_devices >= 2:
            with cuda.Device(0):
                tensor0 = rm.tensor([1, 2, 3])
                self.assertEqual(tensor0.device.index, 0)
        
            with cuda.Device(1):
                tensor1 = rm.tensor([4, 5, 6])
                self.assertEqual(tensor1.device.index, 1)
    
    def _test_context_exception_handling(self):
        """测试上下文异常处理（子用例）"""
        original_context = cuda.is_in_cuda_context()
        
        try:
            with cuda.Device('cuda:0'):
                raise ValueError("测试异常")
        except:
            pass
        
        self.assertEqual(cuda.is_in_cuda_context(), original_context)
    
    def _test_large_tensor_creation(self):
        """测试大规模张量创建（子用例）"""
        with cuda.Device('cuda:0'):
            tensor = rm.tensor(np.random.randn(1000, 1000))
            self.assertEqual(tensor.shape, (1000, 1000))
            self.assertEqual(tensor.device.type, 'cuda')

    def test_tensor_to_functions(self):
        """测试张量 to() 函数功能"""
        try:
            # 定义 to() 函数的子用例
            to_test_cases = [
                {
                    "name": "test_tensor_to_device_migration",
                    "description": "测试 TN 张量 to() 函数在 CPU 和 CUDA 之间的迁移",
                    "test_func": lambda: self._test_tensor_to_device_migration()
                },
                {
                    "name": "test_tensor_to_grad_tracking",
                    "description": "测试 TN 张量 to() 迁移后的梯度跟踪功能",
                    "test_func": lambda: self._test_tensor_to_grad_tracking()
                },
                {
                    "name": "test_tensor_to_complex_migration",
                    "description": "测试复数张量的 to() 迁移功能",
                    "test_func": lambda: self._test_tensor_to_complex_migration()
                },
                {
                    "name": "test_tensor_to_parameter_combinations",
                    "description": "测试 TN 张量 to() 函数中 device 参数和 dtype 的各种取值、各种顺序组合",
                    "test_func": lambda: self._test_tensor_to_parameter_combinations()
                },
                {
                    "name": "test_tensor_cpu_function",
                    "description": "测试 TN 张量 cpu() 函数功能",
                    "test_func": lambda: self._test_tensor_cpu_function()
                },
                {
                    "name": "test_tensor_cuda_function",
                    "description": "测试 TN 张量 cuda() 函数功能及不同参数取值",
                    "test_func": lambda: self._test_tensor_cuda_function()
                }
            ]
            
            # 运行所有 to() 函数子用例
            for case in to_test_cases:
                start_time = time.time()
                try:
                    case["test_func"]()
                    passed = True
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case["name"], passed)
                        print(f"测试用例: {case['description']} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                except Exception as e:
                    passed = False
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case["name"], passed, [str(e)])
                        print(f"测试用例: {case['description']} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                    raise
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("test_tensor_to_functions", True)
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("test_tensor_to_functions", False, [str(e)])
            raise
    
    def _test_tensor_to_device_migration(self):
        """测试 TN 张量 to() 函数在 CPU 和 CUDA 之间的迁移（子用例）"""
        # CPU -> CUDA 迁移测试
        cpu_tensor = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
        cuda_tensor = cpu_tensor.to('cuda:0')
        
        self.assertEqual(cuda_tensor.shape, cpu_tensor.shape)
        self.assertEqual(cuda_tensor.dtype, cpu_tensor.dtype)
        self.assertEqual(cuda_tensor.device.type, 'cuda')
        self.assertEqual(cuda_tensor.device.index, 0)
        self.assertEqual(cuda_tensor.requires_grad, cpu_tensor.requires_grad)
        # is_leaf 属性可能在迁移后变化，因为 to() 会创建新张量
        # self.assertEqual(cuda_tensor.is_leaf, cpu_tensor.is_leaf)
        self.assertTrue(np.allclose(cpu_tensor.data, cuda_tensor.data))
        
        # CUDA -> CPU 迁移测试
        with cuda.Device('cuda:0'):
            cuda_tensor = rm.tensor([4.0, 5.0, 6.0], requires_grad=True)
        cpu_tensor = cuda_tensor.to('cpu')
        
        self.assertEqual(cpu_tensor.shape, cuda_tensor.shape)
        self.assertEqual(cpu_tensor.dtype, cuda_tensor.dtype)
        self.assertEqual(cpu_tensor.device.type, 'cpu')
        self.assertEqual(cpu_tensor.requires_grad, cuda_tensor.requires_grad)
        # self.assertEqual(cpu_tensor.is_leaf, cuda_tensor.is_leaf)
        self.assertTrue(np.allclose(cpu_tensor.data, cuda_tensor.data))
    
    def _test_tensor_to_grad_tracking(self):
        """测试 TN 张量 to() 迁移后的梯度跟踪功能（子用例）"""
        # 浮点张量梯度跟踪
        cpu_tensor = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
        cuda_tensor = cpu_tensor.to('cuda:0')
        
        result = cuda_tensor.sum()
        result.backward()
        
        self.assertIsNotNone(cpu_tensor.grad)
        self.assertTrue(np.allclose(cpu_tensor.grad, [1.0, 1.0, 1.0]))
        
        # 复数张量梯度跟踪 - 需要显式提供梯度
        with cuda.Device('cuda:0'):
            cuda_complex = rm.tensor([1+2j, 3+4j, 5+6j], dtype=rm.complex128, requires_grad=True)
        cpu_complex = cuda_complex.to('cpu')
        
        result = cpu_complex.sum()
        result.backward(gradient=rm.tensor(1.0 + 0j))  # 为复数结果提供复梯度
        
        self.assertIsNotNone(cuda_complex.grad)
    
    def _test_tensor_to_complex_migration(self):
        """测试复数张量的 to() 迁移功能（子用例）"""
        # 复数张量 CPU -> CUDA 迁移
        cpu_complex = rm.tensor([1+2j, 3+4j, 5+6j], dtype=rm.complex128, requires_grad=True)
        cuda_complex = cpu_complex.to('cuda:0')
        
        self.assertEqual(cuda_complex.device.type, 'cuda')
        self.assertEqual(cuda_complex.dtype, rm.complex128)
        self.assertTrue(np.allclose(cpu_complex.data, cuda_complex.data))
    
    def _test_tensor_to_parameter_combinations(self):
        """测试 TN 张量 to() 函数中 device 参数和 dtype 的各种取值、各种顺序组合（子用例）"""
        # 创建初始张量
        cpu_tensor = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float32, device='cpu')
        
        # 测试1: 位置参数 - device在前，dtype在后
        # 1.1 device为字符串
        result1 = cpu_tensor.to('cuda:0', rm.float64)
        self.assertEqual(result1.device.type, 'cuda')
        self.assertEqual(result1.device.index, 0)
        self.assertEqual(result1.dtype, rm.float64)
        
        # 1.2 device为整数
        result2 = cpu_tensor.to(0, rm.int32)
        self.assertEqual(result2.device.type, 'cuda')
        self.assertEqual(result2.device.index, 0)
        self.assertEqual(result2.dtype, rm.int32)
        
        # 1.3 device为Device对象
        result3 = cpu_tensor.to(rm.cuda.Device('cuda:0'), rm.float16)
        self.assertEqual(result3.device.type, 'cuda')
        self.assertEqual(result3.device.index, 0)
        self.assertEqual(result3.dtype, rm.float16)
        
        # 测试2: 位置参数 - dtype在前，device在后
        # 2.1 device为字符串
        result4 = cpu_tensor.to(rm.float64, 'cuda:0')
        self.assertEqual(result4.device.type, 'cuda')
        self.assertEqual(result4.device.index, 0)
        self.assertEqual(result4.dtype, rm.float64)
        
        # 2.2 device为整数
        result5 = cpu_tensor.to(rm.int32, 0)
        self.assertEqual(result5.device.type, 'cuda')
        self.assertEqual(result5.device.index, 0)
        self.assertEqual(result5.dtype, rm.int32)
        
        # 2.3 device为Device对象
        result6 = cpu_tensor.to(rm.float16, rm.cuda.Device('cuda:0'))
        self.assertEqual(result6.device.type, 'cuda')
        self.assertEqual(result6.device.index, 0)
        self.assertEqual(result6.dtype, rm.float16)
        
        # 测试3: 关键字参数
        # 3.1 只传device
        result7 = cpu_tensor.to(device='cuda:0')
        self.assertEqual(result7.device.type, 'cuda')
        self.assertEqual(result7.device.index, 0)
        self.assertEqual(result7.dtype, cpu_tensor.dtype)  # 保持原dtype
        
        # 3.2 只传dtype
        result8 = cpu_tensor.to(dtype=rm.float64)
        self.assertEqual(result8.device, cpu_tensor.device)  # 保持原device
        self.assertEqual(result8.dtype, rm.float64)
        
        # 3.3 同时传device和dtype，顺序1
        result9 = cpu_tensor.to(device='cuda:0', dtype=rm.float64)
        self.assertEqual(result9.device.type, 'cuda')
        self.assertEqual(result9.device.index, 0)
        self.assertEqual(result9.dtype, rm.float64)
        
        # 3.4 同时传device和dtype，顺序2
        result10 = cpu_tensor.to(dtype=rm.float64, device='cuda:0')
        self.assertEqual(result10.device.type, 'cuda')
        self.assertEqual(result10.device.index, 0)
        self.assertEqual(result10.dtype, rm.float64)
        
        # 测试4: 从另一个张量复制
        cuda_tensor = rm.tensor([4.0, 5.0, 6.0], dtype=rm.float64, device='cuda:0')
        result11 = cpu_tensor.to(cuda_tensor)
        self.assertEqual(result11.device.type, 'cuda')
        self.assertEqual(result11.device.index, 0)
        self.assertEqual(result11.dtype, rm.float64)
        
        # 测试5: 字符串dtype和字符串device的组合
        result12 = cpu_tensor.to('float64', 'cuda:0')
        self.assertEqual(result12.device.type, 'cuda')
        self.assertEqual(result12.device.index, 0)
        self.assertEqual(result12.dtype, rm.float64)
    
    def _test_tensor_cpu_function(self):
        """测试 TN 张量 cpu() 函数功能（子用例）"""
        # CUDA -> CPU 迁移测试
        with rm.cuda.Device('cuda:0'):
            cuda_tensor = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        cpu_tensor = cuda_tensor.cpu()
        
        self.assertEqual(cpu_tensor.shape, cuda_tensor.shape)
        self.assertEqual(cpu_tensor.dtype, cuda_tensor.dtype)
        self.assertEqual(cpu_tensor.device.type, 'cpu')
        self.assertEqual(cpu_tensor.requires_grad, cuda_tensor.requires_grad)
        self.assertTrue(np.allclose(cpu_tensor.data, cuda_tensor.data))
        
        # CPU -> CPU 迁移测试（应该返回自身或相同数据的张量）
        cpu_tensor2 = cpu_tensor.cpu()
        self.assertEqual(cpu_tensor2.device.type, 'cpu')
        self.assertTrue(np.allclose(cpu_tensor2.data, cpu_tensor.data))
        
        # 复数张量 CUDA -> CPU 迁移
        with rm.cuda.Device('cuda:0'):
            cuda_complex = rm.tensor([1+2j, 3+4j, 5+6j], dtype=rm.complex128, requires_grad=True)
        
        cpu_complex = cuda_complex.cpu()
        self.assertEqual(cpu_complex.device.type, 'cpu')
        self.assertEqual(cpu_complex.dtype, rm.complex128)
        self.assertTrue(np.allclose(cpu_complex.data, cuda_complex.data))
    
    def _test_tensor_cuda_function(self):
        """测试 TN 张量 cuda() 函数功能及不同参数取值（子用例）"""
        # 创建CPU张量
        cpu_tensor = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # 测试1: 默认参数（无参数，使用当前设备）
        cuda_tensor1 = cpu_tensor.cuda()
        self.assertEqual(cuda_tensor1.device.type, 'cuda')
        self.assertEqual(cuda_tensor1.shape, cpu_tensor.shape)
        self.assertEqual(cuda_tensor1.dtype, cpu_tensor.dtype)
        self.assertEqual(cuda_tensor1.requires_grad, cpu_tensor.requires_grad)
        self.assertTrue(np.allclose(cuda_tensor1.data, cpu_tensor.data))
        
        # 测试2: 整数设备ID参数
        cuda_tensor2 = cpu_tensor.cuda(0)
        self.assertEqual(cuda_tensor2.device.type, 'cuda')
        self.assertEqual(cuda_tensor2.device.index, 0)
        self.assertTrue(np.allclose(cuda_tensor2.data, cpu_tensor.data))
        
        # 测试3: 字符串设备名称参数
        cuda_tensor3 = cpu_tensor.cuda('cuda:0')
        self.assertEqual(cuda_tensor3.device.type, 'cuda')
        self.assertEqual(cuda_tensor3.device.index, 0)
        self.assertTrue(np.allclose(cuda_tensor3.data, cpu_tensor.data))
        
        # 测试4: 上下文管理器中使用cuda()
        with rm.cuda.Device('cuda:0'):
            cuda_tensor4 = cpu_tensor.cuda()
            self.assertEqual(cuda_tensor4.device.type, 'cuda')
            self.assertEqual(cuda_tensor4.device.index, 0)  # 应该使用上下文指定的设备
            self.assertTrue(np.allclose(cuda_tensor4.data, cpu_tensor.data))
        
        # 测试5: CUDA -> CUDA 迁移（设备相同）
        cuda_tensor5 = cuda_tensor1.cuda()
        self.assertEqual(cuda_tensor5.device.type, 'cuda')
        self.assertEqual(cuda_tensor5.device.index, cuda_tensor1.device.index)
        self.assertTrue(np.allclose(cuda_tensor5.data, cuda_tensor1.data))
        
        # 测试6: 复数张量 CPU -> CUDA 迁移
        cpu_complex = rm.tensor([1+2j, 3+4j, 5+6j], dtype=rm.complex128, requires_grad=True)
        cuda_complex = cpu_complex.cuda()
        self.assertEqual(cuda_complex.device.type, 'cuda')
        self.assertEqual(cuda_complex.dtype, rm.complex128)
        self.assertTrue(np.allclose(cuda_complex.data, cpu_complex.data))
        
        # 测试7: 带有梯度的运算测试
        with rm.cuda.Device('cuda:0'):
            x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True).cuda()
            y = x * 2.
            z = y.sum()
            z.backward()
            self.assertIsNotNone(x.grad)
            # 检查梯度数据，确保可以被numpy处理
            grad_data = x.grad.data if hasattr(x.grad, 'data') else x.grad
            self.assertTrue(np.allclose(grad_data, [2.0, 2.0, 2.0]))
        
        # 测试8: 多设备情况下的行为（如果有多个GPU可用）
        if rm.cuda.device_count() > 1:
            # 测试设备1
            with rm.cuda.Device('cuda:1'):
                cuda_tensor6 = cpu_tensor.cuda()
                self.assertEqual(cuda_tensor6.device.type, 'cuda')
                self.assertEqual(cuda_tensor6.device.index, 1)
                self.assertTrue(np.allclose(cuda_tensor6.data, cpu_tensor.data))
                
            # 测试显式指定设备1
            cuda_tensor7 = cpu_tensor.cuda(1)
            self.assertEqual(cuda_tensor7.device.type, 'cuda')
            self.assertEqual(cuda_tensor7.device.index, 1)
            self.assertTrue(np.allclose(cuda_tensor7.data, cpu_tensor.data))
    
    def test_basic_cuda_features(self):
        """测试基本 CUDA 功能"""
        try:
            # 定义基本 CUDA 功能的子用例
            basic_cuda_test_cases = [
                {
                    "name": "test_basic_cuda_functions",
                    "description": "测试基本 CUDA 功能函数",
                    "test_func": lambda: self._test_basic_cuda_functions()
                },
                {
                    "name": "test_default_device_setting",
                    "description": "测试默认设备设置",
                    "test_func": lambda: self._test_default_device_setting()
                },
                {
                    "name": "test_device_class",
                    "description": "测试 Device 类基本功能",
                    "test_func": lambda: self._test_device_class()
                },
                {
                    "name": "test_large_tensor_creation",
                    "description": "测试大规模张量创建",
                    "test_func": lambda: self._test_large_tensor_creation()
                }
            ]
            
            # 运行所有基本 CUDA 功能子用例
            for case in basic_cuda_test_cases:
                start_time = time.time()
                try:
                    case["test_func"]()
                    passed = True
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case["name"], passed)
                        print(f"测试用例: {case['description']} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                except Exception as e:
                    passed = False
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case["name"], passed, [str(e)])
                        print(f"测试用例: {case['description']} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                    raise
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("test_basic_cuda_features", True)
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("test_basic_cuda_features", False, [str(e)])
            raise
    
    def _test_basic_cuda_functions(self):
        """测试基本 CUDA 功能函数（子用例）"""
        # CUDA 可用检查
        self.assertTrue(cuda.is_available(), "CUDA 应该可用")
        
        # 可用设备数量检查
        self.assertGreaterEqual(cuda.device_count(), 1, "至少有一个 CUDA 设备")
        
        # 设备名称检查
        for i in range(cuda.device_count()):
            device_name = cuda.get_device_name(i)
            self.assertGreater(len(device_name), 0, f"设备 {i} 名称不应为空")
        
        # 内存分配和缓存清理检查
        mem_before = cuda.memory_allocated()
        
        # 分配一些内存
        temp_array = np.random.rand(1000, 1000).astype('float32')
        import cupy as cp
        temp_cupy_array = cp.asarray(temp_array)
        mem_after = cuda.memory_allocated()
        
        # 验证内存已分配
        self.assertGreater(mem_after, mem_before, "内存分配失败")
        
        # 清理缓存
        cuda.empty_cache()
        mem_after_clean = cuda.memory_allocated()
    
    def _test_default_device_setting(self):
        """测试默认设备设置（子用例）"""
        # 保存当前默认设备
        original_default = cuda.get_default_device()
        
        try:
            # 设置为 CPU
            cuda.set_default_device('cpu')
            current_default = cuda.get_default_device()
            self.assertEqual(current_default.type, 'cpu', "默认设备应已设置为 CPU")
            
            # 测试默认设备下创建的张量
            cpu_tensor = rm.tensor([1, 2, 3])
            self.assertEqual(cpu_tensor.device.type, 'cpu', "张量应创建在 CPU 上")
            
            # 设置为 CUDA
            cuda.set_default_device('cuda:0')
            current_default = cuda.get_default_device()
            self.assertEqual(current_default.type, 'cuda', "默认设备应已设置为 CUDA")
            
            # 测试默认设备下创建的张量
            cuda_tensor = rm.tensor([1, 2, 3])
            self.assertEqual(cuda_tensor.device.type, 'cuda', "张量应创建在 CUDA 上")
            self.assertEqual(cuda_tensor.device.index, 0, "张量应创建在 CUDA:0 上")
        finally:
            # 恢复原始默认设备
            cuda.set_default_device(original_default)
    
    def _test_device_class(self):
        """测试 Device 类基本功能（子用例）"""
        # 创建 CPU 设备对象
        cpu_device = cuda.Device('cpu')
        self.assertEqual(cpu_device.type, 'cpu', "CPU 设备类型不正确")
        
        # 创建 CUDA 设备对象（使用索引）
        cuda_device = cuda.Device(0)
        self.assertEqual(cuda_device.type, 'cuda', "CUDA 设备类型不正确")
        self.assertEqual(cuda_device.index, 0, "CUDA 设备索引不正确")
        
        # 创建 CUDA 设备对象（使用字符串）
        cuda_device2 = cuda.Device('cuda:0')
        self.assertEqual(cuda_device, cuda_device2, "不同方式创建的 CUDA 设备应相同")

    def _test_context_device_priority(self):
        """测试上下文优先级高于默认设备设置（子用例）"""
        # 保存当前默认设备
        original_default = cuda.get_default_device()
        
        try:
            with cuda.Device('cuda:0'):
                # 在 CUDA 上下文中，默认设备应该为 CUDA
                tensor = rm.tensor([1, 2, 3])
                self.assertEqual(tensor.device.type, 'cuda', "CUDA 上下文中创建的张量应在 CUDA 上")
                
                # 测试上下文优先级高于默认设备设置
                cuda.set_default_device('cpu')
                tensor2 = rm.tensor([4, 5, 6])
                self.assertEqual(tensor2.device.type, 'cuda', "上下文优先级应高于默认设备设置")
        finally:
            # 恢复原始默认设备
            cuda.set_default_device(original_default)

if __name__ == "__main__":
    # 清屏
    clear_screen()
    
    # 检查CUDA是否可用
    if not CUDA_AVAILABLE:
        print(f"{Colors.WARNING}警告: CUDA 不可用，无法运行 CUDA 测试{Colors.ENDC}")
        sys.exit(0)
    
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行 CUDA 功能测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(CUDAUnitTest)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)