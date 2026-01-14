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
    print("警告: 无法导入PyTorch，将只测试riemann的随机函数")
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

# 定义设备列表：始终包含CPU，CUDA可用时添加CUDA
device_list = [None, "cpu"]
if rm.cuda.CUPY_AVAILABLE:
    device_list.extend(["cuda", "cuda:0"])  # 添加cuda和cuda:0

# 辅助函数：获取设备名称
def get_device_name(device):
    return device if device is not None else "默认设备"

# 辅助函数：检查张量设备是否匹配
def check_device_match(rm_tensor, expected_device):
    """检查张量设备是否与期望设备匹配"""
    if not expected_device:
        return True
    
    actual_device = str(rm_tensor.device)
    expected_device_str = expected_device if isinstance(expected_device, str) else str(expected_device)
    
    # 处理cuda和cuda:0的情况
    if expected_device_str == "cuda" and actual_device.startswith("cuda:"):
        return True
    
    return actual_device == expected_device_str

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

class TestRandomFunctions(unittest.TestCase):
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
    
    def compare_tensor_shapes_and_dtypes(self, rm_tensor, torch_tensor):
        """比较两个张量的形状和数据类型"""
        try:
            # 比较形状
            rm_shape = rm_tensor.shape if hasattr(rm_tensor, 'shape') else rm_tensor.data.shape
            torch_shape = torch_tensor.shape
            if rm_shape != torch_shape:
                return False, f"形状不匹配: Riemann={rm_shape}, PyTorch={torch_shape}"
            
            # 比较数据类型
            rm_dtype = rm_tensor.data.dtype
            torch_dtype = torch_tensor.numpy().dtype
            if rm_dtype != torch_dtype:
                return False, f"数据类型不匹配: Riemann={rm_dtype}, PyTorch={torch_dtype}"
            
            return True, None
        except Exception as e:
            return False, f"比较时出错: {str(e)}"
    
    def check_rand_range(self, rm_tensor):
        """检查rand函数生成的随机数是否在[0,1)范围内"""
        try:
            data = rm_tensor.data if hasattr(rm_tensor, 'data') else rm_tensor
            min_val, max_val = np.min(data), np.max(data)
            if min_val < 0 or max_val >= 1:
                return False, f"随机数范围超出[0,1): [{min_val}, {max_val}]"
            return True, None
        except Exception as e:
            return False, f"检查范围时出错: {str(e)}"
    
    def check_randint_range(self, rm_tensor, low, high):
        """检查randint函数生成的随机数是否在指定范围内"""
        try:
            data = rm_tensor.data if hasattr(rm_tensor, 'data') else rm_tensor
            min_val, max_val = np.min(data), np.max(data)
            if min_val < low or max_val >= high:
                return False, f"随机数范围不在[{low}, {high})内: [{min_val}, {max_val}]"
            return True, None
        except Exception as e:
            return False, f"检查范围时出错: {str(e)}"
    
    def check_normal_stats(self, rm_tensor, mean, std, tolerance=0.5):
        """检查normal函数生成的随机数的统计特性是否接近预期"""
        try:
            data = rm_tensor.data if hasattr(rm_tensor, 'data') else rm_tensor
            actual_mean, actual_std = np.mean(data), np.std(data)
            
            # 由于是随机样本，允许一定的误差
            if abs(actual_mean - mean) > tolerance or abs(actual_std - std) > tolerance:
                return False, f"统计特性不符合预期: 均值={actual_mean:.4f} vs {mean}, 标准差={actual_std:.4f} vs {std}"
            return True, None
        except Exception as e:
            return False, f"检查统计特性时出错: {str(e)}"
    
    def check_permutation(self, rm_tensor, n):
        """检查randperm函数生成的是否为有效的排列"""
        try:
            data = rm_tensor.data if hasattr(rm_tensor, 'data') else rm_tensor
            sorted_data = sorted(data)
            expected = list(range(n))
            if sorted_data != expected:
                return False, f"不是有效排列: {sorted_data} vs {expected}"
            return True, None
        except Exception as e:
            return False, f"检查排列时出错: {str(e)}"
    
    def test_rand(self):
        """测试rand函数 - 生成[0,1)均匀分布的随机数"""
        # 测试不同形状参数
        shapes = [(2, 3), (4,), (1, 5, 3)]
        dtypes = [np.float32, np.float64]
        
        for shape in shapes:
            for dtype in dtypes:
                for device in device_list:
                    device_name = get_device_name(device)
                    case_name = f"rand({shape}, {dtype}) - 设备:{device_name}"
                    start_time = time.time()
                    try:
                        # 创建Riemann张量
                        rm_result = rm.rand(*shape, dtype=dtype, device=device)
                        # 创建PyTorch张量作为参考（仅用于形状和类型比较）
                        if TORCH_AVAILABLE:
                            torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
                            torch_result = torch.rand(shape, dtype=torch_dtype)
                        else:
                            torch_result = None
                        
                        # 比较形状和数据类型
                        if TORCH_AVAILABLE:
                            shape_dtype_passed, shape_dtype_details = self.compare_tensor_shapes_and_dtypes(rm_result, torch_result)
                            self.assertTrue(shape_dtype_passed, f"{case_name} 形状/类型: {shape_dtype_details}")
                        
                        # 检查Riemann随机数范围
                        range_passed, range_details = self.check_rand_range(rm_result)
                        self.assertTrue(range_passed, f"{case_name} 范围: {range_details}")
                        
                        # 检查设备是否匹配
                        device_passed = check_device_match(rm_result, device)
                        self.assertTrue(device_passed, f"{case_name} 设备不匹配: 期望{device}, 实际{rm_result.device}")
                        
                        # 检查requires_grad参数
                        rm_result_grad = rm.rand(*shape, requires_grad=True, device=device)
                        self.assertTrue(hasattr(rm_result_grad, 'requires_grad') and rm_result_grad.requires_grad, 
                                       f"{case_name} requires_grad: requires_grad参数未正确设置")
                        
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
                        raise
    
    def test_randn(self):
        """测试randn函数 - 生成标准正态分布的随机数"""
        # 测试不同形状参数
        shapes = [(2, 3), (4,), (1, 5, 3)]
        dtypes = [np.float32, np.float64]
        
        for shape in shapes:
            for dtype in dtypes:
                for device in device_list:
                    device_name = get_device_name(device)
                    case_name = f"randn({shape}, {dtype}) - 设备:{device_name}"
                    start_time = time.time()
                    try:
                        # 创建Riemann张量
                        rm_result = rm.randn(*shape, dtype=dtype, device=device)
                        # 创建PyTorch张量作为参考（仅用于形状和类型比较）
                        if TORCH_AVAILABLE:
                            torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
                            torch_result = torch.randn(shape, dtype=torch_dtype)
                        else:
                            torch_result = None
                        
                        # 比较形状和数据类型
                        if TORCH_AVAILABLE:
                            shape_dtype_passed, shape_dtype_details = self.compare_tensor_shapes_and_dtypes(rm_result, torch_result)
                            self.assertTrue(shape_dtype_passed, f"{case_name} 形状/类型: {shape_dtype_details}")
                        
                        # 检查设备是否匹配
                        device_passed = check_device_match(rm_result, device)
                        self.assertTrue(device_passed, f"{case_name} 设备不匹配: 期望{device}, 实际{rm_result.device}")
                        
                        # 检查requires_grad参数
                        rm_result_grad = rm.randn(*shape, requires_grad=True, device=device)
                        self.assertTrue(hasattr(rm_result_grad, 'requires_grad') and rm_result_grad.requires_grad, 
                                       f"{case_name} requires_grad: requires_grad参数未正确设置")
                        
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
                        raise
    
    def test_randint(self):
        """测试randint函数 - 生成整数随机数"""
        # 测试不同参数组合
        test_cases = [
            ((2, 3), 0, 10, np.int32),
            ((4,), 5, 15, np.int64),
            ((1, 5, 3), -10, 0, np.int32)
        ]
        
        for shape, low, high, dtype in test_cases:
            for device in device_list:
                device_name = get_device_name(device)
                case_name = f"randint({low}, {high}, {shape}, {dtype}) - 设备:{device_name}"
                start_time = time.time()
                try:
                    # 创建Riemann张量 - 注意接口差异：riemann的size是显式参数
                    rm_result = rm.randint(low, high, size=shape, dtype=dtype, device=device)
                    # 创建PyTorch张量作为参考（仅用于形状和类型比较）
                    if TORCH_AVAILABLE:
                        torch_dtype = torch.int32 if dtype == np.int32 else torch.int64
                        torch_result = torch.randint(low, high, shape, dtype=torch_dtype)
                    else:
                        torch_result = None
                    
                    # 比较形状和数据类型
                    if TORCH_AVAILABLE:
                        shape_dtype_passed, shape_dtype_details = self.compare_tensor_shapes_and_dtypes(rm_result, torch_result)
                        self.assertTrue(shape_dtype_passed, f"{case_name} 形状/类型: {shape_dtype_details}")
                    
                    # 检查随机数范围
                    range_passed, range_details = self.check_randint_range(rm_result, low, high)
                    self.assertTrue(range_passed, f"{case_name} 范围: {range_details}")
                    
                    # 检查设备是否匹配
                    device_passed = check_device_match(rm_result, device)
                    self.assertTrue(device_passed, f"{case_name} 设备不匹配: 期望{device}, 实际{rm_result.device}")
                    
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
                    raise
        
        # 添加PyTorch风格测试 - randint(high, size)
        for device in device_list:
            device_name = get_device_name(device)
            case_name = f"PyTorch style: randint(high, size) - 设备:{device_name}"
            start_time = time.time()
            try:
                rm_result = rm.randint(10, (3, 4), device=device)
                if TORCH_AVAILABLE:
                    torch_result = torch.randint(10, (3, 4))
                    shape_dtype_passed, shape_dtype_details = self.compare_tensor_shapes_and_dtypes(rm_result, torch_result)
                    self.assertTrue(shape_dtype_passed, f"{case_name} 形状/类型: {shape_dtype_details}")
                range_passed, range_details = self.check_randint_range(rm_result, 0, 10)
                self.assertTrue(range_passed, f"{case_name} 范围: {range_details}")
                
                # 检查设备是否匹配
                device_passed = check_device_match(rm_result, device)
                self.assertTrue(device_passed, f"{case_name} 设备不匹配: 期望{device}, 实际{rm_result.device}")
                
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
                raise
        
        # 添加一维张量测试 - randint(high, size)（整数size）
        for device in device_list:
            device_name = get_device_name(device)
            case_name = f"1D tensor with integer size: randint(high, size) - 设备:{device_name}"
            start_time = time.time()
            try:
                rm_result = rm.randint(10, 6, device=device)
                expected_shape = (6,)
                self.assertEqual(rm_result.shape, expected_shape, f"{case_name} 形状不匹配: {rm_result.shape} vs {expected_shape}")
                range_passed, range_details = self.check_randint_range(rm_result, 0, 10)
                self.assertTrue(range_passed, f"{case_name} 范围: {range_details}")
                
                # 检查设备是否匹配
                device_passed = check_device_match(rm_result, device)
                self.assertTrue(device_passed, f"{case_name} 设备不匹配: 期望{device}, 实际{rm_result.device}")
                
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
                raise
        
        # 添加关键字参数测试
        for device in device_list:
            device_name = get_device_name(device)
            case_name = f"Keyword arguments: randint(low=5, high=15, size=(3, 4)) - 设备:{device_name}"
            start_time = time.time()
            try:
                rm_result = rm.randint(low=5, high=15, size=(3, 4), device=device)
                range_passed, range_details = self.check_randint_range(rm_result, 5, 15)
                self.assertTrue(range_passed, f"{case_name} 范围: {range_details}")
                
                # 检查设备是否匹配
                device_passed = check_device_match(rm_result, device)
                self.assertTrue(device_passed, f"{case_name} 设备不匹配: 期望{device}, 实际{rm_result.device}")
                
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
                raise
        
        # 添加PyTorch混合风格测试 - randint(high, size=size)
        for device in device_list:
            device_name = get_device_name(device)
            case_name = f"PyTorch hybrid style: randint(high, size=size) - 设备:{device_name}"
            start_time = time.time()
            try:
                rm_result = rm.randint(10, size=(3, 4), device=device)
                if TORCH_AVAILABLE:
                    torch_result = torch.randint(10, (3, 4))
                    shape_dtype_passed, shape_dtype_details = self.compare_tensor_shapes_and_dtypes(rm_result, torch_result)
                    self.assertTrue(shape_dtype_passed, f"{case_name} 形状/类型: {shape_dtype_details}")
                range_passed, range_details = self.check_randint_range(rm_result, 0, 10)
                self.assertTrue(range_passed, f"{case_name} 范围: {range_details}")
                
                # 检查设备是否匹配
                device_passed = check_device_match(rm_result, device)
                self.assertTrue(device_passed, f"{case_name} 设备不匹配: 期望{device}, 实际{rm_result.device}")
                
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
                raise
    
    def test_randperm(self):
        """测试randperm函数 - 生成随机排列"""
        # 测试不同大小
        test_sizes = [5, 10, 15]
        dtypes = [np.int32, np.int64]
        
        for n in test_sizes:
            for dtype in dtypes:
                for device in device_list:
                    device_name = get_device_name(device)
                    case_name = f"randperm({n}, {dtype}) - 设备:{device_name}"
                    start_time = time.time()
                    try:
                        # 创建Riemann张量
                        rm_result = rm.randperm(n, dtype=dtype, device=device)
                        # 创建PyTorch张量作为参考（仅用于形状比较）
                        if TORCH_AVAILABLE:
                            torch_dtype = torch.int32 if dtype == np.int32 else torch.int64
                            torch_result = torch.randperm(n, dtype=torch_dtype)
                        else:
                            torch_result = None
                        
                        # 比较形状
                        rm_shape = (n,) if hasattr(rm_result, 'shape') and rm_result.shape == (n,) else \
                                  (n,) if hasattr(rm_result, 'data') and rm_result.data.shape == (n,) else None
                        
                        if TORCH_AVAILABLE:
                            torch_shape = torch_result.shape
                            self.assertEqual(rm_shape, torch_shape, 
                                            f"{case_name} 形状不匹配: Riemann={rm_shape}, PyTorch={torch_shape}")
                        
                        # 检查是否为有效排列
                        perm_passed, perm_details = self.check_permutation(rm_result, n)
                        self.assertTrue(perm_passed, f"{case_name} 排列: {perm_details}")
                        
                        # 检查设备是否匹配
                        device_passed = check_device_match(rm_result, device)
                        self.assertTrue(device_passed, f"{case_name} 设备不匹配: 期望{device}, 实际{rm_result.device}")
                        
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
                        raise
    
    def test_normal(self):
        """测试normal函数 - 生成指定均值和标准差的正态分布随机数"""
        # 测试不同参数组合
        test_cases = [
            ((2, 3), 0.0, 1.0, np.float32),   # 标准正态分布
            ((4,), 5.0, 2.0, np.float64),      # 均值5，标准差2
            ((1, 5, 3), -2.0, 0.5, np.float32) # 均值-2，标准差0.5
        ]
        
        for shape, mean, std, dtype in test_cases:
            for device in device_list:
                device_name = get_device_name(device)
                case_name = f"normal({mean}, {std}, {shape}, {dtype}) - 设备:{device_name}"
                start_time = time.time()
                try:
                    # 创建Riemann张量 - 注意接口差异：riemann的参数顺序是mean, std, size
                    rm_result = rm.normal(mean, std, size=shape, dtype=dtype, device=device)
                    # 创建PyTorch张量作为参考（仅用于形状和类型比较）
                    if TORCH_AVAILABLE:
                        torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
                        # 使用正确的参数形式：float mean, float std, tuple size
                        torch_result = torch.normal(mean, std, shape, dtype=torch_dtype)
                    else:
                        torch_result = None
                    
                    # 比较形状和数据类型
                    if TORCH_AVAILABLE:
                        shape_dtype_passed, shape_dtype_details = self.compare_tensor_shapes_and_dtypes(rm_result, torch_result)
                        self.assertTrue(shape_dtype_passed, f"{case_name} 形状/类型: {shape_dtype_details}")
                    
                    # 检查统计特性（对于较大的张量，结果更接近预期）
                    if np.prod(shape) > 10:  # 只对足够大的张量进行统计检查
                        stats_passed, stats_details = self.check_normal_stats(rm_result, mean, std)
                        self.assertTrue(stats_passed, f"{case_name} 统计: {stats_details}")
                    
                    # 检查设备是否匹配
                    device_passed = check_device_match(rm_result, device)
                    self.assertTrue(device_passed, f"{case_name} 设备不匹配: 期望{device}, 实际{rm_result.device}")
                    
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
                    raise

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行随机函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRandomFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)