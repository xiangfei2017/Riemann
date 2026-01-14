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

# 检查CUDA是否可用
try:
    CUDA_AVAILABLE = rm.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

# 定义设备列表
device_list = [None, "cpu"]
if CUDA_AVAILABLE:
    device_list.extend(["cuda", "cuda:0"])

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

# 辅助函数：获取设备名称
def get_device_name(device):
    return device if device is not None else "默认设备"

# 辅助函数：检查张量是否与期望值匹配
def check_tensor_match(rm_tensor, expected_array, device, dtype=None):
    """检查张量是否与期望值匹配，处理CUDA张量转换"""
    if device and device.startswith("cuda"):
        # CUDA张量需要先转回CPU进行检查
        tensor_data = rm_tensor.to("cpu").numpy()
    else:
        tensor_data = rm_tensor.numpy()
    
    # 检查值是否匹配
    if isinstance(expected_array, np.ndarray):
        if tensor_data.dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            match = np.allclose(tensor_data, expected_array)
        else:
            match = np.array_equal(tensor_data, expected_array)
    else:
        # 对于标量比较
        match = tensor_data.item() == expected_array
    
    # 检查数据类型
    dtype_match = dtype is None or rm_tensor.dtype == dtype
    
    return match, dtype_match

# 辅助函数：检查设备是否匹配
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
    
    def test_zeros(self):
        """测试zeros函数"""
        test_cases = []
        for device in device_list:
            device_name = get_device_name(device)
            test_cases.extend([
                {"name": f"标量形状 - 设备:{device_name}", "shape": (), "dtype": None, "device": device},
                {"name": f"一维张量 - 设备:{device_name}", "shape": (5,), "dtype": None, "device": device},
                {"name": f"二维张量 - 设备:{device_name}", "shape": (3, 4), "dtype": None, "device": device},
                {"name": f"三维张量 - 设备:{device_name}", "shape": (2, 3, 4), "dtype": None, "device": device},
                {"name": f"指定float32类型 - 设备:{device_name}", "shape": (3, 4), "dtype": rm.float32, "device": device},
                {"name": f"指定int32类型 - 设备:{device_name}", "shape": (3, 4), "dtype": rm.int32, "device": device},
            ])
        
        for case in test_cases:
            case_name = f"{self.test_zeros.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建张量
                rm_tensor = rm.zeros(case["shape"], dtype=case["dtype"], device=case["device"])
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 检查所有元素是否为0
                    expected_array = np.zeros(case["shape"])
                    value_match, dtype_match = check_tensor_match(rm_tensor, expected_array, case["device"], case["dtype"])
                    
                    if not value_match:
                        passed = False
                        error_msg = "不是所有元素都是0"
                    elif not dtype_match:
                        passed = False
                        error_msg = f"数据类型不匹配: 期望{case['dtype']}, 得到{rm_tensor.dtype}"
                    elif not check_device_match(rm_tensor, case["device"]):
                        passed = False
                        error_msg = f"设备不匹配: 期望{case['device']}, 得到{rm_tensor.device}"
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
            for device in device_list:
                # 构建设备名称
                device_name = device if device is not None else "默认设备"
                case_name = f"{self.test_zeros_like.__doc__} - {case['name']} - 设备:{device_name}"
                start_time = time.time()
                try:
                    # 创建输入张量
                    input_tensor = rm.ones(case["shape"], dtype=case["dtype"], device=device)
                    
                    # 使用Riemann创建zeros_like张量
                    rm_tensor = rm.zeros_like(input_tensor, device=device)
                    
                    # 检查形状
                    shape_match = rm_tensor.shape == case["shape"]
                    if not shape_match:
                        passed = False
                        error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                    else:
                        # 检查所有元素是否为0
                        if device and device.startswith("cuda"):
                            # CUDA张量需要先转回CPU进行检查
                            all_zeros = np.allclose(rm_tensor.to("cpu").numpy(), np.zeros(case["shape"]))
                        else:
                            all_zeros = np.allclose(rm_tensor.numpy(), np.zeros(case["shape"]))
                        if not all_zeros:
                            passed = False
                            error_msg = "不是所有元素都是0"
                        else:
                            # 检查数据类型
                            if case["dtype"] is not None and rm_tensor.dtype != case["dtype"]:
                                passed = False
                                error_msg = f"数据类型不匹配: 期望{case['dtype']}, 得到{rm_tensor.dtype}"
                            else:
                                # 检查设备
                                if device:
                                    expected_device = device if isinstance(device, str) else str(device)
                                    actual_device = str(rm_tensor.device)
                                    if expected_device == "cuda" and actual_device.startswith("cuda:"):
                                        # 处理cuda和cuda:0的情况
                                        pass
                                    elif actual_device != expected_device:
                                        passed = False
                                        error_msg = f"设备不匹配: 期望{expected_device}, 得到{actual_device}"
                                    else:
                                        passed = True
                                        error_msg = ""
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
        test_cases = []
        for device in device_list:
            device_name = get_device_name(device)
            test_cases.extend([
                {"name": f"标量形状 - 设备:{device_name}", "shape": (), "dtype": None, "device": device},
                {"name": f"一维张量 - 设备:{device_name}", "shape": (5,), "dtype": None, "device": device},
                {"name": f"二维张量 - 设备:{device_name}", "shape": (3, 4), "dtype": None, "device": device},
                {"name": f"三维张量 - 设备:{device_name}", "shape": (2, 3, 4), "dtype": None, "device": device},
                {"name": f"指定float32类型 - 设备:{device_name}", "shape": (3, 4), "dtype": rm.float32, "device": device},
                {"name": f"指定int32类型 - 设备:{device_name}", "shape": (3, 4), "dtype": rm.int32, "device": device},
            ])
        
        for case in test_cases:
            case_name = f"{self.test_ones.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建张量
                rm_tensor = rm.ones(case["shape"], dtype=case["dtype"], device=case["device"])
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 检查所有元素是否为1
                    expected_array = np.ones(case["shape"])
                    value_match, dtype_match = check_tensor_match(rm_tensor, expected_array, case["device"], case["dtype"])
                    
                    if not value_match:
                        passed = False
                        error_msg = "不是所有元素都是1"
                    elif not dtype_match:
                        passed = False
                        error_msg = f"数据类型不匹配: 期望{case['dtype']}, 得到{rm_tensor.dtype}"
                    elif not check_device_match(rm_tensor, case["device"]):
                        passed = False
                        error_msg = f"设备不匹配: 期望{case['device']}, 得到{rm_tensor.device}"
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
            for device in device_list:
                # 构建设备名称
                device_name = device if device is not None else "默认设备"
                case_name = f"{self.test_ones_like.__doc__} - {case['name']} - 设备:{device_name}"
                start_time = time.time()
                try:
                    # 创建输入张量
                    input_tensor = rm.zeros(case["shape"], dtype=case["dtype"], device=device)
                    
                    # 使用Riemann创建ones_like张量
                    rm_tensor = rm.ones_like(input_tensor, device=device)
                    
                    # 检查形状
                    shape_match = rm_tensor.shape == case["shape"]
                    if not shape_match:
                        passed = False
                        error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                    else:
                        # 检查所有元素是否为1
                        if device and device.startswith("cuda"):
                            # CUDA张量需要先转回CPU进行检查
                            all_ones = np.allclose(rm_tensor.to("cpu").numpy(), np.ones(case["shape"]))
                        else:
                            all_ones = np.allclose(rm_tensor.numpy(), np.ones(case["shape"]))
                        if not all_ones:
                            passed = False
                            error_msg = "不是所有元素都是1"
                        else:
                            # 检查数据类型
                            if case["dtype"] is not None and rm_tensor.dtype != case["dtype"]:
                                passed = False
                                error_msg = f"数据类型不匹配: 期望{case['dtype']}, 得到{rm_tensor.dtype}"
                            else:
                                # 检查设备
                                if device:
                                    expected_device = device if isinstance(device, str) else str(device)
                                    actual_device = str(rm_tensor.device)
                                    if expected_device == "cuda" and actual_device.startswith("cuda:"):
                                        # 处理cuda和cuda:0的情况
                                        pass
                                    elif actual_device != expected_device:
                                        passed = False
                                        error_msg = f"设备不匹配: 期望{expected_device}, 得到{actual_device}"
                                    else:
                                        passed = True
                                        error_msg = ""
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
            for device in device_list:
                # 构建设备名称
                device_name = device if device is not None else "默认设备"
                case_name = f"{self.test_empty.__doc__} - {case['name']} - 设备:{device_name}"
                start_time = time.time()
                try:
                    # 使用Riemann创建empty张量
                    rm_tensor = rm.empty(case["shape"], device=device)
                    
                    # 检查形状
                    shape_match = rm_tensor.shape == case["shape"]
                    if not shape_match:
                        passed = False
                        error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                    else:
                        # 检查设备
                        if device:
                            expected_device = device if isinstance(device, str) else str(device)
                            actual_device = str(rm_tensor.device)
                            if expected_device == "cuda" and actual_device.startswith("cuda:"):
                                # 处理cuda和cuda:0的情况
                                pass
                            elif actual_device != expected_device:
                                passed = False
                                error_msg = f"设备不匹配: 期望{expected_device}, 得到{actual_device}"
                            else:
                                # 对于empty函数，我们只检查形状和设备，不检查值
                                passed = True
                                error_msg = ""
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
            for device in device_list:
                # 构建设备名称
                device_name = device if device is not None else "默认设备"
                case_name = f"{self.test_empty_like.__doc__} - {case['name']} - 设备:{device_name}"
                start_time = time.time()
                try:
                    # 创建输入张量
                    input_tensor = rm.ones(case["shape"], device=device)
                    
                    # 使用Riemann创建empty_like张量
                    rm_tensor = rm.empty_like(input_tensor, device=device)
                    
                    # 检查形状
                    shape_match = rm_tensor.shape == case["shape"]
                    if not shape_match:
                        passed = False
                        error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                    else:
                        # 检查设备
                        if device:
                            expected_device = device if isinstance(device, str) else str(device)
                            actual_device = str(rm_tensor.device)
                            if expected_device == "cuda" and actual_device.startswith("cuda:"):
                                # 处理cuda和cuda:0的情况
                                pass
                            elif actual_device != expected_device:
                                passed = False
                                error_msg = f"设备不匹配: 期望{expected_device}, 得到{actual_device}"
                            else:
                                # 对于empty_like函数，我们只检查形状和设备，不检查值
                                passed = True
                                error_msg = ""
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
        test_cases = []
        for device in device_list:
            device_name = get_device_name(device)
            test_cases.extend([
                {"name": f"标量形状整数填充 - 设备:{device_name}", "shape": (), "fill_value": 5, "dtype": None, "device": device},
                {"name": f"一维张量浮点数填充 - 设备:{device_name}", "shape": (5,), "fill_value": 3.14, "dtype": None, "device": device},
                {"name": f"二维张量整数填充 - 设备:{device_name}", "shape": (3, 4), "fill_value": 7, "dtype": None, "device": device},
                {"name": f"指定float32类型 - 设备:{device_name}", "shape": (3, 4), "fill_value": 3.14, "dtype": rm.float32, "device": device},
                {"name": f"指定int32类型 - 设备:{device_name}", "shape": (3, 4), "fill_value": 7, "dtype": rm.int32, "device": device},
            ])
        
        for case in test_cases:
            case_name = f"{self.test_full.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建张量
                rm_tensor = rm.full(case["shape"], fill_value=case["fill_value"], dtype=case["dtype"], device=case["device"])
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 检查所有元素是否为指定值
                    expected_array = np.full(case["shape"], case["fill_value"])
                    value_match, dtype_match = check_tensor_match(rm_tensor, expected_array, case["device"], case["dtype"])
                    
                    if not value_match:
                        passed = False
                        error_msg = f"值不匹配，期望{case['fill_value']}"
                    elif not dtype_match:
                        passed = False
                        error_msg = f"数据类型不匹配: 期望{case['dtype']}, 得到{rm_tensor.dtype}"
                    elif not check_device_match(rm_tensor, case["device"]):
                        passed = False
                        error_msg = f"设备不匹配: 期望{case['device']}, 得到{rm_tensor.device}"
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
        test_cases = []
        for device in device_list:
            device_name = get_device_name(device)
            test_cases.extend([
                {"name": f"标量输入整数填充 - 设备:{device_name}", "shape": (), "fill_value": 5, "dtype": None, "device": device},
                {"name": f"一维张量输入浮点数填充 - 设备:{device_name}", "shape": (5,), "fill_value": 3.14, "dtype": None, "device": device},
                {"name": f"二维张量输入整数填充 - 设备:{device_name}", "shape": (3, 4), "fill_value": 7, "dtype": None, "device": device},
                {"name": f"float32类型输入 - 设备:{device_name}", "shape": (3, 4), "fill_value": 3.14, "dtype": rm.float32, "device": device},
            ])
        
        for case in test_cases:
            case_name = f"{self.test_full_like.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建输入张量
                input_tensor = rm.ones(case["shape"], dtype=case["dtype"], device=case["device"])
                
                # 使用Riemann创建full_like张量
                rm_tensor = rm.full_like(input_tensor, case["fill_value"], device=case["device"])
                
                # 检查形状
                shape_match = rm_tensor.shape == case["shape"]
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{case['shape']}, 得到{rm_tensor.shape}"
                else:
                    # 检查所有元素是否为指定值
                    expected_array = np.full(case["shape"], case["fill_value"])
                    value_match, dtype_match = check_tensor_match(rm_tensor, expected_array, case["device"], case["dtype"])
                    
                    if not value_match:
                        passed = False
                        error_msg = f"值不匹配，期望{case['fill_value']}"
                    elif not dtype_match:
                        passed = False
                        error_msg = f"数据类型不匹配: 期望{case['dtype']}, 得到{rm_tensor.dtype}"
                    elif not check_device_match(rm_tensor, case["device"]):
                        passed = False
                        error_msg = f"设备不匹配: 期望{case['device']}, 得到{rm_tensor.device}"
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
        test_cases = []
        for device in device_list:
            device_name = get_device_name(device)
            test_cases.extend([
                {"name": f"方阵n=3 - 设备:{device_name}", "n": 3, "m": None, "dtype": None, "device": device},
                {"name": f"矩形矩阵n=2,m=4 - 设备:{device_name}", "n": 2, "m": 4, "dtype": None, "device": device},
                {"name": f"矩形矩阵n=4,m=2 - 设备:{device_name}", "n": 4, "m": 2, "dtype": None, "device": device},
                {"name": f"指定float32类型 - 设备:{device_name}", "n": 3, "m": None, "dtype": rm.float32, "device": device},
            ])
        
        for case in test_cases:
            case_name = f"{self.test_eye.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建单位矩阵
                if case["m"] is None:
                    rm_tensor = rm.eye(case["n"], dtype=case["dtype"], device=case["device"])
                    expected_shape = (case["n"], case["n"])
                else:
                    rm_tensor = rm.eye(case["n"], case["m"], dtype=case["dtype"], device=case["device"])
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
                    value_match, dtype_match = check_tensor_match(rm_tensor, expected_matrix, case["device"], case["dtype"])
                    
                    if not value_match:
                        passed = False
                        error_msg = "对角线元素不为1或非对角线元素不为0"
                    elif not dtype_match:
                        passed = False
                        error_msg = f"数据类型不匹配: 期望{case['dtype']}, 得到{rm_tensor.dtype}"
                    elif not check_device_match(rm_tensor, case["device"]):
                        passed = False
                        error_msg = f"设备不匹配: 期望{case['device']}, 得到{rm_tensor.device}"
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
        test_cases = []
        for device in device_list:
            device_name = get_device_name(device)
            test_cases.extend([
                {"name": f"只有end参数 - 设备:{device_name}", "start": None, "end": 5, "step": None, "device": device},
                {"name": f"有start和end参数 - 设备:{device_name}", "start": 2, "end": 10, "step": None, "device": device},
                {"name": f"有start、end和step参数 - 设备:{device_name}", "start": 1, "end": 10, "step": 2, "device": device},
                {"name": f"负step参数 - 设备:{device_name}", "start": 10, "end": 1, "step": -2, "device": device},
            ])
        
        for case in test_cases:
            case_name = f"{self.test_arange.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建arange张量
                if case["start"] is None:
                    rm_tensor = rm.arange(case["end"], device=case["device"])
                elif case["step"] is None:
                    rm_tensor = rm.arange(case["start"], case["end"], device=case["device"])
                else:
                    rm_tensor = rm.arange(case["start"], case["end"], case["step"], device=case["device"])
                
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
                    value_match, _ = check_tensor_match(rm_tensor, expected_array, case["device"])
                    
                    if not value_match:
                        passed = False
                        error_msg = "值不匹配"
                    elif not check_device_match(rm_tensor, case["device"]):
                        passed = False
                        error_msg = f"设备不匹配: 期望{case['device']}, 得到{rm_tensor.device}"
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
        test_cases = []
        for device in device_list:
            device_name = get_device_name(device)
            test_cases.extend([
                {"name": f"基本用法 - 设备:{device_name}", "start": 0, "end": 1, "num": 5, "device": device},
                {"name": f"更多点 - 设备:{device_name}", "start": 0, "end": 10, "num": 11, "device": device},
                {"name": f"降序 - 设备:{device_name}", "start": 10, "end": 0, "num": 6, "device": device},
                {"name": f"单点 - 设备:{device_name}", "start": 5, "end": 5, "num": 1, "device": device},
            ])
        
        for case in test_cases:
            case_name = f"{self.test_linspace.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 使用Riemann创建linspace张量
                rm_tensor = rm.linspace(case["start"], case["end"], case["num"], device=case["device"])
                
                # 创建期望的数组
                expected_array = np.linspace(case["start"], case["end"], case["num"])
                
                # 检查形状
                shape_match = rm_tensor.shape == expected_array.shape
                if not shape_match:
                    passed = False
                    error_msg = f"形状不匹配: 期望{expected_array.shape}, 得到{rm_tensor.shape}"
                else:
                    # 检查值是否匹配
                    value_match, _ = check_tensor_match(rm_tensor, expected_array, case["device"])
                    
                    if not value_match:
                        passed = False
                        error_msg = "值不匹配"
                    elif not check_device_match(rm_tensor, case["device"]):
                        passed = False
                        error_msg = f"设备不匹配: 期望{case['device']}, 得到{rm_tensor.device}"
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
    
    def test_tensor(self):
        """测试tensor函数"""
        # 构建测试用例列表
        test_cases = []
        
        # 基本测试用例
        test_cases.extend([
            {"name": "标量创建", "data": 5},
            {"name": "一维列表创建", "data": [1, 2, 3, 4, 5]},
            {"name": "二维列表创建", "data": [[1, 2, 3], [4, 5, 6]]},
            {"name": "三维列表创建", "data": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]},
            {"name": "numpy数组创建", "data": np.array([1, 2, 3, 4, 5])},
            {"name": "numpy二维数组创建", "data": np.array([[1, 2, 3], [4, 5, 6]])},
        ])
        
        # dtype测试用例
        test_cases.extend([
            {"name": "指定float32类型", "data": [1, 2, 3], "dtype": rm.float32},
            {"name": "指定float64类型", "data": [1, 2, 3], "dtype": rm.float64},
            {"name": "指定int32类型", "data": [1, 2, 3], "dtype": rm.int32},
            {"name": "指定int64类型", "data": [1, 2, 3], "dtype": rm.int64},
            {"name": "指定bool类型", "data": [True, False, True], "dtype": rm.bool_},
            {"name": "指定complex64类型", "data": [1+2j, 3+4j], "dtype": rm.complex64},
        ])
        
        # device测试用例
        for device in device_list:
            device_name = device if device is not None else "默认设备"
            test_cases.append({
                "name": f"设备:{device_name}", 
                "data": [1, 2, 3], 
                "device": device
            })
        
        # numpy/cupy 跨设备转换测试用例
        # 1. data是numpy数组，device是GPU，无dtype参数
        if CUDA_AVAILABLE:
            test_cases.append({
                "name": "numpy数组转GPU设备", 
                "data": np.array([1, 2, 3, 4, 5]), 
                "device": "cuda"
            })
        
        # 2. data是cupy数组，device是CPU，无dtype参数
        if rm.cuda.CUPY_AVAILABLE:
            test_cases.append({
                "name": "cupy数组转CPU设备", 
                "data": rm.cuda.cp.array([1, 2, 3, 4, 5]), 
                "device": "cpu"
            })
        
        # 3. data是numpy数组，device是GPU，dtype与numpy数组不一致
        if CUDA_AVAILABLE:
            test_cases.append({
                "name": "numpy数组转GPU设备并转换dtype", 
                "data": np.array([1, 2, 3, 4, 5], dtype=np.int32), 
                "device": "cuda", 
                "dtype": rm.float32
            })
        
        # 4. data是cupy数组，device是CPU，dtype与cupy数组不一致
        if rm.cuda.CUPY_AVAILABLE:
            test_cases.append({
                "name": "cupy数组转CPU设备并转换dtype", 
                "data": rm.cuda.cp.array([1, 2, 3, 4, 5], dtype=rm.cuda.cp.float64), 
                "device": "cpu", 
                "dtype": rm.int64
            })
        
        # requires_grad测试用例
        test_cases.extend([
            {"name": "启用梯度计算", "data": [1.0, 2.0, 3.0], "requires_grad": True},
            {"name": "禁用梯度计算", "data": [1.0, 2.0, 3.0], "requires_grad": False},
        ])
        
        for case in test_cases:
            case_name = f"{self.test_tensor.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 提取参数
                data = case["data"]
                dtype = case.get("dtype", None)
                device = case.get("device", None)
                requires_grad = case.get("requires_grad", False)
                
                # 使用Riemann创建tensor张量
                rm_tensor = rm.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
                
                # 检查基本属性
                passed = True
                error_msg = ""
                
                # 检查数据是否正确
                # 处理cupy数组的转换
                if rm.cuda.CUPY_AVAILABLE and isinstance(data, rm.cuda.cp.ndarray):
                    expected_data = rm.cuda.cp.asnumpy(data)
                else:
                    expected_data = np.array(data)
                if dtype is not None:
                    expected_data = expected_data.astype(dtype)
                
                # 根据设备选择正确的比较方式
                if device and device.startswith("cuda") and CUDA_AVAILABLE:
                    # CUDA张量需要转换回CPU进行比较
                    rm_data = rm_tensor.to("cpu").detach().numpy()
                else:
                    rm_data = rm_tensor.detach().numpy()
                
                if not np.array_equal(rm_data, expected_data):
                    passed = False
                    error_msg = "值不匹配"
                
                # 检查数据类型
                if dtype is not None and rm_tensor.dtype != dtype:
                    passed = False
                    error_msg = f"数据类型不匹配: 期望{dtype}, 得到{rm_tensor.dtype}"
                
                # 检查设备
                if device:
                    expected_device = device if isinstance(device, str) else str(device)
                    actual_device = str(rm_tensor.device)
                    
                    # 处理设备名称格式差异："cuda" vs "cuda:0"
                    if expected_device == "cuda" and actual_device.startswith("cuda:"):
                        # 如果期望的是"cuda"，而实际是"cuda:X"，则认为匹配
                        pass
                    elif actual_device != expected_device:
                        passed = False
                        error_msg = f"设备不匹配: 期望{expected_device}, 得到{actual_device}"
                
                # 检查requires_grad
                if rm_tensor.requires_grad != requires_grad:
                    passed = False
                    error_msg = f"requires_grad不匹配: 期望{requires_grad}, 得到{rm_tensor.requires_grad}"
                
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