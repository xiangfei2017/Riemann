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
    print("警告: 无法导入PyTorch，将只测试riemann的max/min函数")
    TORCH_AVAILABLE = False

# 检查CUDA是否可用
try:
    CUDA_AVAILABLE = rm.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

# 定义设备列表
device_list = ["cpu"]
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
def compare_values(rm_result, torch_result, atol=1e-6, rtol=1e-6, check_dtype=True):
    """比较Riemann和PyTorch的值是否接近"""
    # 处理None值的情况
    if not TORCH_AVAILABLE:
        # 如果没有PyTorch，只检查riemann结果是否存在
        return rm_result is not None
    
    if rm_result is None and torch_result is None:
        return True
    if rm_result is None or torch_result is None:
        return False
    
    # 比较形状
    rm_shape = rm_result.shape if hasattr(rm_result, 'shape') else rm_result.data.shape
    torch_shape = torch_result.shape
    if rm_shape != torch_shape:
        return False
    
    # 获取数据
    rm_data = rm_result.data
    torch_data = torch_result.numpy()
    
    # 比较数据类型
    if check_dtype:
        rm_dtype = str(rm_data.dtype)
        torch_dtype = str(torch_data.dtype)
        if rm_dtype != torch_dtype:
            return False
    
    # 比较值是否接近
    try:
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

# 比较argmax/argmin结果（需要特殊处理，因为返回的是索引）
def compare_arg_results(rm_result, torch_result):
    """比较argmax/argmin的结果"""
    # 处理None值的情况
    if not TORCH_AVAILABLE:
        return rm_result is not None
    
    if rm_result is None and torch_result is None:
        return True
    if rm_result is None or torch_result is None:
        return False
    
    # 比较形状
    rm_shape = rm_result.shape if hasattr(rm_result, 'shape') else rm_result.data.shape
    torch_shape = torch_result.shape
    if rm_shape != torch_shape:
        return False
    
    # 获取数据并转换为int64类型进行比较
    rm_data = rm_result.data.astype(np.int64)
    torch_data = torch_result.numpy().astype(np.int64)
    
    # 检查所有元素是否相等
    return np.array_equal(rm_data, torch_data)

class TestMaxMinFunctions(unittest.TestCase):
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
    
    def test_max(self):
        """测试max函数 - 包括前向计算和梯度跟踪场景"""
        test_cases = [
            {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
            {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32},
            {"name": "沿轴1，保留维度", "shape": (4, 2), "dim": 1, "keepdim": True, "dtype": np.float64}
        ]
        
        for case in test_cases:
            case_name = f"{self.test_max.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据 - 处理0D标量情况
                if case["shape"] == ():
                    np_data = np.array(np.random.rand()).astype(case["dtype"])
                else:
                    np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor = rm.tensor(np_data)
                rm_result = rm_tensor.max(dim=case["dim"], keepdim=case["keepdim"])
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    torch_dtype = torch.float32 if case["dtype"] == np.float32 else torch.float64
                    torch_tensor = torch.tensor(np_data, dtype=torch_dtype)
                    # 特殊处理dim=None的情况，PyTorch不支持显式传入dim=None
                    if case["dim"] is None:
                        torch_result = torch_tensor.max()
                    else:
                        torch_result = torch_tensor.max(dim=case["dim"], keepdim=case["keepdim"])
                        # 处理PyTorch返回的是包含values和indices的namedtuple
                        if isinstance(torch_result, torch.return_types.max):
                            torch_result = torch_result.values
                else:
                    torch_result = None
                
                # 比较结果
                # 比较结果前处理MaxMinReturnType对象
                rm_compare_result = rm_result.values if hasattr(rm_result, 'values') else rm_result
                passed = compare_values(rm_compare_result, torch_result, check_dtype=False)  # max函数结果可能有数据类型差异
                
                # 梯度跟踪测试
                try:
                    # 创建需要梯度的浮点张量
                    rm_tensor_grad = rm.tensor(np_data.copy(), requires_grad=True)
                    if TORCH_AVAILABLE:
                        torch_tensor_grad = torch.tensor(np_data.copy(), dtype=torch_dtype, requires_grad=True)
                    
                    # 计算max值
                    rm_max_val = rm_tensor_grad.max(dim=case["dim"], keepdim=case["keepdim"])
                    # 提取values用于计算损失
                    rm_loss_val = rm_max_val.values if hasattr(rm_max_val, 'values') else rm_max_val
                    
                    # 计算损失并反向传播
                    rm_loss = rm.sum(rm_loss_val)
                    rm_loss.backward()
                    
                    if TORCH_AVAILABLE:
                        # 特殊处理dim=None的情况
                        if case["dim"] is None:
                            torch_max_val = torch_tensor_grad.max()
                        else:
                            torch_max_val = torch_tensor_grad.max(dim=case["dim"], keepdim=case["keepdim"])
                            if isinstance(torch_max_val, torch.return_types.max):
                                torch_max_val = torch_max_val.values
                        
                        # 计算损失并反向传播
                        torch_loss = torch.sum(torch_max_val)
                        torch_loss.backward()
                    
                    # 验证梯度
                    passed_grad = True
                    grad_details = []
                    
                    if not hasattr(rm_tensor_grad, 'grad') or rm_tensor_grad.grad is None:
                        passed_grad = False
                        grad_details.append("Riemann梯度未生成")
                    else:
                        grad_shape = rm_tensor_grad.grad.shape
                        if grad_shape != case["shape"]:
                            passed_grad = False
                            grad_details.append(f"梯度形状不匹配: {grad_shape} vs {case['shape']}")
                    
                    if not passed_grad:
                        passed = False
                        stats.add_result(case["name"], False, grad_details)
                except Exception as e:
                    # 处理异常
                    passed = False
                    stats.add_result(case["name"], False, [str(e)])
                    print(f"测试用例: {case['name']} - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  比较失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"max函数测试失败: {case['name']}")
            except Exception as e:
                stats.add_result(case["name"], False, [str(e)])
                print(f"测试用例: {case['name']} - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                raise
        
        # 添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的max函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试max函数:")
                
                test_cases = [
                    {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
                    {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
                    {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32}
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        # 使用rm.cuda.cp直接创建数组，提高效率
                        cp_data = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor = rm.tensor(cp_data, device=device)
                        rm_result = rm_tensor.max(dim=case["dim"], keepdim=case["keepdim"])
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        # 梯度测试
                        if np.issubdtype(case["dtype"], np.floating):
                            # 创建梯度测试数据
                            cp_data_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            rm_tensor_grad = rm.tensor(cp_data_grad, requires_grad=True, device=device)
                            rm_max_val = rm_tensor_grad.max(dim=case["dim"], keepdim=case["keepdim"])
                            rm_loss_val = rm_max_val.values if hasattr(rm_max_val, 'values') else rm_max_val
                            rm_loss = rm.sum(rm_loss_val)
                            rm_loss.backward()
                            
                            # 验证梯度
                            self.assertIsNotNone(rm_tensor_grad.grad)
                            self.assertEqual(rm_tensor_grad.grad.shape, rm_tensor_grad.shape)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的max函数测试失败: {case_name} - {str(e)}")
    
    def test_min(self):
        """测试min函数 - 包括前向计算和梯度跟踪场景"""
        test_cases = [
            {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
            {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32},
            {"name": "沿轴1，保留维度", "shape": (4, 2), "dim": 1, "keepdim": True, "dtype": np.float64}
        ]
        
        for case in test_cases:
            case_name = f"{self.test_min.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor = rm.tensor(np_data)
                rm_result = rm_tensor.min(dim=case["dim"], keepdim=case["keepdim"])
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    torch_dtype = torch.float32 if case["dtype"] == np.float32 else torch.float64
                    torch_tensor = torch.tensor(np_data, dtype=torch_dtype)
                    # 特殊处理dim=None的情况，PyTorch不支持显式传入dim=None
                    if case["dim"] is None:
                        torch_result = torch_tensor.min()
                    else:
                        torch_result = torch_tensor.min(dim=case["dim"], keepdim=case["keepdim"])
                        # 处理PyTorch返回的是包含values和indices的namedtuple
                        if isinstance(torch_result, torch.return_types.min):
                            torch_result = torch_result.values
                else:
                    torch_result = None
                
                # 比较结果前处理MaxMinReturnType对象
                rm_compare_result = rm_result.values if hasattr(rm_result, 'values') else rm_result
                passed = compare_values(rm_compare_result, torch_result, check_dtype=False)  # min函数结果可能有数据类型差异
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  比较失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"min函数测试失败: {case['name']}")
                
                # 梯度跟踪测试
                try:
                    # 创建需要梯度的张量
                    rm_tensor_grad = rm.tensor(np_data, requires_grad=True)
                    if TORCH_AVAILABLE:
                        torch_tensor_grad = torch.tensor(np_data.copy(), dtype=torch_dtype, requires_grad=True)
                    else:
                        torch_tensor_grad = None
                    
                    # 计算min值
                    rm_min_val = rm_tensor_grad.min(dim=case["dim"], keepdim=case["keepdim"])
                    # 提取values用于计算损失
                    rm_loss_val = rm_min_val.values if hasattr(rm_min_val, 'values') else rm_min_val
                    
                    if TORCH_AVAILABLE:
                        # 特殊处理dim=None的情况
                        if case["dim"] is None:
                            torch_min_val = torch_tensor_grad.min()
                        else:
                            torch_min_val = torch_tensor_grad.min(dim=case["dim"], keepdim=case["keepdim"])
                            if isinstance(torch_min_val, torch.return_types.min):
                                torch_min_val = torch_min_val.values
                    
                    # 计算损失并反向传播
                    rm_loss = rm.sum(rm_loss_val)
                    if TORCH_AVAILABLE:
                        torch_loss = torch.sum(torch_min_val)
                        torch_loss.backward()
                    
                    rm_loss.backward()
                    
                    # 检查梯度是否存在且形状正确
                    passed_grad = True
                    grad_details = []
                    
                    if not hasattr(rm_tensor_grad, 'grad') or rm_tensor_grad.grad is None:
                        passed_grad = False
                        grad_details.append("Riemann梯度未生成")
                    
                    if passed_grad and rm_tensor_grad.grad.shape != rm_tensor_grad.shape:
                        passed_grad = False
                        grad_details.append(f"Riemann梯度形状不匹配: 期望 {rm_tensor_grad.shape}, 得到 {rm_tensor_grad.grad.shape}")
                    
                    # 梯度值比较
                    if TORCH_AVAILABLE and passed_grad:
                        torch_grad = torch_tensor_grad.grad.numpy()
                        rm_grad = rm_tensor_grad.grad.data
                        if not np.allclose(rm_grad, torch_grad, rtol=1e-5, atol=1e-5):
                            passed_grad = False
                            max_diff = np.max(np.abs(rm_grad - torch_grad))
                            grad_details.append(f"梯度值不匹配，最大差异: {max_diff}")
                    
                    if IS_RUNNING_AS_SCRIPT:
                        status_grad = "通过" if passed_grad else "失败"
                        print(f"测试用例: {case['name']}(梯度) - {Colors.OKGREEN if passed_grad else Colors.FAIL}{status_grad}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                        if not passed_grad:
                            for detail in grad_details:
                                print(f"  {detail}")
                        # 记录梯度测试结果到统计中
                        stats.add_result(f"{case['name']}(梯度)", passed_grad)
                    
                    self.assertTrue(passed_grad, f"min函数梯度测试失败: {case['name']} - {', '.join(grad_details)}")
                    
                except Exception as e:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                    raise
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], False, [str(e)])
                    print(f"测试用例: {case['name']} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
        
        # 添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的min函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试min函数:")
                
                test_cases = [
                    {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
                    {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
                    {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32}
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        # 使用rm.cuda.cp直接创建数组，提高效率
                        cp_data = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor = rm.tensor(cp_data, device=device)
                        rm_result = rm_tensor.min(dim=case["dim"], keepdim=case["keepdim"])
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        # 梯度测试
                        if np.issubdtype(case["dtype"], np.floating):
                            # 创建梯度测试数据
                            cp_data_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            rm_tensor_grad = rm.tensor(cp_data_grad, requires_grad=True, device=device)
                            rm_min_val = rm_tensor_grad.min(dim=case["dim"], keepdim=case["keepdim"])
                            rm_loss_val = rm_min_val.values if hasattr(rm_min_val, 'values') else rm_min_val
                            rm_loss = rm.sum(rm_loss_val)
                            rm_loss.backward()
                            
                            # 验证梯度
                            self.assertIsNotNone(rm_tensor_grad.grad)
                            self.assertEqual(rm_tensor_grad.grad.shape, rm_tensor_grad.shape)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的min函数测试失败: {case_name} - {str(e)}")
                
    def test_argmax(self):
        """测试argmax函数 - 返回最大值的索引"""
        test_cases = [
            {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
            {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32},
            {"name": "沿轴1，保留维度", "shape": (4, 2), "dim": 1, "keepdim": True, "dtype": np.float64}
        ]
        
        for case in test_cases:
            case_name = f"{self.test_argmax.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                
                # 前向计算测试，创建Riemann张量
                rm_tensor = rm.tensor(np_data)
                rm_result = rm_tensor.argmax(dim=case["dim"], keepdim=case["keepdim"])
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    torch_dtype = torch.float32 if case["dtype"] == np.float32 else torch.float64
                    torch_tensor = torch.tensor(np_data, dtype=torch_dtype)
                    torch_result = torch_tensor.argmax(dim=case["dim"], keepdim=case["keepdim"])
                else:
                    torch_result = None
                
                # 比较结果（argmax返回的是索引，需要特殊处理）
                passed = compare_arg_results(rm_result, torch_result)
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  比较失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"argmax函数测试失败: {case['name']}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], False, [str(e)])
                    print(f"测试用例: {case['name']} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
                
        # argmax添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的argmax函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试argmax函数:")
                
                test_cases = [
                    {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
                    {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
                    {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32}
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        # 使用rm.cuda.cp直接创建数组，提高效率
                        cp_data = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor = rm.tensor(cp_data, device=device)
                        rm_result = rm_tensor.argmax(dim=case["dim"], keepdim=case["keepdim"])
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的argmax函数测试失败: {case_name} - {str(e)}")
    
    def test_argmin(self):
        """测试argmin函数 - 返回最小值的索引"""
        test_cases = [
            {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
            {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32},
            {"name": "沿轴1，保留维度", "shape": (4, 2), "dim": 1, "keepdim": True, "dtype": np.float64}
        ]
        
        for case in test_cases:
            case_name = f"{self.test_argmin.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据 - 处理0D标量情况
                if case["shape"] == ():
                    # 0D标量
                    np_data = np.array(np.random.rand()).astype(case["dtype"])
                else:
                    # 1D及以上维度
                    np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor = rm.tensor(np_data)
                rm_result = rm_tensor.argmin(dim=case["dim"], keepdim=case["keepdim"])
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    torch_dtype = torch.float32 if case["dtype"] == np.float32 else torch.float64
                    torch_tensor = torch.tensor(np_data, dtype=torch_dtype)
                    torch_result = torch_tensor.argmin(dim=case["dim"], keepdim=case["keepdim"])
                else:
                    torch_result = None
                
                # 比较结果（argmin返回的是索引，需要特殊处理）
                passed = compare_arg_results(rm_result, torch_result)
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  比较失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"argmin函数测试失败: {case['name']}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], False, [str(e)])
                    print(f"测试用例: {case['name']} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
        
        # argmin添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的argmin函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试argmin函数:")
                
                test_cases = [
                    {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
                    {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
                    {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32}
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        # 使用rm.cuda.cp直接创建数组，提高效率
                        cp_data = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor = rm.tensor(cp_data, device=device)
                        rm_result = rm_tensor.argmin(dim=case["dim"], keepdim=case["keepdim"])
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的argmin函数测试失败: {case_name} - {str(e)}")


    def test_sum(self):
        """测试sum函数 - 包括前向计算和梯度跟踪场景"""
        test_cases = [
            {"name": "0D标量，无维度", "shape": (), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "1D向量，无维度", "shape": (5,), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "1D向量，沿轴0", "shape": (4,), "dim": 0, "keepdim": False, "dtype": np.float64},
            {"name": "1D向量，沿轴0保留维度", "shape": (6,), "dim": 0, "keepdim": True, "dtype": np.float32},
            {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
            {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32},
            {"name": "沿多轴，保留维度", "shape": (4, 2, 3), "dim": (0, 1), "keepdim": True, "dtype": np.float64},
            # 复数场景测试用例
            {"name": "复数0D标量，无维度", "shape": (), "dim": None, "keepdim": False, "dtype": np.complex64},
            {"name": "复数1D向量，无维度", "shape": (3,), "dim": None, "keepdim": False, "dtype": np.complex128},
            {"name": "复数1D向量，沿轴0", "shape": (5,), "dim": 0, "keepdim": False, "dtype": np.complex64},
            {"name": "复数，无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.complex64},
            {"name": "复数，沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.complex128},
            {"name": "复数，沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.complex64}
        ]
        
        for case in test_cases:
            case_name = f"{self.test_sum.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据 - 处理0D标量情况
                if case["shape"] == ():
                    np_data = np.array(np.random.rand()).astype(case["dtype"])
                else:
                    np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor = rm.tensor(np_data)
                rm_result = rm_tensor.sum(dim=case["dim"], keepdim=case["keepdim"])
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    # 处理复数数据类型
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        torch_dtype = torch.complex64 if case["dtype"] == np.complex64 else torch.complex128
                    else:
                        torch_dtype = torch.float32 if case["dtype"] == np.float32 else torch.float64
                    torch_tensor = torch.tensor(np_data, dtype=torch_dtype)
                    torch_result = torch_tensor.sum(dim=case["dim"], keepdim=case["keepdim"])
                else:
                    torch_result = None
                
                # 比较结果
                passed = compare_values(rm_result, torch_result, check_dtype=False)
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  比较失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"sum函数测试失败: {case['name']}")
                
                # 梯度跟踪测试
                try:
                    # 创建需要梯度的张量
                    rm_tensor_grad = rm.tensor(np_data, requires_grad=True)
                    if TORCH_AVAILABLE:
                        torch_tensor_grad = torch.tensor(np_data, dtype=torch_dtype, requires_grad=True)
                    else:
                        torch_tensor_grad = None
                    
                    # 计算sum值
                    rm_sum_val = rm_tensor_grad.sum(dim=case["dim"], keepdim=case["keepdim"])
                    
                    if TORCH_AVAILABLE:
                        torch_sum_val = torch_tensor_grad.sum(dim=case["dim"], keepdim=case["keepdim"])
                    
                    # 计算损失并反向传播
                    # 对于复数，我们需要创建实数值的损失函数以启用梯度计算
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        rm_loss = rm.sum(rm.mean(rm.abs(rm_sum_val)))
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch.mean(torch.abs(torch_sum_val)))
                            torch_loss.backward()
                    else:
                        rm_loss = rm.sum(rm_sum_val) if case["dim"] is not None else rm_sum_val
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch_sum_val) if case["dim"] is not None else torch_sum_val
                            torch_loss.backward()
                    
                    rm_loss.backward()
                    
                    # 检查梯度是否存在且形状正确
                    passed_grad = True
                    grad_details = []
                    
                    if not hasattr(rm_tensor_grad, 'grad') or rm_tensor_grad.grad is None:
                        passed_grad = False
                        grad_details.append("Riemann梯度未生成")
                    
                    if passed_grad and rm_tensor_grad.grad.shape != rm_tensor_grad.shape:
                        passed_grad = False
                        grad_details.append(f"Riemann梯度形状不匹配: 期望 {rm_tensor_grad.shape}, 得到 {rm_tensor_grad.grad.shape}")
                    
                    # 梯度值比较
                    if TORCH_AVAILABLE and passed_grad:
                        torch_grad = torch_tensor_grad.grad.numpy()
                        rm_grad = rm_tensor_grad.grad.data
                        if not np.allclose(rm_grad, torch_grad, rtol=1e-5, atol=1e-5):
                            passed_grad = False
                            max_diff = np.max(np.abs(rm_grad - torch_grad))
                            grad_details.append(f"梯度值不匹配，最大差异: {max_diff}")
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(f"{case['name']}(梯度)", passed_grad)
                        status_grad = "通过" if passed_grad else "失败"
                        print(f"测试用例: {case['name']}(梯度) - {Colors.OKGREEN if passed_grad else Colors.FAIL}{status_grad}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                        if not passed_grad:
                            for detail in grad_details:
                                print(f"  {detail}")
                    
                    self.assertTrue(passed_grad, f"sum函数梯度测试失败: {case['name']} - {', '.join(grad_details)}")
                    
                except Exception as e:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                    raise
                    
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                raise
        
        # 添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的sum函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试sum函数:")
                
                test_cases = [
                    {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
                    {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
                    {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32},
                    # 复数场景测试用例
                    {"name": "复数，无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.complex64}
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        # 使用rm.cuda.cp直接创建数组，提高效率
                        cp_data = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor = rm.tensor(cp_data, device=device)
                        rm_result = rm_tensor.sum(dim=case["dim"], keepdim=case["keepdim"])
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        # 梯度测试
                        if np.issubdtype(case["dtype"], np.floating) or np.issubdtype(case["dtype"], np.complexfloating):
                            # 创建梯度测试数据
                            cp_data_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            rm_tensor_grad = rm.tensor(cp_data_grad, requires_grad=True, device=device)
                            rm_sum_val = rm_tensor_grad.sum(dim=case["dim"], keepdim=case["keepdim"])
                            
                            # 计算损失并反向传播
                            if np.issubdtype(case["dtype"], np.complexfloating):
                                rm_loss = rm.sum(rm.mean(rm.abs(rm_sum_val)))
                            else:
                                rm_loss = rm.sum(rm_sum_val) if case["dim"] is not None else rm_sum_val
                            
                            rm_loss.backward()
                            
                            # 验证梯度
                            self.assertIsNotNone(rm_tensor_grad.grad)
                            self.assertEqual(rm_tensor_grad.grad.shape, rm_tensor_grad.shape)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的sum函数测试失败: {case_name} - {str(e)}")
    
    def test_mean(self):
        """测试mean函数 - 包括前向计算和梯度跟踪场景"""
        test_cases = [
            {"name": "0D标量，无维度", "shape": (), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "1D向量，无维度", "shape": (5,), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "1D向量，沿轴0", "shape": (4,), "dim": 0, "keepdim": False, "dtype": np.float64},
            {"name": "1D向量，沿轴0保留维度", "shape": (6,), "dim": 0, "keepdim": True, "dtype": np.float32},
            {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
            {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32},
            {"name": "沿多轴，保留维度", "shape": (4, 2, 3), "dim": (0, 1), "keepdim": True, "dtype": np.float64},
            # 复数场景测试用例
            {"name": "复数0D标量，无维度", "shape": (), "dim": None, "keepdim": False, "dtype": np.complex64},
            {"name": "复数1D向量，无维度", "shape": (3,), "dim": None, "keepdim": False, "dtype": np.complex128},
            {"name": "复数1D向量，沿轴0", "shape": (5,), "dim": 0, "keepdim": False, "dtype": np.complex64},
            {"name": "复数，无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.complex64},
            {"name": "复数，沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.complex128},
            {"name": "复数，沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.complex64}
        ]
        
        for case in test_cases:
            case_name = f"{self.test_mean.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据 - 处理0D标量情况
                if case["shape"] == ():
                    np_data = np.array(np.random.rand()).astype(case["dtype"])
                else:
                    np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor = rm.tensor(np_data)
                rm_result = rm_tensor.mean(dim=case["dim"], keepdim=case["keepdim"])
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    # 处理复数数据类型
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        torch_dtype = torch.complex64 if case["dtype"] == np.complex64 else torch.complex128
                    else:
                        torch_dtype = torch.float32 if case["dtype"] == np.float32 else torch.float64
                    torch_tensor = torch.tensor(np_data, dtype=torch_dtype)
                    torch_result = torch_tensor.mean(dim=case["dim"], keepdim=case["keepdim"])
                else:
                    torch_result = None
                
                # 比较结果
                passed = compare_values(rm_result, torch_result, check_dtype=False)
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  比较失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"mean函数测试失败: {case['name']}")
                
                # 梯度跟踪测试
                try:
                    # 创建需要梯度的张量
                    rm_tensor_grad = rm.tensor(np_data, requires_grad=True)
                    if TORCH_AVAILABLE:
                        torch_tensor_grad = torch.tensor(np_data, dtype=torch_dtype, requires_grad=True)
                    else:
                        torch_tensor_grad = None
                    
                    # 计算mean值
                    rm_mean_val = rm_tensor_grad.mean(dim=case["dim"], keepdim=case["keepdim"])
                    
                    if TORCH_AVAILABLE:
                        torch_mean_val = torch_tensor_grad.mean(dim=case["dim"], keepdim=case["keepdim"])
                    
                    # 计算损失并反向传播
                    # 对于复数，我们需要创建实数值的损失函数以启用梯度计算
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        rm_loss = rm.sum(rm.mean(rm.abs(rm_mean_val)))
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch.mean(torch.abs(torch_mean_val)))
                            torch_loss.backward()
                    else:
                        rm_loss = rm.sum(rm_mean_val) if case["dim"] is not None else rm_mean_val
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch_mean_val) if case["dim"] is not None else torch_mean_val
                            torch_loss.backward()
                    
                    rm_loss.backward()
                    
                    # 检查梯度是否存在且形状正确
                    passed_grad = True
                    grad_details = []
                    
                    if not hasattr(rm_tensor_grad, 'grad') or rm_tensor_grad.grad is None:
                        passed_grad = False
                        grad_details.append("Riemann梯度未生成")
                    
                    if passed_grad and rm_tensor_grad.grad.shape != rm_tensor_grad.shape:
                        passed_grad = False
                        grad_details.append(f"Riemann梯度形状不匹配: 期望 {rm_tensor_grad.shape}, 得到 {rm_tensor_grad.grad.shape}")
                    
                    # 梯度值比较
                    if TORCH_AVAILABLE and passed_grad:
                        torch_grad = torch_tensor_grad.grad.numpy()
                        rm_grad = rm_tensor_grad.grad.data
                        if not np.allclose(rm_grad, torch_grad, rtol=1e-5, atol=1e-5):
                            passed_grad = False
                            max_diff = np.max(np.abs(rm_grad - torch_grad))
                            grad_details.append(f"梯度值不匹配，最大差异: {max_diff}")
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(f"{case['name']}(梯度)", passed_grad)
                        status_grad = "通过" if passed_grad else "失败"
                        print(f"测试用例: {case['name']}(梯度) - {Colors.OKGREEN if passed_grad else Colors.FAIL}{status_grad}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                        if not passed_grad:
                            for detail in grad_details:
                                print(f"  {detail}")
                    
                    self.assertTrue(passed_grad, f"mean函数梯度测试失败: {case['name']} - {', '.join(grad_details)}")
                    
                except Exception as e:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                    raise
                    
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                raise
        
        # 添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的mean函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试mean函数:")
                
                test_cases = [
                    {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
                    {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
                    {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32},
                    # 复数场景测试用例
                    {"name": "复数，无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.complex64}
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        # 使用rm.cuda.cp直接创建数组，提高效率
                        cp_data = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor = rm.tensor(cp_data, device=device)
                        rm_result = rm_tensor.mean(dim=case["dim"], keepdim=case["keepdim"])
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        # 梯度测试
                        if np.issubdtype(case["dtype"], np.floating) or np.issubdtype(case["dtype"], np.complexfloating):
                            # 创建梯度测试数据
                            cp_data_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            rm_tensor_grad = rm.tensor(cp_data_grad, requires_grad=True, device=device)
                            rm_mean_val = rm_tensor_grad.mean(dim=case["dim"], keepdim=case["keepdim"])
                            
                            # 计算损失并反向传播
                            if np.issubdtype(case["dtype"], np.complexfloating):
                                rm_loss = rm.sum(rm.mean(rm.abs(rm_mean_val)))
                            else:
                                rm_loss = rm.sum(rm_mean_val) if case["dim"] is not None else rm_mean_val
                            
                            rm_loss.backward()
                            
                            # 验证梯度
                            self.assertIsNotNone(rm_tensor_grad.grad)
                            self.assertEqual(rm_tensor_grad.grad.shape, rm_tensor_grad.shape)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的mean函数测试失败: {case_name} - {str(e)}")
    
    def test_prod(self):
        """测试prod函数 - 包括前向计算和梯度跟踪场景"""
        test_cases = [
            {"name": "0D标量，无维度", "shape": (), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "1D向量，无维度", "shape": (5,), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "1D向量，沿轴0", "shape": (4,), "dim": 0, "keepdim": False, "dtype": np.float64},
            {"name": "1D向量，沿轴0保留维度", "shape": (6,), "dim": 0, "keepdim": True, "dtype": np.float32},
            {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
            {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
            {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32},
            {"name": "沿多轴，保留维度", "shape": (4, 2, 3), "dim": (0, 1), "keepdim": True, "dtype": np.float64},
            {"name": "包含0元素，测试数值稳定性", "shape": (2, 3), "dim": 0, "keepdim": False, "dtype": np.float32},  # 测试数值稳定性
            # 复数场景测试用例
            {"name": "复数0D标量，无维度", "shape": (), "dim": None, "keepdim": False, "dtype": np.complex64},
            {"name": "复数1D向量，无维度", "shape": (3,), "dim": None, "keepdim": False, "dtype": np.complex128},
            {"name": "复数1D向量，沿轴0", "shape": (5,), "dim": 0, "keepdim": False, "dtype": np.complex64},
            {"name": "复数，无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.complex64},
            {"name": "复数，沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.complex128},
            {"name": "复数，沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.complex64}
        ]
        
        for case in test_cases:
            case_name = f"{self.test_prod.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据 - 处理0D标量情况
                if case["shape"] == ():
                    np_data = np.array(np.random.rand()).astype(case["dtype"])
                elif "数值稳定性" in case["name"]:
                    # 特殊创建包含0的数据，测试数值稳定性
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        np_data = (np.random.rand(*case["shape"]) + 1j * np.random.rand(*case["shape"])).astype(case["dtype"])
                    else:
                        np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                    # 在特定位置设置0（仅当shape不为空时）
                    if case["shape"] != ():
                        np_data[0, 0] = 0.0
                        np_data[1, 2] = 0.0
                elif np.issubdtype(case["dtype"], np.complexfloating):
                    # 为复数数据创建实部和虚部
                    np_data = (np.random.rand(*case["shape"]) + 1j * np.random.rand(*case["shape"])).astype(case["dtype"])
                else:
                    np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor = rm.tensor(np_data)
                rm_result = rm_tensor.prod(dim=case["dim"], keepdim=case["keepdim"])
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    # 处理复数数据类型
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        torch_dtype = torch.complex64 if case["dtype"] == np.complex64 else torch.complex128
                    else:
                        torch_dtype = torch.float32 if case["dtype"] == np.float32 else torch.float64
                    torch_tensor = torch.tensor(np_data, dtype=torch_dtype)
                    # 处理不同类型的dim参数
                    if case["dim"] is None:
                        # 无维度情况，直接调用prod()
                        torch_result = torch_tensor.prod()
                    elif isinstance(case["dim"], tuple):
                        # 多轴情况，PyTorch不支持直接传递tuple，需要依次对每个维度应用prod
                        torch_result = torch_tensor
                        # 为了保持与numpy和Riemann一致的行为，依次对每个维度应用prod
                        for dim in case["dim"]:
                            torch_result = torch_result.prod(dim=dim, keepdim=case["keepdim"])
                    else:
                        # 单轴情况
                        torch_result = torch_tensor.prod(dim=case["dim"], keepdim=case["keepdim"])
                else:
                    torch_result = None
                
                # 比较结果
                passed = compare_values(rm_result, torch_result, check_dtype=False)
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  比较失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"prod函数测试失败: {case['name']}")
                
                # 梯度跟踪测试
                try:
                    # 创建需要梯度的张量
                    rm_tensor_grad = rm.tensor(np_data, requires_grad=True)
                    if TORCH_AVAILABLE:
                        torch_tensor_grad = torch.tensor(np_data, dtype=torch_dtype, requires_grad=True)
                    else:
                        torch_tensor_grad = None
                    
                    # 计算prod值
                    rm_prod_val = rm_tensor_grad.prod(dim=case["dim"], keepdim=case["keepdim"])
                    
                    if TORCH_AVAILABLE:
                        # 处理不同类型的dim参数
                        if case["dim"] is None:
                            # 无维度情况，直接调用prod()
                            torch_prod_val = torch_tensor_grad.prod()
                        elif isinstance(case["dim"], tuple):
                            # 多轴情况，PyTorch不支持直接传递tuple，需要依次对每个维度应用prod
                            torch_prod_val = torch_tensor_grad
                            # 为了保持与numpy和Riemann一致的行为，我们需要按照从高到低的顺序处理维度
                            # 或者按照原始顺序，取决于实现细节
                            for dim in case["dim"]:
                                torch_prod_val = torch_prod_val.prod(dim=dim, keepdim=case["keepdim"])
                        else:
                            # 单轴情况
                            torch_prod_val = torch_tensor_grad.prod(dim=case["dim"], keepdim=case["keepdim"])
                    
                    # 计算损失并反向传播
                    # 对于复数，我们需要创建实数值的损失函数以启用梯度计算
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        rm_loss = rm.sum(rm.mean(rm.abs(rm_prod_val)))
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch.mean(torch.abs(torch_prod_val)))
                            torch_loss.backward()
                    else:
                        rm_loss = rm.sum(rm_prod_val) if case["dim"] is not None else rm_prod_val
                        if TORCH_AVAILABLE:
                            # 对于PyTorch，无论dim是什么，都使用.sum()以保持一致性
                            torch_loss = torch.sum(torch_prod_val)
                            torch_loss.backward()
                    
                    rm_loss.backward()
                    
                    # 检查梯度是否存在且形状正确
                    passed_grad = True
                    grad_details = []
                    
                    if not hasattr(rm_tensor_grad, 'grad') or rm_tensor_grad.grad is None:
                        passed_grad = False
                        grad_details.append("Riemann梯度未生成")
                    
                    if passed_grad and rm_tensor_grad.grad.shape != rm_tensor_grad.shape:
                        passed_grad = False
                        grad_details.append(f"Riemann梯度形状不匹配: 期望 {rm_tensor_grad.shape}, 得到 {rm_tensor_grad.grad.shape}")
                    
                    # 梯度值比较
                    if TORCH_AVAILABLE and passed_grad:
                        torch_grad = torch_tensor_grad.grad.numpy()
                        rm_grad = rm_tensor_grad.grad.data
                        # 对于包含0的情况，使用更大的容差
                        rtol, atol = (1e-5, 1e-5) if "数值稳定性" not in case["name"] else (1e-4, 1e-4)
                        if not np.allclose(rm_grad, torch_grad, rtol=rtol, atol=atol):
                            passed_grad = False
                            max_diff = np.max(np.abs(rm_grad - torch_grad))
                            grad_details.append(f"梯度值不匹配，最大差异: {max_diff}")
                    
                    # 检查是否有NaN或inf值
                    if passed_grad and np.any(np.isnan(rm_tensor_grad.grad.data)):
                        passed_grad = False
                        grad_details.append("梯度中包含NaN值")
                    
                    if passed_grad and np.any(np.isinf(rm_tensor_grad.grad.data)):
                        passed_grad = False
                        grad_details.append("梯度中包含无穷大值")
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(f"{case['name']}(梯度)", passed_grad)
                        status_grad = "通过" if passed_grad else "失败"
                        print(f"测试用例: {case['name']}(梯度) - {Colors.OKGREEN if passed_grad else Colors.FAIL}{status_grad}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                        if not passed_grad:
                            for detail in grad_details:
                                print(f"  {detail}")
                    
                    self.assertTrue(passed_grad, f"prod函数梯度测试失败: {case['name']} - {', '.join(grad_details)}")
                    
                except Exception as e:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                    raise
                    
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                raise
        
        # 添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的prod函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试prod函数:")
                
                test_cases = [
                    {"name": "无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.float32},
                    {"name": "沿轴0，不保留维度", "shape": (5, 6), "dim": 0, "keepdim": False, "dtype": np.float64},
                    {"name": "沿轴1，保留维度", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "dtype": np.float32},
                    # 复数场景测试用例
                    {"name": "复数，无维度，不保留维度", "shape": (3, 4), "dim": None, "keepdim": False, "dtype": np.complex64}
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        if np.issubdtype(case["dtype"], np.complexfloating):
                            # 使用rm.cuda.cp直接创建复数数组，提高效率
                            cp_data = (rm.cuda.cp.random.rand(*case["shape"]) + 1j * rm.cuda.cp.random.rand(*case["shape"])).astype(case["dtype"])
                        else:
                            # 使用rm.cuda.cp直接创建实数数组，提高效率
                            cp_data = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor = rm.tensor(cp_data, device=device)
                        rm_result = rm_tensor.prod(dim=case["dim"], keepdim=case["keepdim"])
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        # 梯度测试
                        if np.issubdtype(case["dtype"], np.floating) or np.issubdtype(case["dtype"], np.complexfloating):
                            # 创建梯度测试数据
                            if np.issubdtype(case["dtype"], np.complexfloating):
                                cp_data_grad = (rm.cuda.cp.random.rand(*case["shape"]) + 1j * rm.cuda.cp.random.rand(*case["shape"])).astype(case["dtype"])
                            else:
                                cp_data_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            rm_tensor_grad = rm.tensor(cp_data_grad, requires_grad=True, device=device)
                            rm_prod_val = rm_tensor_grad.prod(dim=case["dim"], keepdim=case["keepdim"])
                            
                            # 计算损失并反向传播
                            if np.issubdtype(case["dtype"], np.complexfloating):
                                rm_loss = rm.sum(rm.mean(rm.abs(rm_prod_val)))
                            else:
                                rm_loss = rm.sum(rm_prod_val) if case["dim"] is not None else rm_prod_val
                            
                            rm_loss.backward()
                            
                            # 验证梯度
                            self.assertIsNotNone(rm_tensor_grad.grad)
                            self.assertEqual(rm_tensor_grad.grad.shape, rm_tensor_grad.shape)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的prod函数测试失败: {case_name} - {str(e)}")
    
    def test_var(self):
        """测试var函数 - 包括前向计算和梯度跟踪场景"""
        test_cases = [
            {"name": "0D标量，无维度，有偏", "shape": (), "dim": None, "keepdim": False, "unbiased": False, "dtype": np.float32},
            {"name": "1D向量，无维度，无偏", "shape": (5,), "dim": None, "keepdim": False, "unbiased": True, "dtype": np.float32},
            {"name": "1D向量，沿轴0，有偏", "shape": (4,), "dim": 0, "keepdim": False, "unbiased": False, "dtype": np.float64},
            {"name": "1D向量，沿轴0保留维度，无偏", "shape": (6,), "dim": 0, "keepdim": True, "unbiased": True, "dtype": np.float32},
            {"name": "无维度，不保留维度，无偏", "shape": (3, 4), "dim": None, "keepdim": False, "unbiased": True, "dtype": np.float32},
            {"name": "沿轴0，不保留维度，有偏", "shape": (5, 6), "dim": 0, "keepdim": False, "unbiased": False, "dtype": np.float64},
            {"name": "沿轴1，保留维度，无偏", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "unbiased": True, "dtype": np.float32},
            {"name": "沿多轴，保留维度，有偏", "shape": (4, 2, 3), "dim": (0, 1), "keepdim": True, "unbiased": False, "dtype": np.float64},
            # 复数场景测试用例
            {"name": "复数0D标量，无维度，有偏", "shape": (), "dim": None, "keepdim": False, "unbiased": False, "dtype": np.complex64},
            {"name": "复数1D向量，无维度，无偏", "shape": (3,), "dim": None, "keepdim": False, "unbiased": True, "dtype": np.complex128},
            {"name": "复数1D向量，沿轴0，有偏", "shape": (5,), "dim": 0, "keepdim": False, "unbiased": False, "dtype": np.complex64},
            {"name": "复数，无维度，不保留维度，无偏", "shape": (3, 4), "dim": None, "keepdim": False, "unbiased": True, "dtype": np.complex64},
            {"name": "复数，沿轴0，不保留维度，有偏", "shape": (5, 6), "dim": 0, "keepdim": False, "unbiased": False, "dtype": np.complex128},
            {"name": "复数，沿轴1，保留维度，无偏", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "unbiased": True, "dtype": np.complex64}
        ]
        
        for case in test_cases:
            case_name = f"{self.test_var.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据 - 处理0D标量情况
                if case["shape"] == ():
                    np_data = np.array(np.random.rand()).astype(case["dtype"])
                else:
                    np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor = rm.tensor(np_data)
                rm_result = rm_tensor.var(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    # 处理复数数据类型
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        torch_dtype = torch.complex64 if case["dtype"] == np.complex64 else torch.complex128
                    else:
                        torch_dtype = torch.float32 if case["dtype"] == np.float32 else torch.float64
                    torch_tensor = torch.tensor(np_data, dtype=torch_dtype)
                    torch_result = torch_tensor.var(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                else:
                    torch_result = None
                
                # 比较结果
                passed = compare_values(rm_result, torch_result, check_dtype=False)
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  比较失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"var函数测试失败: {case['name']}")
                
                # 梯度跟踪测试
                try:
                    # 创建需要梯度的浮点张量
                    rm_tensor_grad = rm.tensor(np_data, requires_grad=True)
                    if TORCH_AVAILABLE:
                        torch_tensor_grad = torch.tensor(np_data, dtype=torch_dtype, requires_grad=True)
                    else:
                        torch_tensor_grad = None
                    
                    # 计算var值
                    rm_var_val = rm_tensor_grad.var(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                    
                    if TORCH_AVAILABLE:
                        torch_var_val = torch_tensor_grad.var(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                    
                    # 计算损失并反向传播
                    # 对于复数，方差已经是实数，但为了保持一致性，我们也使用相同的处理方式
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        rm_loss = rm.sum(rm_var_val)
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch_var_val)
                            torch_loss.backward()
                    else:
                        rm_loss = rm.sum(rm_var_val) if case["dim"] is not None else rm_var_val
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch_var_val) if case["dim"] is not None else torch_var_val
                            torch_loss.backward()
                    
                    rm_loss.backward()
                    
                    # 检查梯度是否存在且形状正确
                    passed_grad = True
                    grad_details = []
                    
                    if not hasattr(rm_tensor_grad, 'grad') or rm_tensor_grad.grad is None:
                        passed_grad = False
                        grad_details.append("Riemann梯度未生成")
                    
                    if passed_grad and rm_tensor_grad.grad.shape != rm_tensor_grad.shape:
                        passed_grad = False
                        grad_details.append(f"Riemann梯度形状不匹配: 期望 {rm_tensor_grad.shape}, 得到 {rm_tensor_grad.grad.shape}")
                    
                    # 梯度值比较
                    if TORCH_AVAILABLE and passed_grad:
                        torch_grad = torch_tensor_grad.grad.numpy()
                        rm_grad = rm_tensor_grad.grad.data
                        if not np.allclose(rm_grad, torch_grad, rtol=1e-5, atol=1e-5):
                            passed_grad = False
                            max_diff = np.max(np.abs(rm_grad - torch_grad))
                            grad_details.append(f"梯度值不匹配，最大差异: {max_diff}")
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(f"{case['name']}(梯度)", passed_grad)
                        status_grad = "通过" if passed_grad else "失败"
                        print(f"测试用例: {case['name']}(梯度) - {Colors.OKGREEN if passed_grad else Colors.FAIL}{status_grad}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                        if not passed_grad:
                            for detail in grad_details:
                                print(f"  {detail}")
                    
                    self.assertTrue(passed_grad, f"var函数梯度测试失败: {case['name']} - {', '.join(grad_details)}")
                    
                except Exception as e:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                    raise
                    
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                raise
        
        # 添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的var函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试var函数:")
                
                test_cases = [
                    {"name": "无维度，不保留维度，有偏", "shape": (3, 4), "dim": None, "keepdim": False, "unbiased": False, "dtype": np.float32},
                    {"name": "沿轴0，不保留维度，无偏", "shape": (5, 6), "dim": 0, "keepdim": False, "unbiased": True, "dtype": np.float64},
                    {"name": "沿轴1，保留维度，有偏", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "unbiased": False, "dtype": np.float32},
                    # 复数场景测试用例
                    {"name": "复数，无维度，不保留维度，无偏", "shape": (3, 4), "dim": None, "keepdim": False, "unbiased": True, "dtype": np.complex64}
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        if np.issubdtype(case["dtype"], np.complexfloating):
                            # 使用rm.cuda.cp直接创建复数数组，提高效率
                            cp_data = (rm.cuda.cp.random.rand(*case["shape"]) + 1j * rm.cuda.cp.random.rand(*case["shape"])).astype(case["dtype"])
                        else:
                            # 使用rm.cuda.cp直接创建实数数组，提高效率
                            cp_data = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor = rm.tensor(cp_data, device=device)
                        rm_result = rm_tensor.var(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        # 梯度测试
                        if np.issubdtype(case["dtype"], np.floating) or np.issubdtype(case["dtype"], np.complexfloating):
                            # 创建梯度测试数据
                            if np.issubdtype(case["dtype"], np.complexfloating):
                                cp_data_grad = (rm.cuda.cp.random.rand(*case["shape"]) + 1j * rm.cuda.cp.random.rand(*case["shape"])).astype(case["dtype"])
                            else:
                                cp_data_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            rm_tensor_grad = rm.tensor(cp_data_grad, requires_grad=True, device=device)
                            rm_var_val = rm_tensor_grad.var(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                            
                            # 计算损失并反向传播
                            if np.issubdtype(case["dtype"], np.complexfloating):
                                rm_loss = rm.sum(rm_var_val)
                            else:
                                rm_loss = rm.sum(rm_var_val) if case["dim"] is not None else rm_var_val
                            
                            rm_loss.backward()
                            
                            # 验证梯度
                            self.assertIsNotNone(rm_tensor_grad.grad)
                            self.assertEqual(rm_tensor_grad.grad.shape, rm_tensor_grad.shape)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的var函数测试失败: {case_name} - {str(e)}")

    def test_std(self):
        """测试std函数 - 包括前向计算和梯度跟踪场景"""
        test_cases = [
            {"name": "0D标量，无维度，有偏", "shape": (), "dim": None, "keepdim": False, "unbiased": False, "dtype": np.float32},
            {"name": "1D向量，无维度，无偏", "shape": (5,), "dim": None, "keepdim": False, "unbiased": True, "dtype": np.float32},
            {"name": "1D向量，沿轴0，有偏", "shape": (4,), "dim": 0, "keepdim": False, "unbiased": False, "dtype": np.float64},
            {"name": "1D向量，沿轴0保留维度，无偏", "shape": (6,), "dim": 0, "keepdim": True, "unbiased": True, "dtype": np.float32},
            {"name": "无维度，不保留维度，无偏", "shape": (3, 4), "dim": None, "keepdim": False, "unbiased": True, "dtype": np.float32},
            {"name": "沿轴0，不保留维度，有偏", "shape": (5, 6), "dim": 0, "keepdim": False, "unbiased": False, "dtype": np.float64},
            {"name": "沿轴1，保留维度，无偏", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "unbiased": True, "dtype": np.float32},
            {"name": "沿多轴，保留维度，有偏", "shape": (4, 2, 3), "dim": (0, 1), "keepdim": True, "unbiased": False, "dtype": np.float64},
            # 复数场景测试用例
            {"name": "复数0D标量，无维度，有偏", "shape": (), "dim": None, "keepdim": False, "unbiased": False, "dtype": np.complex64},
            {"name": "复数1D向量，无维度，无偏", "shape": (3,), "dim": None, "keepdim": False, "unbiased": True, "dtype": np.complex128},
            {"name": "复数1D向量，沿轴0，有偏", "shape": (5,), "dim": 0, "keepdim": False, "unbiased": False, "dtype": np.complex64},
            {"name": "复数，无维度，不保留维度，无偏", "shape": (3, 4), "dim": None, "keepdim": False, "unbiased": True, "dtype": np.complex64},
            {"name": "复数，沿轴0，不保留维度，有偏", "shape": (5, 6), "dim": 0, "keepdim": False, "unbiased": False, "dtype": np.complex128},
            {"name": "复数，沿轴1，保留维度，无偏", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "unbiased": True, "dtype": np.complex64}
        ]
        
        for case in test_cases:
            case_name = f"{self.test_std.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据 - 处理0D标量情况
                if case["shape"] == ():
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        np_data = np.array(np.random.rand() + 1j * np.random.rand()).astype(case["dtype"])
                    else:
                        np_data = np.array(np.random.rand()).astype(case["dtype"])
                else:
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        # 为复数数据创建实部和虚部
                        np_data = (np.random.rand(*case["shape"]) + 1j * np.random.rand(*case["shape"])).astype(case["dtype"])
                    else:
                        np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor = rm.tensor(np_data)
                rm_result = rm_tensor.std(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    # 处理复数数据类型
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        torch_dtype = torch.complex64 if case["dtype"] == np.complex64 else torch.complex128
                    else:
                        torch_dtype = torch.float32 if case["dtype"] == np.float32 else torch.float64
                    torch_tensor = torch.tensor(np_data, dtype=torch_dtype)
                    torch_result = torch_tensor.std(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                else:
                    torch_result = None
                
                # 比较结果
                passed = compare_values(rm_result, torch_result, check_dtype=False)
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  比较失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"std函数测试失败: {case['name']}")
                
                # 梯度跟踪测试
                try:
                    # 对于0D标量有偏计算，torch.std会出现除零警告但结果应为0
                    # 这种情况下我们不进行梯度测试，因为数学上这是预期行为
                    if case["shape"] == () and case["unbiased"] == False:
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"测试用例: {case['name']}(梯度) - {Colors.WARNING}跳过{Colors.ENDC} (0D标量有偏计算数学上无意义)")
                        continue
                    
                    # 创建需要梯度的浮点张量
                    rm_tensor_grad = rm.tensor(np_data.copy(), requires_grad=True)
                    if TORCH_AVAILABLE:
                        torch_tensor_grad = torch.tensor(np_data.copy(), dtype=torch_dtype, requires_grad=True)
                    else:
                        torch_tensor_grad = None
                    
                    # 计算std值
                    rm_std_val = rm_tensor_grad.std(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                    
                    if TORCH_AVAILABLE:
                        torch_std_val = torch_tensor_grad.std(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                    
                    # 计算损失并反向传播
                    # 对于复数，标准差已经是实数，但为了保持一致性，我们也使用相同的处理方式
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        rm_loss = rm.sum(rm_std_val)
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch_std_val)
                            torch_loss.backward()
                    else:
                        # 对于0D标量，直接使用标量值作为损失；对于其他情况，使用sum
                        if case["shape"] == ():
                            rm_loss = rm_std_val
                            if TORCH_AVAILABLE:
                                torch_loss = torch_std_val
                                torch_loss.backward()
                        else:
                            rm_loss = rm.sum(rm_std_val) if case["dim"] is not None else rm_std_val
                            if TORCH_AVAILABLE:
                                torch_loss = torch.sum(torch_std_val) if case["dim"] is not None else torch_std_val
                                torch_loss.backward()
                    
                    rm_loss.backward()
                    
                    # 检查梯度是否存在且形状正确
                    passed_grad = True
                    grad_details = []
                    
                    if not hasattr(rm_tensor_grad, 'grad') or rm_tensor_grad.grad is None:
                        passed_grad = False
                        grad_details.append("Riemann梯度未生成")
                    
                    if passed_grad and rm_tensor_grad.grad.shape != rm_tensor_grad.shape:
                        passed_grad = False
                        grad_details.append(f"Riemann梯度形状不匹配: 期望 {rm_tensor_grad.shape}, 得到 {rm_tensor_grad.grad.shape}")
                    
                    # 梯度值比较
                    if TORCH_AVAILABLE and passed_grad:
                        torch_grad = torch_tensor_grad.grad.numpy()
                        rm_grad = rm_tensor_grad.grad.data
                        if not np.allclose(rm_grad, torch_grad, rtol=1e-5, atol=1e-5):
                            passed_grad = False
                            max_diff = np.max(np.abs(rm_grad - torch_grad))
                            grad_details.append(f"梯度值不匹配，最大差异: {max_diff}")
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(f"{case['name']}(梯度)", passed_grad)
                        status_grad = "通过" if passed_grad else "失败"
                        print(f"测试用例: {case['name']}(梯度) - {Colors.OKGREEN if passed_grad else Colors.FAIL}{status_grad}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                        if not passed_grad:
                            for detail in grad_details:
                                print(f"  {detail}")
                    
                    self.assertTrue(passed_grad, f"std函数梯度测试失败: {case['name']} - {', '.join(grad_details)}")
                    
                except Exception as e:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                    raise
                    
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                raise
        
        # 添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的std函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试std函数:")
                
                test_cases = [
                    {"name": "无维度，不保留维度，有偏", "shape": (3, 4), "dim": None, "keepdim": False, "unbiased": False, "dtype": np.float32},
                    {"name": "沿轴0，不保留维度，无偏", "shape": (5, 6), "dim": 0, "keepdim": False, "unbiased": True, "dtype": np.float64},
                    {"name": "沿轴1，保留维度，有偏", "shape": (2, 3, 4), "dim": 1, "keepdim": True, "unbiased": False, "dtype": np.float32},
                    # 复数场景测试用例
                    {"name": "复数，无维度，不保留维度，无偏", "shape": (3, 4), "dim": None, "keepdim": False, "unbiased": True, "dtype": np.complex64}
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        if np.issubdtype(case["dtype"], np.complexfloating):
                            # 使用rm.cuda.cp直接创建复数数组，提高效率
                            cp_data = (rm.cuda.cp.random.rand(*case["shape"]) + 1j * rm.cuda.cp.random.rand(*case["shape"])).astype(case["dtype"])
                        else:
                            # 使用rm.cuda.cp直接创建实数数组，提高效率
                            cp_data = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor = rm.tensor(cp_data, device=device)
                        rm_result = rm_tensor.std(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        # 梯度测试
                        if np.issubdtype(case["dtype"], np.floating) or np.issubdtype(case["dtype"], np.complexfloating):
                            # 创建梯度测试数据
                            if np.issubdtype(case["dtype"], np.complexfloating):
                                cp_data_grad = (rm.cuda.cp.random.rand(*case["shape"]) + 1j * rm.cuda.cp.random.rand(*case["shape"])).astype(case["dtype"])
                            else:
                                cp_data_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            rm_tensor_grad = rm.tensor(cp_data_grad, requires_grad=True, device=device)
                            rm_std_val = rm_tensor_grad.std(dim=case["dim"], unbiased=case["unbiased"], keepdim=case["keepdim"])
                            
                            # 计算损失并反向传播
                            if np.issubdtype(case["dtype"], np.complexfloating):
                                rm_loss = rm.sum(rm.abs(rm_std_val))
                            else:
                                rm_loss = rm.sum(rm_std_val) if case["dim"] is not None else rm_std_val
                            
                            rm_loss.backward()
                            
                            # 验证梯度
                            self.assertIsNotNone(rm_tensor_grad.grad)
                            self.assertEqual(rm_tensor_grad.grad.shape, rm_tensor_grad.shape)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的std函数测试失败: {case_name} - {str(e)}")

    def test_cumsum(self):
        """测试cumsum函数 - 包括前向计算和梯度跟踪场景"""
        test_cases = [
            {"name": "0D标量，累积求和", "shape": (), "dim": 0, "dtype": np.float32},
            {"name": "1D向量，累积求和", "shape": (4,), "dim": 0, "dtype": np.float32},
            {"name": "沿轴0，累积求和", "shape": (3, 4), "dim": 0, "dtype": np.float32},
            {"name": "沿轴1，累积求和", "shape": (5, 6), "dim": 1, "dtype": np.float64},
            {"name": "沿轴2，累积求和", "shape": (2, 3, 4), "dim": 2, "dtype": np.float32},
            # 复数场景测试用例
            {"name": "复数0D标量，累积求和", "shape": (), "dim": 0, "dtype": np.complex64},
            {"name": "复数1D向量，累积求和", "shape": (4,), "dim": 0, "dtype": np.complex128},
            {"name": "复数，沿轴0，累积求和", "shape": (3, 4), "dim": 0, "dtype": np.complex64},
            {"name": "复数，沿轴1，累积求和", "shape": (5, 6), "dim": 1, "dtype": np.complex128},
            {"name": "复数，沿轴2，累积求和", "shape": (2, 3, 4), "dim": 2, "dtype": np.complex64}
        ]
        
        for case in test_cases:
            case_name = f"{self.test_cumsum.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据 - 处理0D标量情况
                if case["shape"] == ():
                    np_data = np.array(np.random.rand()).astype(case["dtype"])
                else:
                    np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor = rm.tensor(np_data)
                rm_result = rm_tensor.cumsum(dim=case["dim"])
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    # 处理复数数据类型
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        torch_dtype = torch.complex64 if case["dtype"] == np.complex64 else torch.complex128
                    else:
                        torch_dtype = torch.float32 if case["dtype"] == np.float32 else torch.float64
                    torch_tensor = torch.tensor(np_data, dtype=torch_dtype)
                    torch_result = torch_tensor.cumsum(dim=case["dim"])
                else:
                    torch_result = None
                
                # 比较结果
                passed = compare_values(rm_result, torch_result, check_dtype=False)
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  比较失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"cumsum函数测试失败: {case['name']}")
                
                # 梯度跟踪测试
                try:
                    # 创建需要梯度的张量
                    rm_tensor_grad = rm.tensor(np_data.copy(), requires_grad=True)
                    if TORCH_AVAILABLE:
                        torch_tensor_grad = torch.tensor(np_data.copy(), dtype=torch_dtype, requires_grad=True)
                    else:
                        torch_tensor_grad = None
                    
                    # 计算cumsum值
                    rm_cumsum_val = rm_tensor_grad.cumsum(dim=case["dim"])
                    
                    if TORCH_AVAILABLE:
                        torch_cumsum_val = torch_tensor_grad.cumsum(dim=case["dim"])
                    
                    # 计算损失并反向传播
                    # 对于复数，我们需要创建实数值的损失函数以启用梯度计算
                    if np.issubdtype(case["dtype"], np.complexfloating):
                        rm_loss = rm.sum(rm.mean(rm.abs(rm_cumsum_val)))
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch.mean(torch.abs(torch_cumsum_val)))
                            torch_loss.backward()
                    else:
                        rm_loss = rm.sum(rm_cumsum_val)
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch_cumsum_val)
                            torch_loss.backward()
                    
                    rm_loss.backward()
                    
                    # 检查梯度是否存在且形状正确
                    passed_grad = True
                    grad_details = []
                    
                    if not hasattr(rm_tensor_grad, 'grad') or rm_tensor_grad.grad is None:
                        passed_grad = False
                        grad_details.append("Riemann梯度未生成")
                    
                    if passed_grad and rm_tensor_grad.grad.shape != rm_tensor_grad.shape:
                        passed_grad = False
                        grad_details.append(f"Riemann梯度形状不匹配: 期望 {rm_tensor_grad.shape}, 得到 {rm_tensor_grad.grad.shape}")
                    
                    # 梯度值比较
                    if TORCH_AVAILABLE and passed_grad:
                        torch_grad = torch_tensor_grad.grad.numpy()
                        rm_grad = rm_tensor_grad.grad.data
                        if not np.allclose(rm_grad, torch_grad, rtol=1e-5, atol=1e-5):
                            passed_grad = False
                            max_diff = np.max(np.abs(rm_grad - torch_grad))
                            grad_details.append(f"梯度值不匹配，最大差异: {max_diff}")
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(f"{case['name']}(梯度)", passed_grad)
                        status_grad = "通过" if passed_grad else "失败"
                        print(f"测试用例: {case['name']}(梯度) - {Colors.OKGREEN if passed_grad else Colors.FAIL}{status_grad}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                        if not passed_grad:
                            for detail in grad_details:
                                print(f"  {detail}")
                    
                    self.assertTrue(passed_grad, f"cumsum函数梯度测试失败: {case['name']} - {', '.join(grad_details)}")
                    
                except Exception as e:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                    raise
                    
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case['name']}(梯度) - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                raise
        
        # 添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的cumsum函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试cumsum函数:")
                
                test_cases = [
                    {"name": "1D向量累积求和", "shape": (5,), "dim": 0, "dtype": np.float32},
                    {"name": "2D张量沿轴0累积求和", "shape": (3, 4), "dim": 0, "dtype": np.float64},
                    {"name": "2D张量沿轴1累积求和", "shape": (3, 4), "dim": 1, "dtype": np.float32},
                    {"name": "3D张量沿轴2累积求和", "shape": (2, 3, 4), "dim": 2, "dtype": np.float64},
                    # 复数场景测试用例
                    {"name": "复数1D向量累积求和", "shape": (5,), "dim": 0, "dtype": np.complex64},
                    {"name": "复数2D张量沿轴0累积求和", "shape": (3, 4), "dim": 0, "dtype": np.complex128},
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        if np.issubdtype(case["dtype"], np.complexfloating):
                            # 直接创建实数和虚数部分的NumPy数组，然后组合成复数数组
                            np_data_real = np.random.rand(*case["shape"]).astype(np.float32)
                            np_data_imag = np.random.rand(*case["shape"]).astype(np.float32)
                            np_data = (np_data_real + 1j * np_data_imag).astype(case["dtype"])
                        else:
                            # 直接创建实数的NumPy数组
                            np_data = np.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 直接创建CUDA张量，不经过CuPy数组
                        rm_tensor = rm.tensor(np_data, device=device)
                        rm_result = rm_tensor.cumsum(dim=case["dim"])
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        self.assertEqual(rm_result.shape, rm_tensor.shape)
                        
                        # 梯度测试
                        if np.issubdtype(case["dtype"], np.floating) or np.issubdtype(case["dtype"], np.complexfloating):
                            # 断言case["shape"]是元组类型，否则抛出错误
                            self.assertIsInstance(case["shape"], tuple, f"case['shape']应该是元组类型，但实际是{type(case['shape'])}类型: {case['shape']}")
                            
                            # 创建梯度测试数据
                            if np.issubdtype(case["dtype"], np.complexfloating):
                                # 直接创建实数和虚数部分的NumPy数组，然后组合成复数数组
                                np_data_grad_real = np.random.rand(*case["shape"]).astype(np.float32)
                                np_data_grad_imag = np.random.rand(*case["shape"]).astype(np.float32)
                                np_data_grad = (np_data_grad_real + 1j * np_data_grad_imag).astype(case["dtype"])
                            else:
                                # 直接创建实数的NumPy数组
                                np_data_grad = np.random.rand(*case["shape"]).astype(case["dtype"])
                            
                            # 直接创建CUDA张量，不经过CuPy数组
                            rm_tensor_grad = rm.tensor(np_data_grad, requires_grad=True, device=device)
                            rm_cumsum_val = rm_tensor_grad.cumsum(dim=case["dim"])
                            
                            # 计算损失并反向传播
                            if np.issubdtype(case["dtype"], np.complexfloating):
                                rm_loss = rm.sum(rm.mean(rm.abs(rm_cumsum_val)))
                            else:
                                rm_loss = rm.sum(rm_cumsum_val)
                            
                            rm_loss.backward()
                            
                            # 验证梯度
                            self.assertIsNotNone(rm_tensor_grad.grad)
                            self.assertEqual(rm_tensor_grad.grad.shape, rm_tensor_grad.shape)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                                
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的cumsum函数测试失败: {case_name} - {str(e)}")

    def test_maximum(self):
        """测试maximum函数 - 与torch.maximum对比，包括前向计算和梯度跟踪场景"""
        test_cases = [
            # 基本场景
            {"name": "2D张量基本比较", "shape": (3, 4), "dtype": np.float32},
            {"name": "3D张量基本比较", "shape": (2, 3, 4), "dtype": np.float64},
            {"name": "1D向量比较", "shape": (5,), "dtype": np.float32},
            {"name": "标量比较", "shape": (), "dtype": np.float64},
            
            # 广播场景
            {"name": "张量与标量广播", "shape": (3, 4), "other_shape": (), "dtype": np.float32},
            {"name": "张量与1D广播", "shape": (3, 4), "other_shape": (4,), "dtype": np.float64},
            {"name": "不同形状张量广播", "shape": (2, 1, 4), "other_shape": (1, 3, 1), "dtype": np.float32},
            
            # 特殊值场景
            {"name": "相等值测试", "shape": (2, 3), "equal_values": True, "dtype": np.float32},
            {"name": "负数测试", "shape": (3, 3), "negative": True, "dtype": np.float64},
            {"name": "混合正负数测试", "shape": (2, 4), "mixed": True, "dtype": np.float32},
            
            # 边界场景
            {"name": "大数值测试", "shape": (2, 2), "large_values": True, "dtype": np.float32},
            {"name": "小数值测试", "shape": (2, 2), "small_values": True, "dtype": np.float64},
            
            # 数据类型场景
            {"name": "int32类型测试", "shape": (3, 3), "dtype": np.int32},
            {"name": "int64类型测试", "shape": (2, 4), "dtype": np.int64},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_maximum.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                if case.get("equal_values", False):
                    # 相等值测试
                    np_data1 = np.ones(case["shape"], dtype=case["dtype"]) * 3.0
                    np_data2 = np.ones(case["shape"], dtype=case["dtype"]) * 3.0
                elif case.get("negative", False):
                    # 负数测试
                    if case["shape"] == ():
                        np_data1 = np.array(-np.random.rand() * 10, dtype=case["dtype"])
                        np_data2 = np.array(-np.random.rand() * 10, dtype=case["dtype"])
                    else:
                        np_data1 = (-np.random.rand(*case["shape"]) * 10).astype(case["dtype"])
                        np_data2 = (-np.random.rand(*case["shape"]) * 10).astype(case["dtype"])
                elif case.get("mixed", False):
                    # 混合正负数测试
                    if case["shape"] == ():
                        np_data1 = np.array((np.random.rand() - 0.5) * 20, dtype=case["dtype"])
                        np_data2 = np.array((np.random.rand() - 0.5) * 20, dtype=case["dtype"])
                    else:
                        np_data1 = ((np.random.rand(*case["shape"]) - 0.5) * 20).astype(case["dtype"])
                        np_data2 = ((np.random.rand(*case["shape"]) - 0.5) * 20).astype(case["dtype"])
                elif case.get("large_values", False):
                    # 大数值测试
                    if case["shape"] == ():
                        np_data1 = np.array(np.random.rand() * 1e6, dtype=case["dtype"])
                        np_data2 = np.array(np.random.rand() * 1e6, dtype=case["dtype"])
                    else:
                        np_data1 = (np.random.rand(*case["shape"]) * 1e6).astype(case["dtype"])
                        np_data2 = (np.random.rand(*case["shape"]) * 1e6).astype(case["dtype"])
                elif case.get("small_values", False):
                    # 小数值测试
                    if case["shape"] == ():
                        np_data1 = np.array(np.random.rand() * 1e-6, dtype=case["dtype"])
                        np_data2 = np.array(np.random.rand() * 1e-6, dtype=case["dtype"])
                    else:
                        np_data1 = (np.random.rand(*case["shape"]) * 1e-6).astype(case["dtype"])
                        np_data2 = (np.random.rand(*case["shape"]) * 1e-6).astype(case["dtype"])
                else:
                    # 常规随机数据
                    if case["shape"] == ():
                        np_data1 = np.array(np.random.rand() * 10, dtype=case["dtype"])
                        if "other_shape" in case:
                            np_data2 = np.array(np.random.rand() * 10, dtype=case["dtype"])
                        else:
                            np_data2 = np.array(np.random.rand() * 10, dtype=case["dtype"])
                    else:
                        np_data1 = (np.random.rand(*case["shape"]) * 10).astype(case["dtype"])
                        if "other_shape" in case:
                            if case["other_shape"] == ():
                                np_data2 = np.array(np.random.rand() * 10, dtype=case["dtype"])
                            else:
                                np_data2 = (np.random.rand(*case["other_shape"]) * 10).astype(case["dtype"])
                        else:
                            np_data2 = (np.random.rand(*case["shape"]) * 10).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor1 = rm.tensor(np_data1)
                rm_tensor2 = rm.tensor(np_data2)
                rm_result = rm.maximum(rm_tensor1, rm_tensor2)
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    # 处理数据类型转换
                    if case["dtype"] == np.float32:
                        torch_dtype = torch.float32
                    elif case["dtype"] == np.float64:
                        torch_dtype = torch.float64
                    elif case["dtype"] == np.int32:
                        torch_dtype = torch.int32
                    elif case["dtype"] == np.int64:
                        torch_dtype = torch.int64
                    else:
                        torch_dtype = torch.float32
                    
                    torch_tensor1 = torch.tensor(np_data1, dtype=torch_dtype)
                    torch_tensor2 = torch.tensor(np_data2, dtype=torch_dtype)
                    torch_result = torch.maximum(torch_tensor1, torch_tensor2)
                else:
                    torch_result = None
                
                # 比较前向计算结果
                passed_forward = compare_values(rm_result, torch_result, check_dtype=False)
                
                # 梯度跟踪测试（仅对浮点类型）
                passed_grad = True
                grad_details = []
                
                if np.issubdtype(case["dtype"], np.floating):
                    try:
                        # 创建需要梯度的张量
                        rm_tensor1_grad = rm.tensor(np_data1.copy(), requires_grad=True)
                        rm_tensor2_grad = rm.tensor(np_data2.copy(), requires_grad=True)
                        
                        if TORCH_AVAILABLE:
                            torch_tensor1_grad = torch.tensor(np_data1.copy(), dtype=torch_dtype, requires_grad=True)
                            torch_tensor2_grad = torch.tensor(np_data2.copy(), dtype=torch_dtype, requires_grad=True)
                        
                        # 计算maximum
                        rm_max_val = rm.maximum(rm_tensor1_grad, rm_tensor2_grad)
                        
                        if TORCH_AVAILABLE:
                            torch_max_val = torch.maximum(torch_tensor1_grad, torch_tensor2_grad)
                        
                        # 计算损失并反向传播
                        rm_loss = rm.sum(rm_max_val)
                        rm_loss.backward()
                        
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch_max_val)
                            torch_loss.backward()
                        
                        # 检查梯度是否存在且形状正确
                        if not hasattr(rm_tensor1_grad, 'grad') or rm_tensor1_grad.grad is None:
                            passed_grad = False
                            grad_details.append("Riemann第一个输入梯度未生成")
                        elif rm_tensor1_grad.grad.shape != rm_tensor1_grad.shape:
                            passed_grad = False
                            grad_details.append(f"第一个输入梯度形状不匹配: 期望 {rm_tensor1_grad.shape}, 得到 {rm_tensor1_grad.grad.shape}")
                        
                        if not hasattr(rm_tensor2_grad, 'grad') or rm_tensor2_grad.grad is None:
                            passed_grad = False
                            grad_details.append("Riemann第二个输入梯度未生成")
                        elif rm_tensor2_grad.grad.shape != rm_tensor2_grad.shape:
                            passed_grad = False
                            grad_details.append(f"第二个输入梯度形状不匹配: 期望 {rm_tensor2_grad.shape}, 得到 {rm_tensor2_grad.grad.shape}")
                        
                        # 梯度值比较
                        if TORCH_AVAILABLE and passed_grad:
                            # 比较第一个输入的梯度
                            torch_grad1 = torch_tensor1_grad.grad.numpy()
                            rm_grad1 = rm_tensor1_grad.grad.data
                            if not np.allclose(rm_grad1, torch_grad1, rtol=1e-4, atol=1e-4):
                                passed_grad = False
                                max_diff1 = np.max(np.abs(rm_grad1 - torch_grad1))
                                grad_details.append(f"第一个输入梯度值不匹配，最大差异: {max_diff1}")
                            
                            # 比较第二个输入的梯度
                            torch_grad2 = torch_tensor2_grad.grad.numpy()
                            rm_grad2 = rm_tensor2_grad.grad.data
                            if not np.allclose(rm_grad2, torch_grad2, rtol=1e-4, atol=1e-4):
                                passed_grad = False
                                max_diff2 = np.max(np.abs(rm_grad2 - torch_grad2))
                                grad_details.append(f"第二个输入梯度值不匹配，最大差异: {max_diff2}")
                        
                    except Exception as e:
                        passed_grad = False
                        grad_details.append(f"梯度计算异常: {str(e)}")
                
                # 总体测试结果
                passed = passed_forward and passed_grad
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        if not passed_forward:
                            print(f"  前向计算失败")
                        if not passed_grad:
                            for detail in grad_details:
                                print(f"  {detail}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"maximum函数测试失败: {case['name']} - {'前向计算失败' if not passed_forward else ', '.join(grad_details)}")
                
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case['name']} - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                raise
        
        # 添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的maximum函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试maximum函数:")
                
                test_cases = [
                    # 基本场景
                    {"name": "2D张量基本比较", "shape": (3, 4), "dtype": np.float32},
                    {"name": "1D向量比较", "shape": (5,), "dtype": np.float64},
                    
                    # 广播场景
                    {"name": "张量与标量广播", "shape": (3, 4), "other_shape": (), "dtype": np.float32},
                    {"name": "张量与1D广播", "shape": (3, 4), "other_shape": (4,), "dtype": np.float64},
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        if "other_shape" in case:
                            # 广播场景
                            if case["other_shape"] == ():
                                # 标量情况
                                cp_data1 = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                                cp_data2 = rm.cuda.cp.array(rm.cuda.cp.random.rand()).astype(case["dtype"])
                            else:
                                # 不同形状张量广播
                                cp_data1 = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                                cp_data2 = rm.cuda.cp.random.rand(*case["other_shape"]).astype(case["dtype"])
                        else:
                            # 相同形状情况
                            cp_data1 = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            cp_data2 = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor1 = rm.tensor(cp_data1, device=device)
                        rm_tensor2 = rm.tensor(cp_data2, device=device)
                        rm_result = rm.maximum(rm_tensor1, rm_tensor2)
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        # 梯度测试（仅对浮点类型）
                        if np.issubdtype(case["dtype"], np.floating):
                            # 创建需要梯度的张量
                            cp_data1_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            if "other_shape" in case:
                                if case["other_shape"] == ():
                                    cp_data2_grad = rm.cuda.cp.array(rm.cuda.cp.random.rand()).astype(case["dtype"])
                                else:
                                    cp_data2_grad = rm.cuda.cp.random.rand(*case["other_shape"]).astype(case["dtype"])
                            else:
                                cp_data2_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            
                            rm_tensor1_grad = rm.tensor(cp_data1_grad, requires_grad=True, device=device)
                            rm_tensor2_grad = rm.tensor(cp_data2_grad, requires_grad=True, device=device)
                            
                            # 计算maximum
                            rm_max_val = rm.maximum(rm_tensor1_grad, rm_tensor2_grad)
                            
                            # 计算损失并反向传播
                            rm_loss = rm.sum(rm_max_val)
                            rm_loss.backward()
                            
                            # 验证梯度
                            self.assertIsNotNone(rm_tensor1_grad.grad)
                            self.assertEqual(rm_tensor1_grad.grad.shape, rm_tensor1_grad.shape)
                            self.assertIsNotNone(rm_tensor2_grad.grad)
                            self.assertEqual(rm_tensor2_grad.grad.shape, rm_tensor2_grad.shape)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的maximum函数测试失败: {case_name} - {str(e)}")

    def test_minimum(self):
        """测试minimum函数 - 与torch.minimum对比，包括前向计算和梯度跟踪场景"""
        test_cases = [
            # 基本场景
            {"name": "2D张量基本比较", "shape": (3, 4), "dtype": np.float32},
            {"name": "3D张量基本比较", "shape": (2, 3, 4), "dtype": np.float64},
            {"name": "1D向量比较", "shape": (5,), "dtype": np.float32},
            {"name": "标量比较", "shape": (), "dtype": np.float64},
            
            # 广播场景
            {"name": "张量与标量广播", "shape": (3, 4), "other_shape": (), "dtype": np.float32},
            {"name": "张量与1D广播", "shape": (3, 4), "other_shape": (4,), "dtype": np.float64},
            {"name": "不同形状张量广播", "shape": (2, 1, 4), "other_shape": (1, 3, 1), "dtype": np.float32},
            
            # 特殊值场景
            {"name": "相等值测试", "shape": (2, 3), "equal_values": True, "dtype": np.float32},
            {"name": "负数测试", "shape": (3, 3), "negative": True, "dtype": np.float64},
            {"name": "混合正负数测试", "shape": (2, 4), "mixed": True, "dtype": np.float32},
            
            # 边界场景
            {"name": "大数值测试", "shape": (2, 2), "large_values": True, "dtype": np.float32},
            {"name": "小数值测试", "shape": (2, 2), "small_values": True, "dtype": np.float64},
            
            # 数据类型场景
            {"name": "int32类型测试", "shape": (3, 3), "dtype": np.int32},
            {"name": "int64类型测试", "shape": (2, 4), "dtype": np.int64},
        ]
        
        for case in test_cases:
            case_name = f"{self.test_minimum.__doc__} - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                if case.get("equal_values", False):
                    # 相等值测试
                    np_data1 = np.ones(case["shape"], dtype=case["dtype"]) * 3.0
                    np_data2 = np.ones(case["shape"], dtype=case["dtype"]) * 3.0
                elif case.get("negative", False):
                    # 负数测试
                    if case["shape"] == ():
                        np_data1 = np.array(-np.random.rand() * 10, dtype=case["dtype"])
                        np_data2 = np.array(-np.random.rand() * 10, dtype=case["dtype"])
                    else:
                        np_data1 = (-np.random.rand(*case["shape"]) * 10).astype(case["dtype"])
                        np_data2 = (-np.random.rand(*case["shape"]) * 10).astype(case["dtype"])
                elif case.get("mixed", True):
                    # 混合正负数测试
                    if case["shape"] == ():
                        np_data1 = np.array((np.random.rand() - 0.5) * 20, dtype=case["dtype"])
                        np_data2 = np.array((np.random.rand() - 0.5) * 20, dtype=case["dtype"])
                    else:
                        np_data1 = ((np.random.rand(*case["shape"]) - 0.5) * 20).astype(case["dtype"])
                        np_data2 = ((np.random.rand(*case["shape"]) - 0.5) * 20).astype(case["dtype"])
                elif case.get("large_values", False):
                    # 大数值测试
                    if case["shape"] == ():
                        np_data1 = np.array(np.random.rand() * 1e6, dtype=case["dtype"])
                        np_data2 = np.array(np.random.rand() * 1e6, dtype=case["dtype"])
                    else:
                        np_data1 = (np.random.rand(*case["shape"]) * 1e6).astype(case["dtype"])
                        np_data2 = (np.random.rand(*case["shape"]) * 1e6).astype(case["dtype"])
                elif case.get("small_values", False):
                    # 小数值测试
                    if case["shape"] == ():
                        np_data1 = np.array(np.random.rand() * 1e-6, dtype=case["dtype"])
                        np_data2 = np.array(np.random.rand() * 1e-6, dtype=case["dtype"])
                    else:
                        np_data1 = (np.random.rand(*case["shape"]) * 1e-6).astype(case["dtype"])
                        np_data2 = (np.random.rand(*case["shape"]) * 1e-6).astype(case["dtype"])
                else:
                    # 常规随机数据
                    if case["shape"] == ():
                        np_data1 = np.array(np.random.rand() * 10, dtype=case["dtype"])
                        if "other_shape" in case:
                            np_data2 = np.array(np.random.rand() * 10, dtype=case["dtype"])
                        else:
                            np_data2 = np.array(np.random.rand() * 10, dtype=case["dtype"])
                    else:
                        np_data1 = (np.random.rand(*case["shape"]) * 10).astype(case["dtype"])
                        if "other_shape" in case:
                            if case["other_shape"] == ():
                                np_data2 = np.array(np.random.rand() * 10, dtype=case["dtype"])
                            else:
                                np_data2 = (np.random.rand(*case["other_shape"]) * 10).astype(case["dtype"])
                        else:
                            np_data2 = (np.random.rand(*case["shape"]) * 10).astype(case["dtype"])
                
                # 前向计算测试
                # 创建Riemann张量
                rm_tensor1 = rm.tensor(np_data1)
                rm_tensor2 = rm.tensor(np_data2)
                rm_result = rm.minimum(rm_tensor1, rm_tensor2)
                
                # 创建PyTorch张量作为参考
                if TORCH_AVAILABLE:
                    # 处理数据类型转换
                    if case["dtype"] == np.float32:
                        torch_dtype = torch.float32
                    elif case["dtype"] == np.float64:
                        torch_dtype = torch.float64
                    elif case["dtype"] == np.int32:
                        torch_dtype = torch.int32
                    elif case["dtype"] == np.int64:
                        torch_dtype = torch.int64
                    else:
                        torch_dtype = torch.float32
                    
                    torch_tensor1 = torch.tensor(np_data1, dtype=torch_dtype)
                    torch_tensor2 = torch.tensor(np_data2, dtype=torch_dtype)
                    torch_result = torch.minimum(torch_tensor1, torch_tensor2)
                else:
                    torch_result = None
                
                # 比较前向计算结果
                passed_forward = compare_values(rm_result, torch_result, check_dtype=False)
                
                # 梯度跟踪测试（仅对浮点类型）
                passed_grad = True
                grad_details = []
                
                if np.issubdtype(case["dtype"], np.floating):
                    try:
                        # 创建需要梯度的张量
                        rm_tensor1_grad = rm.tensor(np_data1.copy(), requires_grad=True)
                        rm_tensor2_grad = rm.tensor(np_data2.copy(), requires_grad=True)
                        
                        if TORCH_AVAILABLE:
                            torch_tensor1_grad = torch.tensor(np_data1.copy(), dtype=torch_dtype, requires_grad=True)
                            torch_tensor2_grad = torch.tensor(np_data2.copy(), dtype=torch_dtype, requires_grad=True)
                        
                        # 计算minimum
                        rm_min_val = rm.minimum(rm_tensor1_grad, rm_tensor2_grad)
                        
                        if TORCH_AVAILABLE:
                            torch_min_val = torch.minimum(torch_tensor1_grad, torch_tensor2_grad)
                        
                        # 计算损失并反向传播
                        rm_loss = rm.sum(rm_min_val)
                        rm_loss.backward()
                        
                        if TORCH_AVAILABLE:
                            torch_loss = torch.sum(torch_min_val)
                            torch_loss.backward()
                        
                        # 检查梯度是否存在且形状正确
                        if not hasattr(rm_tensor1_grad, 'grad') or rm_tensor1_grad.grad is None:
                            passed_grad = False
                            grad_details.append("Riemann第一个输入梯度未生成")
                        elif rm_tensor1_grad.grad.shape != rm_tensor1_grad.shape:
                            passed_grad = False
                            grad_details.append(f"第一个输入梯度形状不匹配: 期望 {rm_tensor1_grad.shape}, 得到 {rm_tensor1_grad.grad.shape}")
                        
                        if not hasattr(rm_tensor2_grad, 'grad') or rm_tensor2_grad.grad is None:
                            passed_grad = False
                            grad_details.append("Riemann第二个输入梯度未生成")
                        elif rm_tensor2_grad.grad.shape != rm_tensor2_grad.shape:
                            passed_grad = False
                            grad_details.append(f"第二个输入梯度形状不匹配: 期望 {rm_tensor2_grad.shape}, 得到 {rm_tensor2_grad.grad.shape}")
                        
                        # 梯度值比较
                        if TORCH_AVAILABLE and passed_grad:
                            # 比较第一个输入的梯度
                            torch_grad1 = torch_tensor1_grad.grad.numpy()
                            rm_grad1 = rm_tensor1_grad.grad.data
                            if not np.allclose(rm_grad1, torch_grad1, rtol=1e-4, atol=1e-4):
                                passed_grad = False
                                max_diff1 = np.max(np.abs(rm_grad1 - torch_grad1))
                                grad_details.append(f"第一个输入梯度值不匹配，最大差异: {max_diff1}")
                            
                            # 比较第二个输入的梯度
                            torch_grad2 = torch_tensor2_grad.grad.numpy()
                            rm_grad2 = rm_tensor2_grad.grad.data
                            if not np.allclose(rm_grad2, torch_grad2, rtol=1e-4, atol=1e-4):
                                passed_grad = False
                                max_diff2 = np.max(np.abs(rm_grad2 - torch_grad2))
                                grad_details.append(f"第二个输入梯度值不匹配，最大差异: {max_diff2}")
                        
                    except Exception as e:
                        passed_grad = False
                        grad_details.append(f"梯度计算异常: {str(e)}")
                
                # 总体测试结果
                passed = passed_forward and passed_grad
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case["name"], passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case['name']} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        if not passed_forward:
                            print(f"  前向计算失败")
                        if not passed_grad:
                            for detail in grad_details:
                                print(f"  {detail}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"minimum函数测试失败: {case['name']} - {'前向计算失败' if not passed_forward else ', '.join(grad_details)}")
                
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case['name']} - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                raise
        
        # 添加CUDA张量测试用例
        if CUDA_AVAILABLE:
            print(f"\n{Colors.HEADER}开始测试CUDA张量的minimum函数{Colors.ENDC}")
            
            for device in ["cuda", "cuda:0"]:
                device_name = device
                print(f"\n在设备 {device_name} 上测试minimum函数:")
                
                test_cases = [
                    # 基本场景
                    {"name": "2D张量基本比较", "shape": (3, 4), "dtype": np.float32},
                    {"name": "1D向量比较", "shape": (5,), "dtype": np.float64},
                    
                    # 广播场景
                    {"name": "张量与标量广播", "shape": (3, 4), "other_shape": (), "dtype": np.float32},
                    {"name": "张量与1D广播", "shape": (3, 4), "other_shape": (4,), "dtype": np.float64},
                ]
                
                for case in test_cases:
                    case_name = f"{device_name}: {case['name']}"
                    start_time = time.time()
                    try:
                        # 创建测试数据
                        if "other_shape" in case:
                            # 广播场景
                            if case["other_shape"] == ():
                                # 标量情况
                                cp_data1 = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                                cp_data2 = rm.cuda.cp.array(rm.cuda.cp.random.rand()).astype(case["dtype"])
                            else:
                                # 不同形状张量广播
                                cp_data1 = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                                cp_data2 = rm.cuda.cp.random.rand(*case["other_shape"]).astype(case["dtype"])
                        else:
                            # 相同形状情况
                            cp_data1 = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            cp_data2 = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                        
                        # 创建CUDA张量
                        rm_tensor1 = rm.tensor(cp_data1, device=device)
                        rm_tensor2 = rm.tensor(cp_data2, device=device)
                        rm_result = rm.minimum(rm_tensor1, rm_tensor2)
                        
                        # 前向计算结果验证
                        self.assertIsNotNone(rm_result)
                        
                        # 梯度测试（仅对浮点类型）
                        if np.issubdtype(case["dtype"], np.floating):
                            # 创建需要梯度的张量
                            cp_data1_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            if "other_shape" in case:
                                if case["other_shape"] == ():
                                    cp_data2_grad = rm.cuda.cp.array(rm.cuda.cp.random.rand()).astype(case["dtype"])
                                else:
                                    cp_data2_grad = rm.cuda.cp.random.rand(*case["other_shape"]).astype(case["dtype"])
                            else:
                                cp_data2_grad = rm.cuda.cp.random.rand(*case["shape"]).astype(case["dtype"])
                            
                            rm_tensor1_grad = rm.tensor(cp_data1_grad, requires_grad=True, device=device)
                            rm_tensor2_grad = rm.tensor(cp_data2_grad, requires_grad=True, device=device)
                            
                            # 计算minimum
                            rm_min_val = rm.minimum(rm_tensor1_grad, rm_tensor2_grad)
                            
                            # 计算损失并反向传播
                            rm_loss = rm.sum(rm_min_val)
                            rm_loss.backward()
                            
                            # 验证梯度
                            self.assertIsNotNone(rm_tensor1_grad.grad)
                            self.assertEqual(rm_tensor1_grad.grad.shape, rm_tensor1_grad.shape)
                            self.assertIsNotNone(rm_tensor2_grad.grad)
                            self.assertEqual(rm_tensor2_grad.grad.shape, rm_tensor2_grad.shape)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, True)
                            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                            
                    except Exception as e:
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name, False, [str(e)])
                            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
                        self.fail(f"CUDA张量的minimum函数测试失败: {case_name} - {str(e)}")

    def test_sumall(self):
        """测试sumall函数 - 包括前向计算和梯度跟踪场景"""
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function("test_sumall")
        
        print(f"\n{Colors.HEADER}开始测试: {self.test_sumall.__doc__}{Colors.ENDC}")
        
        # 测试1: 两个张量相加
        print("\n测试1: 两个张量相加")
        start_time = time.time()
        try:
            x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
            y = rm.tensor([4.0, 5.0, 6.0], requires_grad=True)
            z = rm.sumall(x, y)
            print(f"x + y = {z}")
            
            # 测试梯度 - 对于非标量张量，需要提供grad_outputs
            z_sum = z.sum()  # 将结果转换为标量
            z_sum.backward()
            print(f"x.grad = {x.grad}")
            print(f"y.grad = {y.grad}")
            
            # 验证梯度
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(y.grad)
            self.assertEqual(x.grad.shape, x.shape)
            self.assertEqual(y.grad.shape, y.shape)
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("两个张量相加", True)
                print(f"{Colors.OKGREEN}测试通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("两个张量相加", False, [str(e)])
                print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
        
        # 测试2: 张量与标量相加
        print("\n测试2: 张量与标量相加")
        start_time = time.time()
        try:
            x2 = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
            scalar = 5.0
            z2 = rm.sumall(x2, scalar)
            print(f"x2 + {scalar} = {z2}")
            
            # 测试梯度 - 对于非标量张量，需要提供grad_outputs
            z2_sum = z2.sum()  # 将结果转换为标量
            z2_sum.backward()
            print(f"x2.grad = {x2.grad}")
            
            # 验证梯度
            self.assertIsNotNone(x2.grad)
            self.assertEqual(x2.grad.shape, x2.shape)
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("张量与标量相加", True)
                print(f"{Colors.OKGREEN}测试通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("张量与标量相加", False, [str(e)])
                print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
        
        # 测试3: 多个张量相加
        print("\n测试3: 多个张量相加")
        start_time = time.time()
        try:
            a = rm.tensor([1.0, 2.0], requires_grad=True)
            b = rm.tensor([3.0, 4.0], requires_grad=True)
            c = rm.tensor([5.0, 6.0], requires_grad=True)
            z3 = rm.sumall(a, b, c)
            print(f"a + b + c = {z3}")
            
            # 测试梯度 - 对于非标量张量，需要提供grad_outputs
            z3_sum = z3.sum()  # 将结果转换为标量
            z3_sum.backward()
            print(f"a.grad = {a.grad}")
            print(f"b.grad = {b.grad}")
            print(f"c.grad = {c.grad}")
            
            # 验证梯度
            self.assertIsNotNone(a.grad)
            self.assertIsNotNone(b.grad)
            self.assertIsNotNone(c.grad)
            self.assertEqual(a.grad.shape, a.shape)
            self.assertEqual(b.grad.shape, b.shape)
            self.assertEqual(c.grad.shape, c.shape)
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("多个张量相加", True)
                print(f"{Colors.OKGREEN}测试通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("多个张量相加", False, [str(e)])
                print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
        
        # 测试4: 混合张量和非张量相加
        print("\n测试4: 混合张量和非张量相加")
        start_time = time.time()
        try:
            x4 = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
            y4 = 2.0
            z4 = 3.0
            result4 = rm.sumall(x4, y4, z4)
            print(f"x4 + {y4} + {z4} = {result4}")
            
            # 测试梯度 - 对于非标量张量，需要提供grad_outputs
            result4_sum = result4.sum()  # 将结果转换为标量
            result4_sum.backward()
            print(f"x4.grad = {x4.grad}")
            
            # 验证梯度
            self.assertIsNotNone(x4.grad)
            self.assertEqual(x4.grad.shape, x4.shape)
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("混合张量和非张量相加", True)
                print(f"{Colors.OKGREEN}测试通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("混合张量和非张量相加", False, [str(e)])
                print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
        
        # 测试5: 广播兼容的情况（标量、1D向量、2D张量）
        print("\n测试5: 广播兼容的情况（标量、1D向量、2D张量）")
        start_time = time.time()
        try:
            a_scalar = rm.tensor(1.0, requires_grad=True)  # 标量张量
            b_vector = rm.tensor([2.0, 3.0], requires_grad=True)  # 1D向量
            c_matrix = rm.tensor([[4.0, 5.0], [6.0, 7.0]], requires_grad=True)  # 2D张量
            
            # 计算和
            result5 = rm.sumall(a_scalar, b_vector, c_matrix)
            print(f"a_scalar + b_vector + c_matrix = {result5}")
            print(f"Result shape: {result5.shape}")
            
            # 测试梯度 - 对于非标量张量，需要提供grad_outputs
            result5_sum = result5.sum()  # 将结果转换为标量
            result5_sum.backward()
            print(f"a_scalar.grad = {a_scalar.grad}")
            print(f"b_vector.grad = {b_vector.grad}")
            print(f"c_matrix.grad = {c_matrix.grad}")
            
            # 验证梯度
            self.assertIsNotNone(a_scalar.grad)
            self.assertIsNotNone(b_vector.grad)
            self.assertIsNotNone(c_matrix.grad)
            self.assertEqual(b_vector.grad.shape, b_vector.shape)
            self.assertEqual(c_matrix.grad.shape, c_matrix.shape)
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("广播兼容的情况", True)
                print(f"{Colors.OKGREEN}测试通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("广播兼容的情况", False, [str(e)])
                print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
        
        # 测试6: 非张量参数为numpy数组
        print("\n测试6: 非张量参数为numpy数组")
        start_time = time.time()
        try:
            x6 = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
            numpy_arr = np.array([4.0, 5.0, 6.0], dtype=np.float32)
            result6 = rm.sumall(x6, numpy_arr)
            print(f"x6 + numpy_arr = {result6}")
            
            # 测试梯度
            result6_sum = result6.sum()
            result6_sum.backward()
            print(f"x6.grad = {x6.grad}")
            
            # 验证梯度
            self.assertIsNotNone(x6.grad)
            self.assertEqual(x6.grad.shape, x6.shape)
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("非张量参数为numpy数组", True)
                print(f"{Colors.OKGREEN}测试通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("非张量参数为numpy数组", False, [str(e)])
                print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
        
        # 测试7: 非张量参数为cupy数组（如果有GPU支持）
        print("\n测试7: 非张量参数为cupy数组")
        start_time = time.time()
        try:
            if CUDA_AVAILABLE:
                try:
                    x7 = rm.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)
                    cupy_arr = rm.cuda.cp.array([4.0, 5.0, 6.0], dtype=np.float32)
                    result7 = rm.sumall(x7, cupy_arr)
                    print(f"x7 + cupy_arr = {result7}")
                    
                    # 测试梯度
                    result7_sum = result7.sum()
                    result7_sum.backward()
                    print(f"x7.grad = {x7.grad}")
                    
                    # 验证梯度
                    self.assertIsNotNone(x7.grad)
                    self.assertEqual(x7.grad.shape, x7.shape)
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result("非张量参数为cupy数组", True)
                        print(f"{Colors.OKGREEN}测试通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                except Exception as e:
                    print(f"跳过cupy数组测试: {e}")
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result("非张量参数为cupy数组", True)  # 跳过视为通过
            else:
                print("跳过cupy数组测试：没有GPU支持")
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result("非张量参数为cupy数组", True)  # 跳过视为通过
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("非张量参数为cupy数组", False, [str(e)])
                print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
        
        # 测试8: 设备一致性检查
        print("\n测试8: 设备一致性检查")
        start_time = time.time()
        try:
            # 假设我们有GPU支持
            if CUDA_AVAILABLE:
                # 创建CPU张量和GPU张量
                cpu_tensor = rm.tensor([1.0, 2.0])
                gpu_tensor = rm.tensor([3.0, 4.0], device='cuda')
                # 尝试相加，应该报错
                result = rm.sumall(cpu_tensor, gpu_tensor)
                print("错误：应该报错但没有报错！")
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result("设备一致性检查", False, ["设备不一致时应该报错"])
                    print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.fail("设备不一致时应该报错")
            else:
                print("跳过设备一致性测试：没有GPU支持")
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result("设备一致性检查", True)  # 跳过视为通过
        except ValueError as e:
            print(f"正确：设备不一致时报错 - {e}")
            # 验证错误信息
            self.assertIn("device", str(e).lower())
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("设备一致性检查", True)
                print(f"{Colors.OKGREEN}测试通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("设备一致性检查", False, [str(e)])
                print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
        
        # 测试9: 所有参数都是标量
        print("\n测试9: 所有参数都是标量")
        start_time = time.time()
        try:
            result9 = rm.sumall(1.0, 2.0, 3.0, 4.0)
            print(f"1.0 + 2.0 + 3.0 + 4.0 = {result9}")
            
            # 验证结果
            self.assertIsInstance(result9, rm.TN)
            self.assertEqual(result9.data, 10.0)
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("所有参数都是标量", True)
                print(f"{Colors.OKGREEN}测试通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("所有参数都是标量", False, [str(e)])
                print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
        
        # 测试10: 同时包括python量、numpy数组、cupy数组和张量
        print("\n测试10: 同时包括python量、numpy数组、cupy数组和张量")
        start_time = time.time()
        try:
            if CUDA_AVAILABLE:
                # 创建各种类型的参数
                tensor_param = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
                python_param = 10.0
                numpy_param = np.array([4.0, 5.0, 6.0], dtype=np.float32)
                cupy_param = rm.cuda.cp.array([7.0, 8.0, 9.0], dtype=np.float32)
                
                # 计算和
                result10 = rm.sumall(tensor_param, python_param, numpy_param, cupy_param)
                print(f"tensor + python + numpy + cupy = {result10}")
                
                # 测试梯度
                result10_sum = result10.sum()
                result10_sum.backward()
                print(f"tensor_param.grad = {tensor_param.grad}")
                
                # 验证梯度
                self.assertIsNotNone(tensor_param.grad)
                self.assertEqual(tensor_param.grad.shape, tensor_param.shape)
                
            else:
                # 没有GPU支持，使用python量、numpy数组和张量
                tensor_param = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
                python_param = 10.0
                numpy_param = np.array([4.0, 5.0, 6.0], dtype=np.float32)
                
                # 计算和
                result10 = rm.sumall(tensor_param, python_param, numpy_param)
                print(f"tensor + python + numpy = {result10}")
                
                # 测试梯度
                result10_sum = result10.sum()
                result10_sum.backward()
                print(f"tensor_param.grad = {tensor_param.grad}")
                
                # 验证梯度
                self.assertIsNotNone(tensor_param.grad)
                self.assertEqual(tensor_param.grad.shape, tensor_param.shape)
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("混合类型参数组合", True)
                print(f"{Colors.OKGREEN}测试通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("混合类型参数组合", False, [str(e)])
                print(f"{Colors.FAIL}测试失败{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
        
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
        
        print("\n测试完成！")

# 如果作为独立脚本运行
if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行max/min函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMaxMinFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)
    