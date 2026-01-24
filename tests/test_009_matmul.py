import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann import tensor
    # 从rm.cuda获取CUDA支持
    CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
    cp = rm.cuda.cp
except ImportError as e:
    print(f"无法导入riemann模块: {e}")
    print("请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的矩阵乘法")
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
        
    def _start_function(self, function_name):
        self.current_function = function_name
        self.current_function_start_time = time.time()
        self.current_test_details = []  # 重置详细信息列表
        
        if function_name not in self.function_stats:
            self.function_stats[function_name] = {"total": 0, "passed": 0, "time": 0.0, "sub_cases_total": 0, "sub_cases_passed": 0}
    
    def add_result(self, case_name, passed, details=None):
        self.total_cases += 1
        if passed:
            self.passed_cases += 1
        
        if self.current_function:
            self.function_stats[self.current_function]["total"] += 1
            if passed:
                self.function_stats[self.current_function]["passed"] += 1
    
    def add_sub_case_result(self, passed, case_name=None, details=None):
        """记录子用例的执行结果"""
        if self.current_function:
            self.function_stats[self.current_function]["sub_cases_total"] += 1
            if passed:
                self.function_stats[self.current_function]["sub_cases_passed"] += 1
                
            # 记录测试详情
            if case_name is not None:
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
        # 避免除零错误
        pass_rate = (self.passed_cases / self.total_cases * 100) if self.total_cases > 0 else 0
        print(f"测试通过率: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{pass_rate:.2f}%{Colors.ENDC}")
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
            # 计算通过率，优先使用子用例统计，并确保避免除零错误
            if stats['sub_cases_total'] > 0:
                pass_rate = (stats["sub_cases_passed"]/stats["sub_cases_total"])*100
            else:
                pass_rate = (stats["passed"]/stats["total"])*100 if stats["total"] > 0 else 0
            status_color = Colors.OKGREEN if pass_rate == 100 else Colors.FAIL
            
            # 计算每个字段的显示宽度并添加适当的填充
            func_name_display = func_name
            func_name_width = self._get_display_width(func_name_display)
            func_name_padding = col_widths[0] - func_name_width
            
            # 使用子用例统计，如果有的话
            if stats['sub_cases_total'] > 0:
                pass_total_display = f"{stats['sub_cases_passed']}/{stats['sub_cases_total']}"
            else:
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
    
    # 处理CUDA张量，先移动到CPU再转换为numpy数组
    try:
        # 处理Riemann结果
        if hasattr(rm_result, 'device'):
            # 检查是否为CUDA设备
            is_cuda = False
            try:
                device = rm_result.device
                if isinstance(device, str) and 'cuda' in device.lower():
                    is_cuda = True
                elif hasattr(device, 'type') and device.type == 'cuda':
                    is_cuda = True
            except Exception:
                pass
            
            if is_cuda:
                # 如果是CUDA张量，先移动到CPU
                rm_data = rm_result.cpu().detach().numpy()
            else:
                rm_data = rm_result.detach().numpy()
        else:
            rm_data = rm_result.detach().numpy()
        
        # 处理PyTorch结果
        if hasattr(torch_result, 'is_cuda') and torch_result.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            torch_data = torch_result.detach().cpu().numpy()
        else:
            torch_data = torch_result.detach().numpy()
    except Exception as e:
        print(f"比较值转换错误: {e}")
        return False
    
    # 处理形状不匹配的情况
    # 增加容差参数
    try:
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

class TestMatmulFunctions(unittest.TestCase):
    def setUp(self):
        # 设置随机种子以确保结果可重复
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
        # 获取当前测试方法名
        self.current_test_name = self._testMethodName
        # 如果是独立脚本运行，则开始记录函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats._start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")
                
    def tearDown(self):
        # 如果是独立脚本运行，则结束函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_vector_multiplication(self):
        """测试场景1: 向量乘法（替代标量乘法）"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"向量乘法 - {device}"
            start_time = time.time()
            try:
                # 创建测试数据 - 一维向量（避免标量）
                np_x = np.array([2.0], dtype=np.float64)
                np_y = np.array([3.0], dtype=np.float64)
                
                # 创建Riemann张量
                if device == "cpu":
                    rm_x = tensor(np_x, requires_grad=True)
                    rm_y = tensor(np_y, requires_grad=True)
                else:  # cuda
                    rm_x = tensor(np_x, requires_grad=True, device=device)
                    rm_y = tensor(np_y, requires_grad=True, device=device)
                
                # 创建PyTorch张量
                if TORCH_AVAILABLE:
                    if device == "cpu":
                        torch_x = torch.tensor(np_x, requires_grad=True)
                        torch_y = torch.tensor(np_y, requires_grad=True)
                    else:  # cuda
                        torch_x = torch.tensor(np_x, requires_grad=True, device=device)
                        torch_y = torch.tensor(np_y, requires_grad=True, device=device)
                else:
                    torch_x, torch_y = None, None
                
                # 执行矩阵乘法
                rm_z = rm_x @ rm_y
                rm_z_sum = rm_z.sum()
                
                if TORCH_AVAILABLE:
                    torch_z = torch_x @ torch_y
                    torch_z_sum = torch_z.sum()
                else:
                    torch_z, torch_z_sum = None, None
                
                # 反向传播
                rm_z_sum.backward()
                if TORCH_AVAILABLE:
                    torch_z_sum.backward()
                
                # 比较乘法结果
                passed_z = compare_values(rm_z, torch_z)
                
                # 比较梯度
                passed_dx = compare_values(rm_x.grad, torch_x.grad if TORCH_AVAILABLE else None)
                passed_dy = compare_values(rm_y.grad, torch_y.grad if TORCH_AVAILABLE else None)
                
                # 综合结果
                passed = passed_z and passed_dx and passed_dy
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    # 记录子用例结果
                    stats.add_sub_case_result(passed)
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        if not passed_z:
                            print(f"  乘积值比较: 失败")
                        if not passed_dx:
                            print(f"  x梯度比较: 失败")
                        if not passed_dy:
                            print(f"  y梯度比较: 失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"矩阵乘法测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_vector_matrix_multiplication(self):
        """测试场景2: 向量与矩阵乘法"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        all_passed = True
        start_time = time.time()
        
        try:
            # 测试用例列表
            test_cases = [
                # (测试名称, x形状, y形状)
                ("一维向量与二维矩阵乘法", (3,), (3, 4)),
                ("二维行向量与二维矩阵乘法", (1, 3), (3, 4)),
                ("二维列向量与二维矩阵乘法", (3, 1), (1, 4)),
                ("二维矩阵与一维向量乘法", (3, 4), (4,)),
            ]
            
            for device in devices:
                device_case_name = f"向量与矩阵乘法 - {device}"
                device_all_passed = True
                
                for sub_case_name, x_shape, y_shape in test_cases:
                    full_sub_case_name = f"{sub_case_name} - {device}"
                    
                    # 创建测试数据
                    np_x = np.random.randn(*x_shape).astype(np.float64)
                    np_y = np.random.randn(*y_shape).astype(np.float64)
                    
                    # 创建Riemann张量
                    if device == "cpu":
                        rm_x = tensor(np_x, requires_grad=True)
                        rm_y = tensor(np_y, requires_grad=True)
                    else:  # cuda
                        rm_x = tensor(np_x, requires_grad=True, device=device)
                        rm_y = tensor(np_y, requires_grad=True, device=device)
                    
                    # 创建PyTorch张量
                    if TORCH_AVAILABLE:
                        if device == "cpu":
                            torch_x = torch.tensor(np_x, requires_grad=True)
                            torch_y = torch.tensor(np_y, requires_grad=True)
                        else:  # cuda
                            torch_x = torch.tensor(np_x, requires_grad=True, device=device)
                            torch_y = torch.tensor(np_y, requires_grad=True, device=device)
                    else:
                        torch_x, torch_y = None, None
                    
                    # 执行矩阵乘法
                    rm_z = rm_x @ rm_y
                    rm_z_sum = rm_z.sum()
                    
                    if TORCH_AVAILABLE:
                        torch_z = torch_x @ torch_y
                        torch_z_sum = torch_z.sum()
                    else:
                        torch_z, torch_z_sum = None, None
                    
                    # 反向传播
                    rm_z_sum.backward()
                    if TORCH_AVAILABLE:
                        torch_z_sum.backward()
                    
                    # 比较乘法结果
                    passed_z = compare_values(rm_z, torch_z)
                    
                    # 比较梯度
                    passed_dx = compare_values(rm_x.grad, torch_x.grad if TORCH_AVAILABLE else None)
                    passed_dy = compare_values(rm_y.grad, torch_y.grad if TORCH_AVAILABLE else None)
                    
                    # 综合结果
                    case_passed = passed_z and passed_dx and passed_dy
                    if not case_passed:
                        device_all_passed = False
                        all_passed = False
                        
                    # 记录子用例结果
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_sub_case_result(case_passed)
                        status = "通过" if case_passed else "失败"
                        print(f"  子用例: {full_sub_case_name} - {Colors.OKGREEN if case_passed else Colors.FAIL}{status}{Colors.ENDC}")
                        if not case_passed:
                            if not passed_z:
                                print(f"    乘积值比较: 失败")
                            if not passed_dx:
                                print(f"    x梯度比较: 失败")
                            if not passed_dy:
                                print(f"    y梯度比较: 失败")
                    
                    # 重置梯度
                    if hasattr(rm_x, 'grad'):
                        rm_x.grad = None
                    if hasattr(rm_y, 'grad'):
                        rm_y.grad = None
                    
                    if TORCH_AVAILABLE and hasattr(torch_x, 'grad'):
                        torch_x.grad = None
                    if TORCH_AVAILABLE and hasattr(torch_y, 'grad'):
                        torch_y.grad = None
                
                # 记录设备测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(device_case_name, device_all_passed)
                    status = "通过" if device_all_passed else "失败"
                    print(f"测试用例: {device_case_name} - {Colors.OKGREEN if device_all_passed else Colors.FAIL}{status}{Colors.ENDC}")
            
            time_taken = time.time() - start_time
            
            # 断言确保所有测试通过
            self.assertTrue(all_passed, f"矩阵乘法测试失败: 向量与矩阵乘法")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("向量与矩阵乘法", False, [str(e)])
                print(f"测试用例: 向量与矩阵乘法 - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_matrix_matrix_multiplication(self):
        """测试场景3: 矩阵与矩阵乘法"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        all_passed = True
        start_time = time.time()
        
        try:
            # 测试用例列表（移除可能不兼容的形状组合）
            test_cases = [
                # (测试名称, x形状, y形状)
                ("二维矩阵与二维矩阵乘法", (2, 3), (3, 4)),
                ("三维矩阵与二维矩阵乘法", (2, 3, 4), (4, 5)),
                # 移除这个可能不兼容的组合：("二维矩阵与三维矩阵乘法", (2, 3), (3, 4, 5)),
                ("三维矩阵与三维矩阵乘法", (2, 3, 4), (2, 4, 5)),
            ]
            
            for device in devices:
                device_case_name = f"矩阵与矩阵乘法 - {device}"
                device_all_passed = True
                
                for sub_case_name, x_shape, y_shape in test_cases:
                    full_sub_case_name = f"{sub_case_name} - {device}"
                    
                    try:
                        # 创建测试数据
                        np_x = np.random.randn(*x_shape).astype(np.float64)
                        np_y = np.random.randn(*y_shape).astype(np.float64)
                        
                        # 创建Riemann张量
                        if device == "cpu":
                            rm_x = tensor(np_x, requires_grad=True)
                            rm_y = tensor(np_y, requires_grad=True)
                        else:  # cuda
                            rm_x = tensor(np_x, requires_grad=True, device=device)
                            rm_y = tensor(np_y, requires_grad=True, device=device)
                        
                        # 创建PyTorch张量
                        if TORCH_AVAILABLE:
                            if device == "cpu":
                                torch_x = torch.tensor(np_x, requires_grad=True)
                                torch_y = torch.tensor(np_y, requires_grad=True)
                            else:  # cuda
                                torch_x = torch.tensor(np_x, requires_grad=True, device=device)
                                torch_y = torch.tensor(np_y, requires_grad=True, device=device)
                        else:
                            torch_x, torch_y = None, None
                        
                        # 执行矩阵乘法
                        rm_z = rm_x @ rm_y
                        rm_z_sum = rm_z.sum()
                        
                        if TORCH_AVAILABLE:
                            torch_z = torch_x @ torch_y
                            torch_z_sum = torch_z.sum()
                        else:
                            torch_z, torch_z_sum = None, None
                        
                        # 实数矩阵的反向传播
                        rm_z_sum.backward()
                        
                        if TORCH_AVAILABLE:
                            torch_z_sum.backward()
                        
                        # 比较乘法结果
                        passed_z = compare_values(rm_z, torch_z)
                        
                        # 比较梯度
                        passed_dx = compare_values(rm_x.grad, torch_x.grad if TORCH_AVAILABLE else None)
                        passed_dy = compare_values(rm_y.grad, torch_y.grad if TORCH_AVAILABLE else None)
                        
                        # 综合结果
                        case_passed = passed_z and passed_dx and passed_dy
                        if not case_passed:
                            device_all_passed = False
                            all_passed = False
                            
                        # 记录子用例结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_sub_case_result(case_passed)
                            status = "通过" if case_passed else "失败"
                            print(f"  子用例: {full_sub_case_name} - {Colors.OKGREEN if case_passed else Colors.FAIL}{status}{Colors.ENDC}")
                            if not case_passed:
                                if not passed_z:
                                    print(f"    乘积值比较: 失败")
                                if not passed_dx:
                                    print(f"    x梯度比较: 失败")
                                if not passed_dy:
                                    print(f"    y梯度比较: 失败")
                    except Exception as e:
                        device_all_passed = False
                        all_passed = False
                        # 记录子用例失败结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_sub_case_result(False)
                            print(f"  子用例: {full_sub_case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                        continue
                    
                    # 重置梯度
                    if hasattr(rm_x, 'grad'):
                        rm_x.grad = None
                    if hasattr(rm_y, 'grad'):
                        rm_y.grad = None
                    
                    if TORCH_AVAILABLE and hasattr(torch_x, 'grad'):
                        torch_x.grad = None
                    if TORCH_AVAILABLE and hasattr(torch_y, 'grad'):
                        torch_y.grad = None
                
                # 记录设备测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(device_case_name, device_all_passed)
                    status = "通过" if device_all_passed else "失败"
                    print(f"测试用例: {device_case_name} - {Colors.OKGREEN if device_all_passed else Colors.FAIL}{status}{Colors.ENDC}")
            
            time_taken = time.time() - start_time
            
            # 断言确保所有测试通过
            self.assertTrue(all_passed, f"矩阵乘法测试失败: 矩阵与矩阵乘法")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("矩阵与矩阵乘法", False, [str(e)])
                print(f"测试用例: 矩阵与矩阵乘法 - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_high_dimensional_multiplication(self):
        """测试场景4: 高维矩阵乘法"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        all_passed = True
        start_time = time.time()
        
        try:
            # 测试用例列表 - 高维张量乘法
            test_cases = [
                # (测试名称, x形状, y形状)
                ("四维矩阵乘法", (2, 3, 4, 5), (2, 3, 5, 6)),
                ("五维矩阵乘法", (1, 2, 3, 4, 5), (1, 2, 3, 5, 6)),
            ]
            
            for device in devices:
                device_case_name = f"高维矩阵乘法 - {device}"
                device_all_passed = True
                
                for sub_case_name, x_shape, y_shape in test_cases:
                    full_sub_case_name = f"{sub_case_name} - {device}"
                    
                    try:
                        # 创建测试数据
                        np_x = np.random.randn(*x_shape).astype(np.float64)
                        np_y = np.random.randn(*y_shape).astype(np.float64)
                        
                        # 创建Riemann张量
                        if device == "cpu":
                            rm_x = tensor(np_x, requires_grad=True)
                            rm_y = tensor(np_y, requires_grad=True)
                        else:  # cuda
                            rm_x = tensor(np_x, requires_grad=True, device=device)
                            rm_y = tensor(np_y, requires_grad=True, device=device)
                        
                        # 创建PyTorch张量
                        if TORCH_AVAILABLE:
                            if device == "cpu":
                                torch_x = torch.tensor(np_x, requires_grad=True)
                                torch_y = torch.tensor(np_y, requires_grad=True)
                            else:  # cuda
                                torch_x = torch.tensor(np_x, requires_grad=True, device=device)
                                torch_y = torch.tensor(np_y, requires_grad=True, device=device)
                        else:
                            torch_x, torch_y = None, None
                        
                        # 执行矩阵乘法
                        rm_z = rm_x @ rm_y
                        rm_z_sum = rm_z.sum()
                        
                        if TORCH_AVAILABLE:
                            torch_z = torch_x @ torch_y
                            torch_z_sum = torch_z.sum()
                        else:
                            torch_z, torch_z_sum = None, None
                        
                        # 实数矩阵的反向传播
                        rm_z_sum.backward()
                        
                        if TORCH_AVAILABLE:
                            torch_z_sum.backward()
                        
                        # 比较乘法结果
                        passed_z = compare_values(rm_z, torch_z)
                        
                        # 比较梯度
                        passed_dx = compare_values(rm_x.grad, torch_x.grad if TORCH_AVAILABLE else None)
                        passed_dy = compare_values(rm_y.grad, torch_y.grad if TORCH_AVAILABLE else None)
                        
                        # 综合结果
                        case_passed = passed_z and passed_dx and passed_dy
                        if not case_passed:
                            device_all_passed = False
                            all_passed = False
                            
                        # 记录子用例结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_sub_case_result(case_passed)
                            status = "通过" if case_passed else "失败"
                            print(f"  子用例: {full_sub_case_name} - {Colors.OKGREEN if case_passed else Colors.FAIL}{status}{Colors.ENDC}")
                            if not case_passed:
                                if not passed_z:
                                    print(f"    乘积值比较: 失败")
                                if not passed_dx:
                                    print(f"    x梯度比较: 失败")
                                if not passed_dy:
                                    print(f"    y梯度比较: 失败")
                    except Exception as e:
                        device_all_passed = False
                        all_passed = False
                        # 记录子用例失败结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_sub_case_result(False)
                            print(f"  子用例: {full_sub_case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                        continue
                    
                    # 重置梯度
                    if hasattr(rm_x, 'grad'):
                        rm_x.grad = None
                    if hasattr(rm_y, 'grad'):
                        rm_y.grad = None
                    
                    if TORCH_AVAILABLE and hasattr(torch_x, 'grad'):
                        torch_x.grad = None
                    if TORCH_AVAILABLE and hasattr(torch_y, 'grad'):
                        torch_y.grad = None
                
                # 记录设备测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(device_case_name, device_all_passed)
                    status = "通过" if device_all_passed else "失败"
                    print(f"测试用例: {device_case_name} - {Colors.OKGREEN if device_all_passed else Colors.FAIL}{status}{Colors.ENDC}")
            
            time_taken = time.time() - start_time
            
            # 断言确保所有测试通过
            self.assertTrue(all_passed, f"矩阵乘法测试失败: 高维矩阵乘法")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("高维矩阵乘法", False, [str(e)])
                print(f"测试用例: 高维矩阵乘法 - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_batch_matrix_multiplication(self):
        """测试场景5: 批量矩阵乘法的各种组合"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        all_passed = True
        start_time = time.time()
        
        try:
            # 测试用例列表 - 批量矩阵乘法
            test_cases = [
                # (测试名称, x形状, y形状)
                ("批量矩阵乘法 (2x3x4) @ (2x4x5)", (2, 3, 4), (2, 4, 5)),
                ("批量矩阵乘法 (3x2x4) @ (3x4x5)", (3, 2, 4), (3, 4, 5)),
                ("批量矩阵与二维矩阵乘法 (2x3x4) @ (4, 5)", (2, 3, 4), (4, 5)),
                ("二维矩阵与批量矩阵乘法 (3x4) @ (2x4x5)", (3, 4), (2, 4, 5)),
                ("批量矩阵与一维向量乘法 (2x3x4) @ (4,)", (2, 3, 4), (4,)),
                ("一维向量与批量矩阵乘法 (3,) @ (2, 3, 4)", (3,), (2, 3, 4)),
                ("批量行向量与批量列向量乘法 (2, 1, 3) @ (2, 3, 1)", (2, 1, 3), (2, 3, 1)),
                ("批量矩阵与批量矩阵乘法 (4x2x3) @ (4x3x5)", (4, 2, 3), (4, 3, 5)),
            ]
            
            for device in devices:
                device_case_name = f"批量矩阵乘法 - {device}"
                device_all_passed = True
                
                for sub_case_name, x_shape, y_shape in test_cases:
                    full_sub_case_name = f"{sub_case_name} - {device}"
                    
                    try:
                        # 创建测试数据
                        np_x = np.random.randn(*x_shape).astype(np.float64)
                        np_y = np.random.randn(*y_shape).astype(np.float64)
                        
                        # 创建Riemann张量
                        if device == "cpu":
                            rm_x = tensor(np_x, requires_grad=True)
                            rm_y = tensor(np_y, requires_grad=True)
                        else:  # cuda
                            rm_x = tensor(np_x, requires_grad=True, device=device)
                            rm_y = tensor(np_y, requires_grad=True, device=device)
                        
                        # 创建PyTorch张量
                        if TORCH_AVAILABLE:
                            if device == "cpu":
                                torch_x = torch.tensor(np_x, requires_grad=True)
                                torch_y = torch.tensor(np_y, requires_grad=True)
                            else:  # cuda
                                torch_x = torch.tensor(np_x, requires_grad=True, device=device)
                                torch_y = torch.tensor(np_y, requires_grad=True, device=device)
                        else:
                            torch_x, torch_y = None, None
                        
                        # 执行矩阵乘法
                        rm_z = rm_x @ rm_y
                        rm_z_sum = rm_z.sum()
                        
                        if TORCH_AVAILABLE:
                            torch_z = torch_x @ torch_y
                            torch_z_sum = torch_z.sum()
                        else:
                            torch_z, torch_z_sum = None, None
                        
                        # 反向传播
                        rm_z_sum.backward()
                        
                        if TORCH_AVAILABLE:
                            torch_z_sum.backward()
                        
                        # 比较乘法结果
                        passed_z = compare_values(rm_z, torch_z)
                        
                        # 比较梯度
                        passed_dx = compare_values(rm_x.grad, torch_x.grad if TORCH_AVAILABLE else None)
                        passed_dy = compare_values(rm_y.grad, torch_y.grad if TORCH_AVAILABLE else None)
                        
                        # 综合结果
                        case_passed = passed_z and passed_dx and passed_dy
                        if not case_passed:
                            device_all_passed = False
                            all_passed = False
                            
                        # 记录子用例结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_sub_case_result(case_passed)
                            status = "通过" if case_passed else "失败"
                            print(f"  子用例: {full_sub_case_name} - {Colors.OKGREEN if case_passed else Colors.FAIL}{status}{Colors.ENDC}")
                            if not case_passed:
                                if not passed_z:
                                    print(f"    乘积值比较: 失败")
                                if not passed_dx:
                                    print(f"    x梯度比较: 失败")
                                if not passed_dy:
                                    print(f"    y梯度比较: 失败")
                    except Exception as e:
                        device_all_passed = False
                        all_passed = False
                        # 记录子用例失败结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_sub_case_result(False)
                            print(f"  子用例: {full_sub_case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                        continue
                    
                    # 重置梯度
                    if hasattr(rm_x, 'grad'):
                        rm_x.grad = None
                    if hasattr(rm_y, 'grad'):
                        rm_y.grad = None
                    
                    if TORCH_AVAILABLE and hasattr(torch_x, 'grad'):
                        torch_x.grad = None
                    if TORCH_AVAILABLE and hasattr(torch_y, 'grad'):
                        torch_y.grad = None
                
                # 记录设备测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(device_case_name, device_all_passed)
                    status = "通过" if device_all_passed else "失败"
                    print(f"测试用例: {device_case_name} - {Colors.OKGREEN if device_all_passed else Colors.FAIL}{status}{Colors.ENDC}")
            
            time_taken = time.time() - start_time
            
            # 断言确保所有测试通过
            self.assertTrue(all_passed, f"矩阵乘法测试失败: 批量矩阵乘法")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("批量矩阵乘法", False, [str(e)])
                print(f"测试用例: 批量矩阵乘法 - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_mixed_dimension_multiplication(self):
        """测试场景6: 不同维度张量之间的乘法"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        all_passed = True
        start_time = time.time()
        
        try:
            # 测试用例列表 - 不同维度张量乘法
            test_cases = [
                # (测试名称, x形状, y形状)
                ("一维向量与二维矩阵乘法 (3,) @ (3, 4)", (3,), (3, 4)),
                ("二维矩阵与一维向量乘法 (3, 4) @ (4,)", (3, 4), (4,)),
                ("二维行向量与一维向量乘法 (1, 3) @ (3,)", (1, 3), (3,)),
                ("一维向量与二维列向量乘法 (3,) @ (3, 1)", (3,), (3, 1)),
                ("二维列向量与二维行向量乘法 (3, 1) @ (1, 4)", (3, 1), (1, 4)),
                ("三维张量与一维向量乘法 (2, 3, 4) @ (4,)", (2, 3, 4), (4,)),
                ("一维向量与三维张量乘法 (3,) @ (2, 3, 4)", (3,), (2, 3, 4)),
            ]
            
            for device in devices:
                device_case_name = f"不同维度张量乘法 - {device}"
                device_all_passed = True
                
                for sub_case_name, x_shape, y_shape in test_cases:
                    full_sub_case_name = f"{sub_case_name} - {device}"
                    
                    try:
                        # 创建测试数据
                        np_x = np.random.randn(*x_shape).astype(np.float64)
                        np_y = np.random.randn(*y_shape).astype(np.float64)
                        
                        # 创建Riemann张量
                        if device == "cpu":
                            rm_x = tensor(np_x, requires_grad=True)
                            rm_y = tensor(np_y, requires_grad=True)
                        else:  # cuda
                            rm_x = tensor(np_x, requires_grad=True, device=device)
                            rm_y = tensor(np_y, requires_grad=True, device=device)
                        
                        # 创建PyTorch张量
                        if TORCH_AVAILABLE:
                            if device == "cpu":
                                torch_x = torch.tensor(np_x, requires_grad=True)
                                torch_y = torch.tensor(np_y, requires_grad=True)
                            else:  # cuda
                                torch_x = torch.tensor(np_x, requires_grad=True, device=device)
                                torch_y = torch.tensor(np_y, requires_grad=True, device=device)
                        else:
                            torch_x, torch_y = None, None
                        
                        # 执行矩阵乘法
                        rm_z = rm_x @ rm_y
                        rm_z_sum = rm_z.sum()
                        
                        if TORCH_AVAILABLE:
                            torch_z = torch_x @ torch_y
                            torch_z_sum = torch_z.sum()
                        else:
                            torch_z, torch_z_sum = None, None
                        
                        # 反向传播
                        rm_z_sum.backward()
                        
                        if TORCH_AVAILABLE:
                            torch_z_sum.backward()
                        
                        # 比较乘法结果
                        passed_z = compare_values(rm_z, torch_z)
                        
                        # 比较梯度
                        passed_dx = compare_values(rm_x.grad, torch_x.grad if TORCH_AVAILABLE else None)
                        passed_dy = compare_values(rm_y.grad, torch_y.grad if TORCH_AVAILABLE else None)
                        
                        # 综合结果
                        case_passed = passed_z and passed_dx and passed_dy
                        if not case_passed:
                            device_all_passed = False
                            all_passed = False
                            
                        # 记录子用例结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_sub_case_result(case_passed)
                            status = "通过" if case_passed else "失败"
                            print(f"  子用例: {full_sub_case_name} - {Colors.OKGREEN if case_passed else Colors.FAIL}{status}{Colors.ENDC}")
                            if not case_passed:
                                if not passed_z:
                                    print(f"    乘积值比较: 失败")
                                if not passed_dx:
                                    print(f"    x梯度比较: 失败")
                                if not passed_dy:
                                    print(f"    y梯度比较: 失败")
                    except Exception as e:
                        device_all_passed = False
                        all_passed = False
                        # 记录子用例失败结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_sub_case_result(False)
                            print(f"  子用例: {full_sub_case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                        continue
                    
                    # 重置梯度
                    if hasattr(rm_x, 'grad'):
                        rm_x.grad = None
                    if hasattr(rm_y, 'grad'):
                        rm_y.grad = None
                    
                    if TORCH_AVAILABLE and hasattr(torch_x, 'grad'):
                        torch_x.grad = None
                    if TORCH_AVAILABLE and hasattr(torch_y, 'grad'):
                        torch_y.grad = None
                
                # 记录设备测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(device_case_name, device_all_passed)
                    status = "通过" if device_all_passed else "失败"
                    print(f"测试用例: {device_case_name} - {Colors.OKGREEN if device_all_passed else Colors.FAIL}{status}{Colors.ENDC}")
            
            time_taken = time.time() - start_time
            
            # 断言确保所有测试通过
            self.assertTrue(all_passed, f"矩阵乘法测试失败: 不同维度张量乘法")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("不同维度张量乘法", False, [str(e)])
                print(f"测试用例: 不同维度张量乘法 - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_complex_vector_matrix_multiplication(self):
        """测试场景5: 复数向量与矩阵乘法"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        all_passed = True
        start_time = time.time()
        
        try:
            # 测试用例列表
            test_cases = [
                # (测试名称, x形状, y形状)
                ("一维复数向量与二维复数矩阵乘法", (3,), (3, 4)),
                ("二维复数行向量与二维复数矩阵乘法", (1, 3), (3, 4)),
                ("二维复数列向量与二维复数矩阵乘法", (3, 1), (1, 4)),
                ("二维复数矩阵与一维复数向量乘法", (3, 4), (4,)),
            ]
            
            for device in devices:
                device_case_name = f"复数向量与矩阵乘法 - {device}"
                device_all_passed = True
                
                for sub_case_name, x_shape, y_shape in test_cases:
                    full_sub_case_name = f"{sub_case_name} - {device}"
                    
                    try:
                        # 创建复数测试数据
                        np_x_real = np.random.randn(*x_shape).astype(np.float64)
                        np_x_imag = np.random.randn(*x_shape).astype(np.float64)
                        np_x = np_x_real + 1j * np_x_imag
                        
                        np_y_real = np.random.randn(*y_shape).astype(np.float64)
                        np_y_imag = np.random.randn(*y_shape).astype(np.float64)
                        np_y = np_y_real + 1j * np_y_imag
                        
                        # 创建Riemann张量
                        if device == "cpu":
                            rm_x = tensor(np_x, requires_grad=True)
                            rm_y = tensor(np_y, requires_grad=True)
                        else:  # cuda
                            rm_x = tensor(np_x, requires_grad=True, device=device)
                            rm_y = tensor(np_y, requires_grad=True, device=device)
                        
                        # 创建PyTorch张量
                        if TORCH_AVAILABLE:
                            if device == "cpu":
                                # PyTorch复数张量需要从实数和虚数部分构造
                                torch_x = torch.tensor(np_x, dtype=torch.complex128, requires_grad=True)
                                torch_y = torch.tensor(np_y, dtype=torch.complex128, requires_grad=True)
                            else:  # cuda
                                torch_x = torch.tensor(np_x, dtype=torch.complex128, requires_grad=True, device=device)
                                torch_y = torch.tensor(np_y, dtype=torch.complex128, requires_grad=True, device=device)
                        else:
                            torch_x, torch_y = None, None
                        
                        # 执行矩阵乘法
                        rm_z = rm_x @ rm_y
                        rm_z_sum = rm_z.sum()
                        
                        if TORCH_AVAILABLE:
                            torch_z = torch_x @ torch_y
                            torch_z_sum = torch_z.sum()
                        else:
                            torch_z, torch_z_sum = None, None
                        
                        # 反向传播 - 将复数和转换为实部以支持PyTorch的反向传播
                        rm_z_sum_real = rm_z_sum.real
                        rm_z_sum_real.backward()
                        if TORCH_AVAILABLE:
                            torch_z_sum_real = torch_z_sum.real
                            torch_z_sum_real.backward()
                        
                        # 比较乘法结果
                        passed_z = compare_values(rm_z, torch_z)
                        
                        # 比较梯度
                        passed_dx = compare_values(rm_x.grad, torch_x.grad if TORCH_AVAILABLE else None)
                        passed_dy = compare_values(rm_y.grad, torch_y.grad if TORCH_AVAILABLE else None)
                        
                        # 综合结果
                        case_passed = passed_z and passed_dx and passed_dy
                        if not case_passed:
                            device_all_passed = False
                            all_passed = False
                            
                        # 记录子用例结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_sub_case_result(case_passed)
                            status = "通过" if case_passed else "失败"
                            print(f"  子用例: {full_sub_case_name} - {Colors.OKGREEN if case_passed else Colors.FAIL}{status}{Colors.ENDC}")
                            if not case_passed:
                                if not passed_z:
                                    print(f"    乘积值比较: 失败")
                                if not passed_dx:
                                    print(f"    x梯度比较: 失败")
                                if not passed_dy:
                                    print(f"    y梯度比较: 失败")
                    except Exception as e:
                        device_all_passed = False
                        all_passed = False
                        # 记录子用例失败结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_sub_case_result(False)
                            print(f"  子用例: {full_sub_case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                        continue
                    
                    # 重置梯度
                    if hasattr(rm_x, 'grad'):
                        rm_x.grad = None
                    if hasattr(rm_y, 'grad'):
                        rm_y.grad = None
                    
                    if TORCH_AVAILABLE and hasattr(torch_x, 'grad'):
                        torch_x.grad = None
                    if TORCH_AVAILABLE and hasattr(torch_y, 'grad'):
                        torch_y.grad = None
                
                # 记录设备测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(device_case_name, device_all_passed)
                    status = "通过" if device_all_passed else "失败"
                    print(f"测试用例: {device_case_name} - {Colors.OKGREEN if device_all_passed else Colors.FAIL}{status}{Colors.ENDC}")
            
            time_taken = time.time() - start_time
            
            # 断言确保所有测试通过
            self.assertTrue(all_passed, f"复数矩阵乘法测试失败: 复数向量与矩阵乘法")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("复数向量与矩阵乘法", False, [str(e)])
                print(f"测试用例: 复数向量与矩阵乘法 - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_complex_vector_vector_multiplication(self):
        """测试场景7: 复数向量与复数向量乘法"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        all_passed = True
        start_time = time.time()
        
        try:
            # 测试用例列表
            test_cases = [
                # (测试名称, x形状, y形状)
                ("一维复数向量与一维复数向量乘法", (3,), (3,)),
                ("二维复数行向量与二维复数列向量乘法", (1, 3), (3, 1)),
                ("二维复数列向量与二维复数行向量乘法", (3, 1), (1, 3)),
            ]
            
            for device in devices:
                device_case_name = f"复数向量与复数向量乘法 - {device}"
                device_all_passed = True
                
                for sub_case_name, x_shape, y_shape in test_cases:
                    full_sub_case_name = f"{sub_case_name} - {device}"
                    
                    try:
                        # 创建复数测试数据
                        np_x_real = np.random.randn(*x_shape).astype(np.float64)
                        np_x_imag = np.random.randn(*x_shape).astype(np.float64)
                        np_x = np_x_real + 1.0j * np_x_imag
                        
                        np_y_real = np.random.randn(*y_shape).astype(np.float64)
                        np_y_imag = np.random.randn(*y_shape).astype(np.float64)
                        np_y = np_y_real + 1.0j * np_y_imag
                        
                        # 创建Riemann张量
                        if device == "cpu":
                            rm_x = tensor(np_x, requires_grad=True)
                            rm_y = tensor(np_y, requires_grad=True)
                        else:  # cuda
                            rm_x = tensor(np_x, requires_grad=True, device=device)
                            rm_y = tensor(np_y, requires_grad=True, device=device)
                        
                        # 创建PyTorch张量
                        if TORCH_AVAILABLE:
                            if device == "cpu":
                                torch_x = torch.tensor(np_x, dtype=torch.complex128, requires_grad=True)
                                torch_y = torch.tensor(np_y, dtype=torch.complex128, requires_grad=True)
                            else:  # cuda
                                torch_x = torch.tensor(np_x, dtype=torch.complex128, requires_grad=True, device=device)
                                torch_y = torch.tensor(np_y, dtype=torch.complex128, requires_grad=True, device=device)
                        else:
                            torch_x, torch_y = None, None
                        
                        # 执行矩阵乘法
                        rm_z = rm_x @ rm_y
                        rm_z_sum = rm_z.sum()
                        
                        if TORCH_AVAILABLE:
                            # torch 2.6.0版本有bug，1D复数向量内积计算错误(为0)，改为手动计算内积
                            if torch_x.ndim == 1 and torch_y.ndim == 1:
                                torch_z = (torch_x * torch_y).sum()
                            else:
                                torch_z = torch_x @ torch_y
                            torch_z_sum = torch_z.sum()
                        else:
                            torch_z, torch_z_sum = None, None
                        
                        # 反向传播 - 将复数和转换为实部以支持PyTorch的反向传播
                        rm_z_sum_real = rm_z_sum.real
                        rm_z_sum_real.backward()
                        if TORCH_AVAILABLE:
                            torch_z_sum_real = torch_z_sum.real
                            torch_z_sum_real.backward()
                        
                        # 比较乘法结果
                        passed_z = compare_values(rm_z, torch_z)
                        
                        # 比较梯度
                        passed_dx = compare_values(rm_x.grad, torch_x.grad if TORCH_AVAILABLE else None)
                        passed_dy = compare_values(rm_y.grad, torch_y.grad if TORCH_AVAILABLE else None)
                        
                        # 综合结果
                        case_passed = passed_z and passed_dx and passed_dy
                        if not case_passed:
                            device_all_passed = False
                            all_passed = False
                            
                        if IS_RUNNING_AS_SCRIPT:
                            # 记录子用例结果
                            stats.add_sub_case_result(case_passed)
                            status = "通过" if case_passed else "失败"
                            print(f"  子用例: {full_sub_case_name} - {Colors.OKGREEN if case_passed else Colors.FAIL}{status}{Colors.ENDC}")
                        if not case_passed:
                            if not passed_z:
                                print(f"    乘积值比较: 失败")
                            if not passed_dx:
                                print(f"    x梯度比较: 失败")
                            if not passed_dy:
                                print(f"    y梯度比较: 失败")
                    except Exception as e:
                            device_all_passed = False
                            all_passed = False
                            # 记录子用例失败结果
                            if IS_RUNNING_AS_SCRIPT:
                                # print(f'rm_z={rm_z},\nrm_x={rm_x},\nrm_y={rm_y}')
                                # print(f'torch_z={torch_z},\ntorch_x={torch_x},\ntorch_y={torch_y}')
                                stats.add_sub_case_result(False)
                                print(f"  子用例: {full_sub_case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                    
                    # 重置梯度
                    if hasattr(rm_x, 'grad'):
                        rm_x.grad = None
                    if hasattr(rm_y, 'grad'):
                        rm_y.grad = None
                    
                    if TORCH_AVAILABLE and hasattr(torch_x, 'grad'):
                        torch_x.grad = None
                    if TORCH_AVAILABLE and hasattr(torch_y, 'grad'):
                        torch_y.grad = None
                
                # 记录设备测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(device_case_name, device_all_passed)
                    status = "通过" if device_all_passed else "失败"
                    print(f"测试用例: {device_case_name} - {Colors.OKGREEN if device_all_passed else Colors.FAIL}{status}{Colors.ENDC}")
            
            time_taken = time.time() - start_time
            
            # 断言确保所有测试通过
            self.assertTrue(all_passed, f"复数矩阵乘法测试失败: 复数向量与复数向量乘法")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("复数向量与复数向量乘法", False, [str(e)])
                print(f"测试用例: 复数向量与复数向量乘法 - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise

    def test_complex_matrix_matrix_multiplication(self):
        """测试场景6: 复数矩阵与矩阵乘法"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        all_passed = True
        start_time = time.time()
        
        try:
            # 测试用例列表
            test_cases = [
                # (测试名称, x形状, y形状)
                ("二维复数矩阵与二维复数矩阵乘法", (2, 3), (3, 4)),
                ("三维复数矩阵与二维复数矩阵乘法", (2, 3, 4), (4, 5)),
                ("三维复数矩阵与三维复数矩阵乘法", (2, 3, 4), (2, 4, 5)),
            ]
            
            for device in devices:
                device_case_name = f"复数矩阵与矩阵乘法 - {device}"
                device_all_passed = True
                
                for sub_case_name, x_shape, y_shape in test_cases:
                    full_sub_case_name = f"{sub_case_name} - {device}"
                    
                    try:
                        # 创建复数测试数据
                        np_x_real = np.random.randn(*x_shape).astype(np.float64)
                        np_x_imag = np.random.randn(*x_shape).astype(np.float64)
                        np_x = np_x_real + 1.j * np_x_imag
                        
                        np_y_real = np.random.randn(*y_shape).astype(np.float64)
                        np_y_imag = np.random.randn(*y_shape).astype(np.float64)
                        np_y = np_y_real + 1.j * np_y_imag
                        
                        # 创建Riemann张量
                        if device == "cpu":
                            rm_x = tensor(np_x, requires_grad=True)
                            rm_y = tensor(np_y, requires_grad=True)
                        else:  # cuda
                            rm_x = tensor(np_x, requires_grad=True, device=device)
                            rm_y = tensor(np_y, requires_grad=True, device=device)
                        
                        # 创建PyTorch张量
                        if TORCH_AVAILABLE:
                            if device == "cpu":
                                # PyTorch复数张量需要从实数和虚数部分构造
                                torch_x = torch.tensor(np_x, dtype=torch.complex128, requires_grad=True)
                                torch_y = torch.tensor(np_y, dtype=torch.complex128, requires_grad=True)
                            else:  # cuda
                                torch_x = torch.tensor(np_x, dtype=torch.complex128, requires_grad=True, device=device)
                                torch_y = torch.tensor(np_y, dtype=torch.complex128, requires_grad=True, device=device)
                        else:
                            torch_x, torch_y = None, None
                        
                        # 执行矩阵乘法
                        rm_z = rm_x @ rm_y
                        rm_z_sum = rm_z.sum()
                        
                        if TORCH_AVAILABLE:
                            torch_z = torch_x @ torch_y
                            torch_z_sum = torch_z.sum()
                        else:
                            torch_z, torch_z_sum = None, None
                        
                        # 反向传播 - 将复数和转换为实部以支持PyTorch的反向传播
                        rm_z_sum_real = rm_z_sum.real
                        rm_z_sum_real.backward()
                        if TORCH_AVAILABLE:
                            torch_z_sum_real = torch_z_sum.real
                            torch_z_sum_real.backward()
                        
                        # 比较乘法结果
                        passed_z = compare_values(rm_z, torch_z)
                        
                        # 比较梯度
                        passed_dx = compare_values(rm_x.grad, torch_x.grad if TORCH_AVAILABLE else None)
                        passed_dy = compare_values(rm_y.grad, torch_y.grad if TORCH_AVAILABLE else None)
                        
                        # 综合结果
                        case_passed = passed_z and passed_dx and passed_dy
                        if not case_passed:
                            device_all_passed = False
                            all_passed = False
                            
                        if IS_RUNNING_AS_SCRIPT:
                            # 记录子用例结果
                            stats.add_sub_case_result(case_passed)
                            status = "通过" if case_passed else "失败"
                            print(f"  子用例: {full_sub_case_name} - {Colors.OKGREEN if case_passed else Colors.FAIL}{status}{Colors.ENDC}")
                            if not case_passed:
                                if not passed_z:
                                    print(f"    乘积值比较: 失败")
                                if not passed_dx:
                                    print(f"    x梯度比较: 失败")
                                if not passed_dy:
                                    print(f"    y梯度比较: 失败")
                    except Exception as e:
                            device_all_passed = False
                            all_passed = False
                            # 记录子用例失败结果
                            if IS_RUNNING_AS_SCRIPT:
                                stats.add_sub_case_result(False)
                                print(f"  子用例: {full_sub_case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                    
                    # 重置梯度
                    if hasattr(rm_x, 'grad'):
                        rm_x.grad = None
                    if hasattr(rm_y, 'grad'):
                        rm_y.grad = None
                    
                    if TORCH_AVAILABLE and hasattr(torch_x, 'grad'):
                        torch_x.grad = None
                    if TORCH_AVAILABLE and hasattr(torch_y, 'grad'):
                        torch_y.grad = None
                
                # 记录设备测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(device_case_name, device_all_passed)
                    status = "通过" if device_all_passed else "失败"
                    print(f"测试用例: {device_case_name} - {Colors.OKGREEN if device_all_passed else Colors.FAIL}{status}{Colors.ENDC}")
            
            time_taken = time.time() - start_time
            
            # 断言确保所有测试通过
            self.assertTrue(all_passed, f"复数矩阵乘法测试失败: 复数矩阵与矩阵乘法")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("复数矩阵与矩阵乘法", False, [str(e)])
                print(f"测试用例: 复数矩阵与矩阵乘法 - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行矩阵乘法测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMatmulFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)