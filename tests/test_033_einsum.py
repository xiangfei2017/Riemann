"""
测试riemann的einsum函数实现
"""
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
    print("请确保项目路径设置正确")
    sys.exit(1)


# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True

    # 在模块级别进行PyTorch预热，避免在测试计时中包含初始化开销
    print("预热PyTorch系统...")
    warmup_start = time.time()
    
    # 执行简单的PyTorch操作以触发初始化
    warmup_input = torch.tensor([[0.0]], requires_grad=True)
    warmup_output = warmup_input.sum()
    warmup_output.backward()
    
    # 清理资源
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"PyTorch预热完成，耗时: {time.time() - warmup_start:.4f}秒")
    
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的einsum函数")
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
def compare_values(rm_result, torch_result, atol=1e-5, rtol=1e-5):
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
    
    # 检查形状是否一致
    if rm_result.shape != torch_result.shape:
        return False
    
    # 转换为numpy数组
    try:
        # 处理Riemann结果
        if hasattr(rm_result, 'is_cuda') and rm_result.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            rm_data = rm_result.detach().cpu().numpy()
        else:
            rm_data = rm_result.detach().numpy()
        
        # 处理PyTorch结果
        if hasattr(torch_result, 'is_cuda') and torch_result.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            torch_data = torch_result.detach().cpu().numpy()
        else:
            torch_data = torch_result.detach().cpu().numpy()
    except Exception as e:
        print(f"比较值转换错误: {e}")
        return False
    
    # 处理形状不匹配的情况
    try:
        # 增加容差参数
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False


# 测试einsum类
class TestEinsum(unittest.TestCase):
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
    
    def test_matrix_multiply(self):
        """测试矩阵乘法: ij,jk->ik"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"矩阵乘法 ij,jk->ik - {device}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_A = np.random.randn(3, 4)
                np_B = np.random.randn(4, 5)
                
                # 根据设备创建张量
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:  # cuda
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                # 前向传播测试
                C = rm.einsum('ij,jk->ik', A, B)
                C_torch = None
                if TORCH_AVAILABLE:
                    C_torch = torch.einsum('ij,jk->ik', A_torch, B_torch)
                
                # 比较前向传播结果
                forward_passed = compare_values(C, C_torch)
                
                # 检查设备一致性
                device_passed = C.device == A.device
                
                # 反向传播测试
                backward_passed = True
                if TORCH_AVAILABLE:
                    rm_loss = C.sum()
                    torch_loss = C_torch.sum()
                    rm_loss.backward()
                    torch_loss.backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"矩阵乘法测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_batch_matrix_multiply(self):
        """测试批量矩阵乘法: bij,bjk->bik"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"批量矩阵乘法 bij,bjk->bik - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(10, 3, 4)
                np_B = np.random.randn(10, 4, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                C = rm.einsum('bij,bjk->bik', A, B)
                C_torch = None
                if TORCH_AVAILABLE:
                    C_torch = torch.einsum('bij,bjk->bik', A_torch, B_torch)
                
                forward_passed = compare_values(C, C_torch)
                device_passed = C.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    rm_loss = C.sum()
                    torch_loss = C_torch.sum()
                    rm_loss.backward()
                    torch_loss.backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"批量矩阵乘法测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_trace(self):
        """测试矩阵迹: ii->"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"矩阵迹 ii-> - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(5, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                trace = rm.einsum('ii->', A)
                trace_torch = None
                if TORCH_AVAILABLE:
                    trace_torch = torch.einsum('ii->', A_torch)
                
                forward_passed = compare_values(trace, trace_torch)
                device_passed = trace.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    trace.backward()
                    trace_torch.backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"矩阵迹测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_diagonal(self):
        """测试对角线提取: ii->i"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"对角线提取 ii->i - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(5, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                diag = rm.einsum('ii->i', A)
                diag_torch = None
                if TORCH_AVAILABLE:
                    diag_torch = torch.einsum('ii->i', A_torch)
                
                forward_passed = compare_values(diag, diag_torch)
                device_passed = diag.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    diag.sum().backward()
                    diag_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"对角线提取测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_transpose(self):
        """测试转置: ij->ji"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"转置 ij->ji - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                B = rm.einsum('ij->ji', A)
                B_torch = None
                if TORCH_AVAILABLE:
                    B_torch = torch.einsum('ij->ji', A_torch)
                
                forward_passed = compare_values(B, B_torch)
                device_passed = B.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    B.sum().backward()
                    B_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"转置测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_outer_product(self):
        """测试外积: i,j->ij"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"外积 i,j->ij - {device}"
            start_time = time.time()
            try:
                np_a = np.random.randn(5)
                np_b = np.random.randn(4)
                
                if device == "cpu":
                    a = rm.tensor(np_a, requires_grad=True)
                    b = rm.tensor(np_b, requires_grad=True)
                    if TORCH_AVAILABLE:
                        a_torch = torch.tensor(np_a, requires_grad=True)
                        b_torch = torch.tensor(np_b, requires_grad=True)
                    else:
                        a_torch = b_torch = None
                else:
                    a = rm.tensor(np_a, requires_grad=True, device=device)
                    b = rm.tensor(np_b, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        a_torch = torch.tensor(np_a, requires_grad=True, device=device)
                        b_torch = torch.tensor(np_b, requires_grad=True, device=device)
                    else:
                        a_torch = b_torch = None
                
                C = rm.einsum('i,j->ij', a, b)
                C_torch = None
                if TORCH_AVAILABLE:
                    C_torch = torch.einsum('i,j->ij', a_torch, b_torch)
                
                forward_passed = compare_values(C, C_torch)
                device_passed = C.device == a.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    C.sum().backward()
                    C_torch.sum().backward()
                    backward_passed = compare_values(a.grad, a_torch.grad) and compare_values(b.grad, b_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"外积测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_dot_product(self):
        """测试点积: i,i->"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"点积 i,i-> - {device}"
            start_time = time.time()
            try:
                np_a = np.random.randn(5)
                np_b = np.random.randn(5)
                
                if device == "cpu":
                    a = rm.tensor(np_a, requires_grad=True)
                    b = rm.tensor(np_b, requires_grad=True)
                    if TORCH_AVAILABLE:
                        a_torch = torch.tensor(np_a, requires_grad=True)
                        b_torch = torch.tensor(np_b, requires_grad=True)
                    else:
                        a_torch = b_torch = None
                else:
                    a = rm.tensor(np_a, requires_grad=True, device=device)
                    b = rm.tensor(np_b, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        a_torch = torch.tensor(np_a, requires_grad=True, device=device)
                        b_torch = torch.tensor(np_b, requires_grad=True, device=device)
                    else:
                        a_torch = b_torch = None
                
                dot = rm.einsum('i,i->', a, b)
                dot_torch = None
                if TORCH_AVAILABLE:
                    dot_torch = torch.einsum('i,i->', a_torch, b_torch)
                
                forward_passed = compare_values(dot, dot_torch)
                device_passed = dot.device == a.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    dot.backward()
                    dot_torch.backward()
                    backward_passed = compare_values(a.grad, a_torch.grad) and compare_values(b.grad, b_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"点积测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_tensor_contraction(self):
        """测试张量缩并: ijkl,jklm->ijm"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"张量缩并 ijkl,jklm->ijm - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4, 5, 6)
                np_B = np.random.randn(4, 5, 6, 7)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                C = rm.einsum('ijkl,jklm->ijm', A, B)
                C_torch = None
                if TORCH_AVAILABLE:
                    C_torch = torch.einsum('ijkl,jklm->ijm', A_torch, B_torch)
                
                forward_passed = compare_values(C, C_torch)
                device_passed = C.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    C.sum().backward()
                    C_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"张量缩并测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_single_operand_transpose(self):
        """测试单操作数转置: ij->ji"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"单操作数转置 ij->ji - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                B = rm.einsum('ij->ji', A)
                B_torch = None
                if TORCH_AVAILABLE:
                    B_torch = torch.einsum('ij->ji', A_torch)
                
                forward_passed = compare_values(B, B_torch)
                device_passed = B.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    B.sum().backward()
                    B_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"单操作数转置测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_implicit_output(self):
        """测试隐式输出格式: ij,jk"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"隐式输出格式 ij,jk - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                np_B = np.random.randn(4, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                # 隐式输出
                C = rm.einsum('ij,jk', A, B)
                # 显式输出
                C_explicit = rm.einsum('ij,jk->ik', A, B)
                
                C_torch = None
                if TORCH_AVAILABLE:
                    C_torch = torch.einsum('ij,jk', A_torch, B_torch)
                
                # 验证隐式和显式结果一致
                implicit_explicit_match = np.allclose(C.data, C_explicit.data, atol=1e-6)
                forward_passed = compare_values(C, C_torch) and implicit_explicit_match
                device_passed = C.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    C.sum().backward()
                    C_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"隐式输出测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_three_operands(self):
        """测试三个操作数: ij,jk,kl->il"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"三个操作数 ij,jk,kl->il - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                np_B = np.random.randn(4, 5)
                np_C = np.random.randn(5, 6)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    C = rm.tensor(np_C, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                        C_torch = torch.tensor(np_C, requires_grad=True)
                    else:
                        A_torch = B_torch = C_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    C = rm.tensor(np_C, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                        C_torch = torch.tensor(np_C, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = C_torch = None
                
                D = rm.einsum('ij,jk,kl->il', A, B, C)
                D_torch = None
                if TORCH_AVAILABLE:
                    D_torch = torch.einsum('ij,jk,kl->il', A_torch, B_torch, C_torch)
                
                forward_passed = compare_values(D, D_torch)
                device_passed = D.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    D.sum().backward()
                    D_torch.sum().backward()
                    backward_passed = (compare_values(A.grad, A_torch.grad) and 
                                     compare_values(B.grad, B_torch.grad) and 
                                     compare_values(C.grad, C_torch.grad))
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"三个操作数测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_batch_matrix_trace(self):
        """测试批量矩阵迹: bii->b"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"批量矩阵迹 bii->b - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(10, 5, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                traces = rm.einsum('bii->b', A)
                traces_torch = None
                if TORCH_AVAILABLE:
                    traces_torch = torch.einsum('bii->b', A_torch)
                
                forward_passed = compare_values(traces, traces_torch)
                device_passed = traces.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    traces.sum().backward()
                    traces_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"批量矩阵迹测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_complex_einsum(self):
        """测试复数张量einsum"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        test_cases = [
            {"name": "复数矩阵乘法", "eq": "ij,jk->ik", "shapes": [(3, 4), (4, 5)]},
            {"name": "复数批量矩阵乘法", "eq": "bij,bjk->bik", "shapes": [(3, 4, 5), (3, 5, 6)]},
            {"name": "复数迹", "eq": "ii->", "shapes": [(5, 5)]},
            {"name": "复数对角线提取", "eq": "ii->i", "shapes": [(5, 5)]},
            {"name": "复数转置", "eq": "ij->ji", "shapes": [(3, 4)]},
            {"name": "复数点积", "eq": "i,i->", "shapes": [(5,), (5,)]},
            {"name": "复数外积", "eq": "i,j->ij", "shapes": [(5,), (4,)]},
        ]
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建复数测试数据
                    shapes = case['shapes']
                    eq = case['eq']
                    
                    if len(shapes) == 1:
                        np_A = np.random.randn(*shapes[0]) + 1j * np.random.randn(*shapes[0])
                        
                        if device == "cpu":
                            A = rm.tensor(np_A, requires_grad=True)
                            if TORCH_AVAILABLE:
                                A_torch = torch.tensor(np_A, dtype=torch.complex64, requires_grad=True)
                            else:
                                A_torch = None
                        else:
                            A = rm.tensor(np_A, requires_grad=True, device=device)
                            if TORCH_AVAILABLE:
                                A_torch = torch.tensor(np_A, dtype=torch.complex64, requires_grad=True, device=device)
                            else:
                                A_torch = None
                        
                        result = rm.einsum(eq, A)
                        result_torch = None
                        if TORCH_AVAILABLE:
                            result_torch = torch.einsum(eq, A_torch)
                        
                        forward_passed = compare_values(result, result_torch, atol=1e-5)
                        device_passed = result.device == A.device
                        
                        backward_passed = True
                        if TORCH_AVAILABLE:
                            if result.ndim == 0:
                                # 复数标量需要显式梯度，确保 dtype 匹配
                                grad = rm.tensor(1+0j, dtype=result.dtype)
                                result.backward(grad)
                                result_torch.backward(torch.tensor(1+0j, dtype=torch.complex64))
                            else:
                                grad = rm.ones_like(result)
                                result.backward(grad)
                                result_torch.backward(torch.ones_like(result_torch, dtype=torch.complex64))
                            backward_passed = compare_values(A.grad, A_torch.grad, atol=1e-5)
                    
                    elif len(shapes) == 2:
                        np_A = np.random.randn(*shapes[0]) + 1j * np.random.randn(*shapes[0])
                        np_B = np.random.randn(*shapes[1]) + 1j * np.random.randn(*shapes[1])
                        
                        if device == "cpu":
                            A = rm.tensor(np_A, requires_grad=True)
                            B = rm.tensor(np_B, requires_grad=True)
                            if TORCH_AVAILABLE:
                                A_torch = torch.tensor(np_A, dtype=torch.complex64, requires_grad=True)
                                B_torch = torch.tensor(np_B, dtype=torch.complex64, requires_grad=True)
                            else:
                                A_torch = B_torch = None
                        else:
                            A = rm.tensor(np_A, requires_grad=True, device=device)
                            B = rm.tensor(np_B, requires_grad=True, device=device)
                            if TORCH_AVAILABLE:
                                A_torch = torch.tensor(np_A, dtype=torch.complex64, requires_grad=True, device=device)
                                B_torch = torch.tensor(np_B, dtype=torch.complex64, requires_grad=True, device=device)
                            else:
                                A_torch = B_torch = None
                        
                        result = rm.einsum(eq, A, B)
                        result_torch = None
                        if TORCH_AVAILABLE:
                            result_torch = torch.einsum(eq, A_torch, B_torch)
                        
                        forward_passed = compare_values(result, result_torch, atol=1e-5)
                        device_passed = result.device == A.device
                        
                        backward_passed = True
                        if TORCH_AVAILABLE:
                            if result.ndim == 0:
                                # 复数标量需要显式梯度，确保 dtype 匹配
                                grad = rm.tensor(1+0j, dtype=result.dtype)
                                result.backward(grad)
                                result_torch.backward(torch.tensor(1+0j, dtype=torch.complex64))
                            else:
                                grad = rm.ones_like(result)
                                result.backward(grad)
                                result_torch.backward(torch.ones_like(result_torch, dtype=torch.complex64))
                            backward_passed = (compare_values(A.grad, A_torch.grad, atol=1e-5) and 
                                             compare_values(B.grad, B_torch.grad, atol=1e-5))
                    
                    passed = forward_passed and backward_passed and device_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"复数einsum测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def test_sum_reduction(self):
        """测试求和归约: ij->, i->"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        test_cases = [
            {"name": "矩阵求和 ij->", "eq": "ij->", "shape": (3, 4), "ndim": 1},
            {"name": "向量求和 i->", "eq": "i->", "shape": (5,), "ndim": 1},
            {"name": "3D张量求和 ijk->", "eq": "ijk->", "shape": (2, 3, 4), "ndim": 1},
        ]
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    np_A = np.random.randn(*case['shape'])
                    
                    if device == "cpu":
                        A = rm.tensor(np_A, requires_grad=True)
                        if TORCH_AVAILABLE:
                            A_torch = torch.tensor(np_A, requires_grad=True)
                        else:
                            A_torch = None
                    else:
                        A = rm.tensor(np_A, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        else:
                            A_torch = None
                    
                    result = rm.einsum(case['eq'], A)
                    result_torch = None
                    if TORCH_AVAILABLE:
                        result_torch = torch.einsum(case['eq'], A_torch)
                    
                    forward_passed = compare_values(result, result_torch)
                    device_passed = result.device == A.device
                    
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        result.backward()
                        result_torch.backward()
                        backward_passed = compare_values(A.grad, A_torch.grad)
                    
                    passed = forward_passed and backward_passed and device_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"求和归约测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_row_column_sum(self):
        """测试行/列求和: ij->i, ij->j"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        test_cases = [
            {"name": "行求和 ij->i", "eq": "ij->i", "shape": (3, 4), "ndim": 1},
            {"name": "列求和 ij->j", "eq": "ij->j", "shape": (3, 4), "ndim": 1},
            {"name": "3D张量轴求和 ijk->ij", "eq": "ijk->ij", "shape": (2, 3, 4), "ndim": 1},
            {"name": "3D张量轴求和 ijk->ik", "eq": "ijk->ik", "shape": (2, 3, 4), "ndim": 1},
        ]
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    np_A = np.random.randn(*case['shape'])
                    
                    if device == "cpu":
                        A = rm.tensor(np_A, requires_grad=True)
                        if TORCH_AVAILABLE:
                            A_torch = torch.tensor(np_A, requires_grad=True)
                        else:
                            A_torch = None
                    else:
                        A = rm.tensor(np_A, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        else:
                            A_torch = None
                    
                    result = rm.einsum(case['eq'], A)
                    result_torch = None
                    if TORCH_AVAILABLE:
                        result_torch = torch.einsum(case['eq'], A_torch)
                    
                    forward_passed = compare_values(result, result_torch)
                    device_passed = result.device == A.device
                    
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        result.sum().backward()
                        result_torch.sum().backward()
                        backward_passed = compare_values(A.grad, A_torch.grad)
                    
                    passed = forward_passed and backward_passed and device_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"行/列求和测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_frobenius_inner_product(self):
        """测试Frobenius内积: ij,ij->"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"Frobenius内积 ij,ij-> - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                np_B = np.random.randn(3, 4)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('ij,ij->', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ij,ij->', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.backward()
                    result_torch.backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"Frobenius内积测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_hadamard_product(self):
        """测试逐元素乘积: ij,ij->ij"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"逐元素乘积 ij,ij->ij - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                np_B = np.random.randn(3, 4)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('ij,ij->ij', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ij,ij->ij', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"逐元素乘积测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_kronecker_product(self):
        """测试Kronecker类外积: ij,kl->ijkl"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"Kronecker外积 ij,kl->ijkl - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(2, 3)
                np_B = np.random.randn(4, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('ij,kl->ijkl', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ij,kl->ijkl', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"Kronecker外积测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_partial_contraction(self):
        """测试部分缩并: ijk,ikl->ijl"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"部分缩并 ijk,ikl->ijl - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(2, 3, 4)
                np_B = np.random.randn(2, 4, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('ijk,ikl->ijl', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ijk,ikl->ijl', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"部分缩并测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_batch_diagonal_extraction(self):
        """测试批量对角线提取: bii->bi"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"批量对角线提取 bii->bi - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(10, 5, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                diag = rm.einsum('bii->bi', A)
                diag_torch = None
                if TORCH_AVAILABLE:
                    diag_torch = torch.einsum('bii->bi', A_torch)
                
                forward_passed = compare_values(diag, diag_torch)
                device_passed = diag.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    diag.sum().backward()
                    diag_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"批量对角线提取测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_higher_dim_transpose(self):
        """测试高维转置: ijk->kji, ijkl->klij"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        test_cases = [
            {"name": "3D转置 ijk->kji", "eq": "ijk->kji", "shape": (2, 3, 4)},
            {"name": "4D转置 ijkl->klij", "eq": "ijkl->klij", "shape": (2, 3, 4, 5)},
            {"name": "3D转置 ijk->jki", "eq": "ijk->jki", "shape": (2, 3, 4)},
        ]
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    np_A = np.random.randn(*case['shape'])
                    
                    if device == "cpu":
                        A = rm.tensor(np_A, requires_grad=True)
                        if TORCH_AVAILABLE:
                            A_torch = torch.tensor(np_A, requires_grad=True)
                        else:
                            A_torch = None
                    else:
                        A = rm.tensor(np_A, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        else:
                            A_torch = None
                    
                    result = rm.einsum(case['eq'], A)
                    result_torch = None
                    if TORCH_AVAILABLE:
                        result_torch = torch.einsum(case['eq'], A_torch)
                    
                    forward_passed = compare_values(result, result_torch)
                    device_passed = result.device == A.device
                    
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        result.sum().backward()
                        result_torch.sum().backward()
                        backward_passed = compare_values(A.grad, A_torch.grad)
                    
                    passed = forward_passed and backward_passed and device_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"高维转置测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_identity_copy(self):
        """测试恒等/复制操作: ij->ij"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"恒等复制 ij->ij - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                result = rm.einsum('ij->ij', A)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ij->ij', A_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"恒等复制测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_duplicate_first_indices(self):
        """测试前两个索引重复: iij->j, iij->ij"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        test_cases = [
            {"name": "前两索引重复求和 iij->j", "eq": "iij->j", "shape": (3, 3, 4)},
            {"name": "前两索引重复部分输出 iij->ij", "eq": "iij->ij", "shape": (3, 3, 4)},
        ]
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    np_A = np.random.randn(*case['shape'])
                    
                    if device == "cpu":
                        A = rm.tensor(np_A, requires_grad=True)
                        if TORCH_AVAILABLE:
                            A_torch = torch.tensor(np_A, requires_grad=True)
                        else:
                            A_torch = None
                    else:
                        A = rm.tensor(np_A, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        else:
                            A_torch = None
                    
                    result = rm.einsum(case['eq'], A)
                    result_torch = None
                    if TORCH_AVAILABLE:
                        result_torch = torch.einsum(case['eq'], A_torch)
                    
                    forward_passed = compare_values(result, result_torch)
                    device_passed = result.device == A.device
                    
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        result.sum().backward()
                        result_torch.sum().backward()
                        backward_passed = compare_values(A.grad, A_torch.grad)
                    
                    passed = forward_passed and backward_passed and device_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"前两索引重复测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_ellipsis_implicit_output(self):
        """测试省略号隐式输出: ...ij,...jk"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"省略号隐式输出 ...ij,...jk - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(2, 3, 4)
                np_B = np.random.randn(2, 4, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('...ij,...jk', A, B)
                result_explicit = rm.einsum('...ij,...jk->...ik', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('...ij,...jk', A_torch, B_torch)
                
                implicit_explicit_match = np.allclose(result.data, result_explicit.data, atol=1e-6)
                forward_passed = compare_values(result, result_torch) and implicit_explicit_match
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"省略号隐式输出测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_batch_outer_product(self):
        """测试批量外积: bi,bj->bij"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"批量外积 bi,bj->bij - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(10, 3)
                np_B = np.random.randn(10, 4)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('bi,bj->bij', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('bi,bj->bij', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"批量外积测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_self_contraction(self):
        """测试自缩并: ij,ij->i"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"自缩并 ij,ij->i - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                np_B = np.random.randn(3, 4)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('ij,ij->i', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ij,ij->i', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"自缩并测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_ellipsis_operations(self):
        """测试省略号（...）批量操作场景"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        test_cases = [
            {"name": "批量外积", "eq": "...i,...j->...ij", "shapes": [(2, 3), (2, 4)], "ndim": 2},
            {"name": "批量外积（3D批次）", "eq": "...i,...j->...ij", "shapes": [(2, 3, 4), (2, 3, 5)], "ndim": 2},
            {"name": "批量矩阵乘法", "eq": "...ij,...jk->...ik", "shapes": [(2, 3, 4), (2, 4, 5)], "ndim": 2},
            {"name": "批量矩阵乘法（3D批次）", "eq": "...ij,...jk->...ik", "shapes": [(2, 3, 4, 5), (2, 3, 5, 6)], "ndim": 2},
            {"name": "批量转置", "eq": "...ij->...ji", "shapes": [(2, 3, 4)], "ndim": 1},
            {"name": "批量迹", "eq": "...ii->...", "shapes": [(2, 3, 4, 4)], "ndim": 1},
            {"name": "批量对角线提取", "eq": "...ii->...i", "shapes": [(2, 3, 4, 4)], "ndim": 1},
            {"name": "批量点积", "eq": "...i,...i->...", "shapes": [(2, 3, 4), (2, 3, 4)], "ndim": 2},
        ]
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    shapes = case['shapes']
                    eq = case['eq']
                    
                    if len(shapes) == 1:
                        # 单操作数场景
                        np_A = np.random.randn(*shapes[0])
                        
                        if device == "cpu":
                            A = rm.tensor(np_A, requires_grad=True)
                            if TORCH_AVAILABLE:
                                A_torch = torch.tensor(np_A, requires_grad=True)
                            else:
                                A_torch = None
                        else:
                            A = rm.tensor(np_A, requires_grad=True, device=device)
                            if TORCH_AVAILABLE:
                                A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                            else:
                                A_torch = None
                        
                        result = rm.einsum(eq, A)
                        result_torch = None
                        if TORCH_AVAILABLE:
                            result_torch = torch.einsum(eq, A_torch)
                        
                        forward_passed = compare_values(result, result_torch)
                        device_passed = result.device == A.device
                        
                        backward_passed = True
                        if TORCH_AVAILABLE:
                            grad = rm.ones_like(result)
                            result.backward(grad)
                            result_torch.backward(torch.ones_like(result_torch))
                            backward_passed = compare_values(A.grad, A_torch.grad)
                    
                    else:
                        # 双操作数场景
                        np_A = np.random.randn(*shapes[0])
                        np_B = np.random.randn(*shapes[1])
                        
                        if device == "cpu":
                            A = rm.tensor(np_A, requires_grad=True)
                            B = rm.tensor(np_B, requires_grad=True)
                            if TORCH_AVAILABLE:
                                A_torch = torch.tensor(np_A, requires_grad=True)
                                B_torch = torch.tensor(np_B, requires_grad=True)
                            else:
                                A_torch = B_torch = None
                        else:
                            A = rm.tensor(np_A, requires_grad=True, device=device)
                            B = rm.tensor(np_B, requires_grad=True, device=device)
                            if TORCH_AVAILABLE:
                                A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                                B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                            else:
                                A_torch = B_torch = None
                        
                        result = rm.einsum(eq, A, B)
                        result_torch = None
                        if TORCH_AVAILABLE:
                            result_torch = torch.einsum(eq, A_torch, B_torch)
                        
                        forward_passed = compare_values(result, result_torch)
                        device_passed = result.device == A.device
                        
                        backward_passed = True
                        if TORCH_AVAILABLE:
                            grad = rm.ones_like(result)
                            result.backward(grad)
                            result_torch.backward(torch.ones_like(result_torch))
                            backward_passed = (compare_values(A.grad, A_torch.grad) and 
                                             compare_values(B.grad, B_torch.grad))
                    
                    passed = forward_passed and backward_passed and device_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"省略号操作测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def test_single_operand_ellipsis_implicit(self):
        """测试单操作数省略号隐式输出: ...ij"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"单操作数省略号隐式输出 ...ij - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(2, 3, 4)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                # 隐式输出应该等价于显式输出 ...ij->...ij（复制）
                result = rm.einsum('...ij', A)
                result_explicit = rm.einsum('...ij->...ij', A)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('...ij', A_torch)
                
                implicit_explicit_match = np.allclose(result.data, result_explicit.data, atol=1e-6)
                forward_passed = compare_values(result, result_torch) and implicit_explicit_match
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"单操作数省略号隐式输出测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_single_operand_ellipsis_trace_implicit(self):
        """测试单操作数省略号迹隐式输出: ...ii（隐式输出为迹）"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"单操作数省略号迹隐式输出 ...ii - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(2, 3, 4, 4)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                # 隐式输出 ...ii 应该等价于显式输出 ...ii->...（迹/求和）
                result = rm.einsum('...ii', A)
                result_explicit = rm.einsum('...ii->...', A)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('...ii', A_torch)
                
                implicit_explicit_match = np.allclose(result.data, result_explicit.data, atol=1e-6)
                forward_passed = compare_values(result, result_torch) and implicit_explicit_match
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"单操作数省略号迹隐式输出测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_multi_operand_complex_sum_reduction(self):
        """测试多操作数复杂情况下的求和归约: iij,ijk->k"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"多操作数复杂求和归约 iij,ijk->k - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 3, 4)
                np_B = np.random.randn(3, 4, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('iij,ijk->k', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('iij,ijk->k', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"多操作数复杂求和归约测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_multiple_duplicate_indices(self):
        """测试多索引重复: iijj->ij"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"多索引重复 iijj->ij - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 3, 4, 4)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                result = rm.einsum('iijj->ij', A)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('iijj->ij', A_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"多索引重复测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_full_contraction(self):
        """测试双操作数全缩并: ij,ji->"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"双操作数全缩并 ij,ji-> - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                np_B = np.random.randn(4, 3)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('ij,ji->', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ij,ji->', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.backward()
                    result_torch.backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"双操作数全缩并测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_3d_tensor_multiply(self):
        """测试3D张量乘法: ijk,jkl->il"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"3D张量乘法 ijk,jkl->il - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4, 5)
                np_B = np.random.randn(4, 5, 6)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('ijk,jkl->il', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ijk,jkl->il', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"3D张量乘法测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_noncontiguous_duplicate(self):
        """测试非连续索引重复: ijji->"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"非连续索引重复 ijji-> - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4, 4, 3)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                    else:
                        A_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                    else:
                        A_torch = None
                
                result = rm.einsum('ijji->', A)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ijji->', A_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.backward()
                    result_torch.backward()
                    backward_passed = compare_values(A.grad, A_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"非连续索引重复测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_four_operands_chain(self):
        """测试四操作数链式运算: ij,jk,kl,lm->im"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"四操作数链式 ij,jk,kl,lm->im - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                np_B = np.random.randn(4, 5)
                np_C = np.random.randn(5, 6)
                np_D = np.random.randn(6, 7)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    C = rm.tensor(np_C, requires_grad=True)
                    D = rm.tensor(np_D, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                        C_torch = torch.tensor(np_C, requires_grad=True)
                        D_torch = torch.tensor(np_D, requires_grad=True)
                    else:
                        A_torch = B_torch = C_torch = D_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    C = rm.tensor(np_C, requires_grad=True, device=device)
                    D = rm.tensor(np_D, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                        C_torch = torch.tensor(np_C, requires_grad=True, device=device)
                        D_torch = torch.tensor(np_D, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = C_torch = D_torch = None
                
                result = rm.einsum('ij,jk,kl,lm->im', A, B, C, D)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ij,jk,kl,lm->im', A_torch, B_torch, C_torch, D_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = (compare_values(A.grad, A_torch.grad) and 
                                      compare_values(B.grad, B_torch.grad) and
                                      compare_values(C.grad, C_torch.grad) and
                                      compare_values(D.grad, D_torch.grad))
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"四操作数链式测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_multi_operand_mixed(self):
        """测试多操作数混合运算: ij,jk,ik->i"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"多操作数混合 ij,jk,ik->i - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4)
                np_B = np.random.randn(4, 5)
                np_C = np.random.randn(3, 5)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    C = rm.tensor(np_C, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                        C_torch = torch.tensor(np_C, requires_grad=True)
                    else:
                        A_torch = B_torch = C_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    C = rm.tensor(np_C, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                        C_torch = torch.tensor(np_C, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = C_torch = None
                
                result = rm.einsum('ij,jk,ik->i', A, B, C)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ij,jk,ik->i', A_torch, B_torch, C_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = (compare_values(A.grad, A_torch.grad) and 
                                      compare_values(B.grad, B_torch.grad) and
                                      compare_values(C.grad, C_torch.grad))
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"多操作数混合测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_batch_three_operands(self):
        """测试批量三操作数: ...ij,...jk,...kl->...il"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"批量三操作数 ...ij,...jk,...kl->...il - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(2, 3, 3, 4)
                np_B = np.random.randn(2, 3, 4, 5)
                np_C = np.random.randn(2, 3, 5, 6)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    C = rm.tensor(np_C, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                        C_torch = torch.tensor(np_C, requires_grad=True)
                    else:
                        A_torch = B_torch = C_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    C = rm.tensor(np_C, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                        C_torch = torch.tensor(np_C, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = C_torch = None
                
                result = rm.einsum('...ij,...jk,...kl->...il', A, B, C)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('...ij,...jk,...kl->...il', A_torch, B_torch, C_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = (compare_values(A.grad, A_torch.grad) and 
                                      compare_values(B.grad, B_torch.grad) and
                                      compare_values(C.grad, C_torch.grad))
                
                passed = forward_passed and backward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"批量三操作数测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_float32_dtype(self):
        """测试float32数据类型: ij,jk->ik"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"float32数据类型 ij,jk->ik - {device}"
            start_time = time.time()
            try:
                np_A = np.random.randn(3, 4).astype(np.float32)
                np_B = np.random.randn(4, 5).astype(np.float32)
                
                if device == "cpu":
                    A = rm.tensor(np_A, requires_grad=True)
                    B = rm.tensor(np_B, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        B_torch = torch.tensor(np_B, requires_grad=True)
                    else:
                        A_torch = B_torch = None
                else:
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    B = rm.tensor(np_B, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_torch = torch.tensor(np_B, requires_grad=True, device=device)
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('ij,jk->ik', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ij,jk->ik', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                dtype_passed = result.dtype == np.float32
                
                backward_passed = True
                if TORCH_AVAILABLE:
                    result.sum().backward()
                    result_torch.sum().backward()
                    backward_passed = compare_values(A.grad, A_torch.grad) and compare_values(B.grad, B_torch.grad)
                
                passed = forward_passed and backward_passed and device_passed and dtype_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"float32数据类型测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_1d_vector_operations(self):
        """测试1维张量各种操作"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"1维向量操作 - {device}"
            start_time = time.time()
            try:
                np_a = np.random.randn(5)
                np_b = np.random.randn(5)
                np_A = np.random.randn(3, 5)
                # 用于向量乘矩阵测试，需要向量长度匹配矩阵行数
                np_a_row = np.random.randn(3)
                
                if device == "cpu":
                    a = rm.tensor(np_a, requires_grad=True)
                    b = rm.tensor(np_b, requires_grad=True)
                    A = rm.tensor(np_A, requires_grad=True)
                    a_row = rm.tensor(np_a_row, requires_grad=True)
                    if TORCH_AVAILABLE:
                        a_torch = torch.tensor(np_a, requires_grad=True)
                        b_torch = torch.tensor(np_b, requires_grad=True)
                        A_torch = torch.tensor(np_A, requires_grad=True)
                        a_row_torch = torch.tensor(np_a_row, requires_grad=True)
                    else:
                        a_torch = b_torch = A_torch = a_row_torch = None
                else:
                    a = rm.tensor(np_a, requires_grad=True, device=device)
                    b = rm.tensor(np_b, requires_grad=True, device=device)
                    A = rm.tensor(np_A, requires_grad=True, device=device)
                    a_row = rm.tensor(np_a_row, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        a_torch = torch.tensor(np_a, requires_grad=True, device=device)
                        b_torch = torch.tensor(np_b, requires_grad=True, device=device)
                        A_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        a_row_torch = torch.tensor(np_a_row, requires_grad=True, device=device)
                    else:
                        a_torch = b_torch = A_torch = a_row_torch = None
                
                # 测试多个1D操作
                tests_passed = []
                
                # 1. 向量点积: i,i->
                result1 = rm.einsum('i,i->', a, b)
                result1_torch = torch.einsum('i,i->', a_torch, b_torch) if TORCH_AVAILABLE else None
                tests_passed.append(compare_values(result1, result1_torch))
                
                # 2. 向量外积: i,j->ij
                result2 = rm.einsum('i,j->ij', a, b)
                result2_torch = torch.einsum('i,j->ij', a_torch, b_torch) if TORCH_AVAILABLE else None
                tests_passed.append(compare_values(result2, result2_torch))
                
                # 3. 矩阵乘向量: ij,j->i
                result3 = rm.einsum('ij,j->i', A, b)
                result3_torch = torch.einsum('ij,j->i', A_torch, b_torch) if TORCH_AVAILABLE else None
                tests_passed.append(compare_values(result3, result3_torch))
                
                # 4. 向量乘矩阵: i,ij->j (注意：i是行索引，向量长度需匹配矩阵行数)
                result4 = rm.einsum('i,ij->j', a_row, A)
                result4_torch = torch.einsum('i,ij->j', a_row_torch, A_torch) if TORCH_AVAILABLE else None
                tests_passed.append(compare_values(result4, result4_torch))
                
                # 5. 批量矩阵乘向量: bij,j->bi
                np_A_batch = np.random.randn(2, 3, 5)
                np_b_batch = np.random.randn(2, 5)
                if device == "cpu":
                    A_batch = rm.tensor(np_A_batch, requires_grad=True)
                    b_batch = rm.tensor(np_b_batch, requires_grad=True)
                    if TORCH_AVAILABLE:
                        A_batch_torch = torch.tensor(np_A_batch, requires_grad=True)
                        b_batch_torch = torch.tensor(np_b_batch, requires_grad=True)
                    else:
                        A_batch_torch = b_batch_torch = None
                else:
                    A_batch = rm.tensor(np_A_batch, requires_grad=True, device=device)
                    b_batch = rm.tensor(np_b_batch, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        A_batch_torch = torch.tensor(np_A_batch, requires_grad=True, device=device)
                        b_batch_torch = torch.tensor(np_b_batch, requires_grad=True, device=device)
                    else:
                        A_batch_torch = b_batch_torch = None
                
                result5 = rm.einsum('bij,bj->bi', A_batch, b_batch)
                result5_torch = torch.einsum('bij,bj->bi', A_batch_torch, b_batch_torch) if TORCH_AVAILABLE else None
                tests_passed.append(compare_values(result5, result5_torch))
                
                forward_passed = all(tests_passed)
                device_passed = result1.device == a.device
                
                passed = forward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"1维向量操作测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_non_contiguous_stride(self):
        """测试非标准步长张量"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            case_name = f"非标准步长 - {device}"
            start_time = time.time()
            try:
                # 创建非连续张量（通过切片）
                np_A = np.random.randn(6, 8)
                np_B = np.random.randn(8, 10)
                
                if device == "cpu":
                    # 创建基础张量，然后通过切片获取非连续视图
                    A_base = rm.tensor(np_A, requires_grad=True)
                    B_base = rm.tensor(np_B, requires_grad=True)
                    # 切片会创建非连续张量
                    A = A_base[::2, ::2]  # 形状 (3, 4)，非连续
                    B = B_base[::2, ::2]  # 形状 (4, 5)，非连续
                    
                    if TORCH_AVAILABLE:
                        A_base_torch = torch.tensor(np_A, requires_grad=True)
                        B_base_torch = torch.tensor(np_B, requires_grad=True)
                        A_torch = A_base_torch[::2, ::2]
                        B_torch = B_base_torch[::2, ::2]
                    else:
                        A_torch = B_torch = None
                else:
                    A_base = rm.tensor(np_A, requires_grad=True, device=device)
                    B_base = rm.tensor(np_B, requires_grad=True, device=device)
                    A = A_base[::2, ::2]
                    B = B_base[::2, ::2]
                    
                    if TORCH_AVAILABLE:
                        A_base_torch = torch.tensor(np_A, requires_grad=True, device=device)
                        B_base_torch = torch.tensor(np_B, requires_grad=True, device=device)
                        A_torch = A_base_torch[::2, ::2]
                        B_torch = B_base_torch[::2, ::2]
                    else:
                        A_torch = B_torch = None
                
                result = rm.einsum('ij,jk->ik', A, B)
                result_torch = None
                if TORCH_AVAILABLE:
                    result_torch = torch.einsum('ij,jk->ik', A_torch, B_torch)
                
                forward_passed = compare_values(result, result_torch)
                device_passed = result.device == A.device
                
                passed = forward_passed and device_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"非标准步长测试失败: {case_name}")
                
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
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行einsum函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}CUDA 可用: {CUDA_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEinsum)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)
