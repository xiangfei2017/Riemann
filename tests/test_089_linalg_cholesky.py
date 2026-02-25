import unittest
import numpy as np
import time
import sys, os

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import riemann as rm
    from riemann.linalg import cholesky as rm_cholesky
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

try:
    import torch
    from torch.linalg import cholesky as torch_cholesky
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的cholesky函数")
    TORCH_AVAILABLE = False

# 用于终端彩色输出
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

# 统计收集器类
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

# 创建统计收集器实例
stats = StatisticsCollector()

# 是否作为独立脚本运行
IS_RUNNING_AS_SCRIPT = False

# 比较riemann和torch的结果
def compare_values(rm_result, torch_result, atol=1e-6, rtol=1e-6):
    """比较riemann张量和torch张量的值"""
    if isinstance(rm_result, tuple) and isinstance(torch_result, tuple):
        if len(rm_result) != len(torch_result):
            return False
        return all(compare_values(r, t, atol, rtol) for r, t in zip(rm_result, torch_result))
    
    # 获取numpy数组
    if hasattr(rm_result, 'numpy'):
        rm_array = rm_result.detach().numpy() if hasattr(rm_result, 'detach') else rm_result.numpy()
    else:
        rm_array = np.array(rm_result)
    
    if hasattr(torch_result, 'numpy'):
        torch_array = torch_result.detach().numpy()
    else:
        torch_array = np.array(torch_result)
    
    # 处理复数情况
    if np.iscomplexobj(rm_array) or np.iscomplexobj(torch_array):
        rm_real = np.real(rm_array)
        rm_imag = np.imag(rm_array)
        torch_real = np.real(torch_array)
        torch_imag = np.imag(torch_array)
        return (np.allclose(rm_real, torch_real, atol=atol, rtol=rtol) and 
                np.allclose(rm_imag, torch_imag, atol=atol, rtol=rtol))
    
    # 实数情况
    return np.allclose(rm_array, torch_array, atol=atol, rtol=rtol)

# 检查Cholesky分解的重构
def check_cholesky_reconstruction(A, L, upper=False, atol=1e-6):
    """检查Cholesky分解的重构是否正确：A ≈ L @ L^T 或 A ≈ L^T @ L（当upper=True时）"""
    try:
        # 处理upper参数
        if upper:
            # 上三角矩阵：A = L^T @ L
            reconstructed = L.mH @ L
        else:
            # 下三角矩阵：A = L @ L^T
            reconstructed = L @ L.mH
        
        # 计算重构误差
        A_array = A.detach().numpy() if hasattr(A, 'detach') else A.numpy()
        reconstructed_array = reconstructed.detach().numpy() if hasattr(reconstructed, 'detach') else reconstructed.numpy()
        
        error = np.linalg.norm(A_array - reconstructed_array)
        reconstruction_ok = error < atol
        
        # 检查L是否为三角矩阵
        L_array = L.detach().numpy() if hasattr(L, 'detach') else L.numpy()
        
        if len(L_array.shape) > 2:  # 批量矩阵
            # 对每个批次分别检查
            is_triangular = True
            for i in range(L_array.shape[0]):
                batch_L = L_array[i]
                if upper:
                    # 检查是否为上三角矩阵
                    batch_lower = np.tril(batch_L, k=-1)
                    is_triangular = is_triangular and np.allclose(batch_lower, 0, atol=atol)
                else:
                    # 检查是否为下三角矩阵
                    batch_upper = np.triu(batch_L, k=1)
                    is_triangular = is_triangular and np.allclose(batch_upper, 0, atol=atol)
        else:  # 单个矩阵
            if upper:
                # 检查是否为上三角矩阵
                L_lower = np.tril(L_array, k=-1)
                is_triangular = np.allclose(L_lower, 0, atol=atol)
            else:
                # 检查是否为下三角矩阵
                L_upper = np.triu(L_array, k=1)
                is_triangular = np.allclose(L_upper, 0, atol=atol)
        
        return reconstruction_ok and is_triangular
    except Exception as e:
        print(f"重构检查错误: {str(e)}")
        return False

# Cholesky分解的梯度测试函数
def gradient_for_cholesky(A, case_name, upper=False):
    """统一的Cholesky梯度测试函数
    
    参数:
        A: 输入矩阵
        case_name: 测试用例名称
        upper: 是否返回上三角矩阵，默认为False
    
    返回:
        (grad_close, error_msg): 梯度比较结果和错误信息
    """
    start_time = time.time()
    try:
        # 转换为riemann张量
        rm_A = rm.tensor(A.copy(), requires_grad=True)
        
        # 转换为torch张量
        if TORCH_AVAILABLE:
            torch_A = torch.tensor(A.copy(), requires_grad=True)
        else:
            torch_A = None
        
        # 执行Cholesky分解
        rm_L = rm_cholesky(rm_A, upper=upper)
        
        # 检查重构
        reconstruction_ok = check_cholesky_reconstruction(rm_A, rm_L, upper=upper)
        if not reconstruction_ok:
            error_msg = f"Cholesky {case_name}重构失败"
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, error_msg)
                print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                print(f"  详情: {error_msg}")
            return False, error_msg
        
        # 与PyTorch比较
        if TORCH_AVAILABLE:
            try:
                torch_L = torch_cholesky(torch_A, upper=upper)
                # 比较分解结果
                result_equal = compare_values(rm_L, torch_L)
                if not result_equal:
                    error_msg = f"Cholesky {case_name}结果与PyTorch不一致"
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, error_msg)
                        print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                        print(f"  详情: {error_msg}")
                    return False, error_msg
            except Exception as e:
                # PyTorch可能不支持某些情况，跳过比较
                pass
        
        # 计算梯度
        # Riemann梯度
        if rm_L.dtype in [np.complex64, np.complex128]:
            loss_rm = rm_L.abs().sum()
        else:
            loss_rm = rm_L.sum()
        loss_rm.backward()
        
        if TORCH_AVAILABLE:
            # PyTorch梯度
            if torch_L.dtype in [torch.complex64, torch.complex128]:
                loss_torch = torch_L.abs().sum()
            else:
                loss_torch = torch_L.sum()
            loss_torch.backward()
            
            # 比较梯度
            if hasattr(rm_A.grad, 'numpy'):
                rm_grad_np = rm_A.grad.numpy()
            else:
                rm_grad_np = rm_A.grad
            
            torch_grad_np = torch_A.grad.numpy()
            grad_close = np.allclose(rm_grad_np, torch_grad_np, rtol=1e-3, atol=1e-3)
            
            if not grad_close:
                error_msg = f"Cholesky {case_name}梯度与PyTorch不一致"
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, error_msg)
                    print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                    print(f"  详情: {error_msg}")
                return False, error_msg
        else:
            # 如果PyTorch不可用，默认梯度测试通过
            grad_close = True
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result(case_name, grad_close)
            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        
        return grad_close, None
        
    except Exception as e:
        error_msg = f"Cholesky {case_name}梯度测试错误: {str(e)}"
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result(case_name, False, error_msg)
            print(f"测试用例: {case_name} - {Colors.FAIL}失败{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
            print(f"  详情: {error_msg}")
        return False, error_msg

# 基本测试类
class TestLinalgCholeskyBasic(unittest.TestCase):
    def setUp(self):
        # 设置随机种子以确保结果可重复
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
        # 获取当前测试方法名
        self.current_test_name = self._testMethodName
        # 初始化stats_ended标志
        self.stats_ended = False
        # 如果是独立脚本运行，则开始记录函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")
                
    def tearDown(self):
        # 如果是独立脚本运行，且函数统计还没有被结束，则结束函数统计
        if IS_RUNNING_AS_SCRIPT and not getattr(self, 'stats_ended', False):
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def _test_cholesky_single_case(self, np_data, case_name, is_batch=False, upper=False):
        """统一的cholesky测试用例处理函数"""
        start_time = time.time()
        try:
            # 转换为riemann张量
            riemann_tensor = rm.tensor(np_data, requires_grad=True)
            
            # 转换为torch张量
            if TORCH_AVAILABLE:
                torch_tensor = torch.tensor(np_data, requires_grad=True)
            else:
                torch_tensor = None
            
            # 执行cholesky分解
            rm_L = rm_cholesky(riemann_tensor, upper=upper)
            
            # 检查重构
            reconstruction_ok = check_cholesky_reconstruction(riemann_tensor, rm_L, upper=upper)
            
            # 检查数据类型
            dtype_ok = rm_L.dtype == riemann_tensor.dtype
            
            # 与torch结果比较
            torch_comparison_ok = True
            if TORCH_AVAILABLE:
                try:
                    torch_L = torch_cholesky(torch_tensor, upper=upper)
                    torch_comparison_ok = compare_values(rm_L, torch_L)
                except Exception as e:
                    # PyTorch可能不支持某些情况，跳过比较
                    torch_comparison_ok = True
            
            # 汇总结果
            all_ok = reconstruction_ok and dtype_ok
            if TORCH_AVAILABLE:
                all_ok = all_ok and torch_comparison_ok
            
            # 记录结果
            if IS_RUNNING_AS_SCRIPT or not all_ok:
                details = []
                if not reconstruction_ok:
                    details.append("重构失败")
                if not dtype_ok:
                    details.append("数据类型不匹配")
                if TORCH_AVAILABLE and not torch_comparison_ok:
                    details.append("与PyTorch结果不匹配")
                
                # 添加结果并显示子用例状态
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, all_ok, " | ".join(details) if details else None)
                status = "通过" if all_ok else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if all_ok else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                if details:
                    print(f"  详情: {', '.join(details)}")
            
            return all_ok
        
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, str(e))
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise
    
    def test_real_positive_definite_matrix(self):
        """测试实对称正定矩阵的Cholesky分解"""
        # 测试不同大小的实对称正定矩阵
        sizes = [2, 3, 5, 10]
        for size in sizes:
            # 生成对称正定矩阵
            A = np.random.randn(size, size)
            A = A @ A.T + np.eye(size) * 0.1  # 确保正定
            case_name = f"实对称正定矩阵 {size}x{size}"
            self.assertTrue(self._test_cholesky_single_case(A, case_name))
    
    def test_complex_hermitian_matrix(self):
        """测试复Hermitian正定矩阵的Cholesky分解"""
        # 测试不同大小的复Hermitian正定矩阵
        sizes = [2, 3, 5]
        for size in sizes:
            # 生成Hermitian正定矩阵
            A = np.random.randn(size, size) + 1j * np.random.randn(size, size)
            A = A @ A.conj().T + np.eye(size) * 0.1  # 确保正定
            case_name = f"复Hermitian正定矩阵 {size}x{size}"
            self.assertTrue(self._test_cholesky_single_case(A, case_name))
    
    def test_batch_matrix(self):
        """测试批量矩阵的Cholesky分解"""
        # 测试批量实对称正定矩阵
        batch_sizes = [(2, 3, 3), (4, 2, 2)]
        for batch_shape in batch_sizes:
            # 生成批量对称正定矩阵
            batch_A = np.random.randn(*batch_shape)
            # 确保每个矩阵都是对称正定的
            for i in range(batch_shape[0]):
                A = batch_A[i]
                batch_A[i] = A @ A.T + np.eye(batch_shape[1]) * 0.1
            case_name = f"批量实对称正定矩阵 {batch_shape}"
            self.assertTrue(self._test_cholesky_single_case(batch_A, case_name, is_batch=True))
        
        # 测试批量复Hermitian正定矩阵
        batch_sizes = [(2, 3, 3)]
        for batch_shape in batch_sizes:
            # 生成批量Hermitian正定矩阵
            batch_A = np.random.randn(*batch_shape) + 1j * np.random.randn(*batch_shape)
            # 确保每个矩阵都是Hermitian正定的
            for i in range(batch_shape[0]):
                A = batch_A[i]
                batch_A[i] = A @ A.conj().T + np.eye(batch_shape[1]) * 0.1
            case_name = f"批量复Hermitian正定矩阵 {batch_shape}"
            self.assertTrue(self._test_cholesky_single_case(batch_A, case_name, is_batch=True))
    
    def test_upper_parameter(self):
        """测试upper参数的影响"""
        # 测试upper=True的情况
        sizes = [2, 3, 5]
        for size in sizes:
            # 生成对称正定矩阵
            A = np.random.randn(size, size)
            A = A @ A.T + np.eye(size) * 0.1
            case_name = f"upper=True 实对称正定矩阵 {size}x{size}"
            self.assertTrue(self._test_cholesky_single_case(A, case_name, upper=True))
    
    def test_exceptions(self):
        """测试各种异常情况"""
        # 测试非方阵
        non_square = np.random.randn(3, 2)
        case_name = "非方阵异常"
        with self.assertRaises(Exception):
            rm_cholesky(rm.tensor(non_square))
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result(case_name, True)
            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC}")
        
        # 测试非正定矩阵
        non_positive_definite = np.array([[1, 2], [2, 1]])  # 非正定
        case_name = "非正定矩阵异常"
        with self.assertRaises(Exception):
            rm_cholesky(rm.tensor(non_positive_definite))
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result(case_name, True)
            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC}")
        
        # 测试一维张量
        one_dimensional = np.random.randn(3)
        case_name = "一维张量异常"
        with self.assertRaises(Exception):
            rm_cholesky(rm.tensor(one_dimensional))
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result(case_name, True)
            print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC}")

# 梯度测试类
class TestLinalgCholeskyGradients(unittest.TestCase):
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
        # 如果是独立脚本运行，且函数统计还没有被结束，则结束函数统计
        if IS_RUNNING_AS_SCRIPT and not getattr(self, 'stats_ended', False):
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_cholesky_gradients_basic(self):
        """测试基本矩阵的Cholesky分解梯度"""
        # 测试配置列表
        test_configs = [
            # 实对称正定矩阵
            {'data_type': 'real', 'sizes': [2, 3, 5]},
            # 复Hermitian正定矩阵
            {'data_type': 'complex', 'sizes': [2, 3, 5]}
        ]
        
        for config in test_configs:
            data_type = config['data_type']
            sizes = config['sizes']
            
            for size in sizes:
                # 生成矩阵数据
                if data_type == 'real':
                    A = np.random.randn(size, size)
                    A = A @ A.T + np.eye(size) * 0.1  # 确保正定
                    case_name = f"实对称正定矩阵梯度 {size}x{size}"
                else:  # complex
                    A = np.random.randn(size, size) + 1j * np.random.randn(size, size)
                    A = A @ A.conj().T + np.eye(size) * 0.1  # 确保正定
                    case_name = f"复Hermitian正定矩阵梯度 {size}x{size}"
                
                # 计算梯度
                grad_ok, error_msg = gradient_for_cholesky(A, case_name, upper=False)
                
                # 记录结果
                if IS_RUNNING_AS_SCRIPT:
                    details = []
                    if error_msg:
                        details.append(error_msg)
                    stats.add_result(case_name, grad_ok, " | ".join(details) if details else None)
                
                # 验证梯度
                self.assertTrue(grad_ok, error_msg)
    
    def test_cholesky_gradients_batch(self):
        """测试批量矩阵的Cholesky分解梯度"""
        # 测试配置列表
        test_configs = [
            # 批量实对称正定矩阵
            {'data_type': 'real', 'batch_shapes': [(2, 3, 3), (4, 2, 2)]},
            # 批量复Hermitian正定矩阵
            {'data_type': 'complex', 'batch_shapes': [(2, 3, 3)]}
        ]
        
        for config in test_configs:
            data_type = config['data_type']
            batch_shapes = config['batch_shapes']
            
            for batch_shape in batch_shapes:
                # 生成批量矩阵数据
                if data_type == 'real':
                    batch_A = np.random.randn(*batch_shape)
                    # 确保每个矩阵都是对称正定的
                    for i in range(batch_shape[0]):
                        A = batch_A[i]
                        batch_A[i] = A @ A.T + np.eye(batch_shape[1]) * 0.1
                    case_name = f"批量实对称正定矩阵梯度 {batch_shape}"
                else:  # complex
                    batch_A = np.random.randn(*batch_shape) + 1j * np.random.randn(*batch_shape)
                    # 确保每个矩阵都是Hermitian正定的
                    for i in range(batch_shape[0]):
                        A = batch_A[i]
                        batch_A[i] = A @ A.conj().T + np.eye(batch_shape[1]) * 0.1
                    case_name = f"批量复Hermitian正定矩阵梯度 {batch_shape}"
                
                # 计算梯度
                grad_ok, error_msg = gradient_for_cholesky(batch_A, case_name, upper=False)
                
                # 记录结果
                if IS_RUNNING_AS_SCRIPT:
                    details = []
                    if error_msg:
                        details.append(error_msg)
                    stats.add_result(case_name, grad_ok, " | ".join(details) if details else None)
                
                # 验证梯度
                self.assertTrue(grad_ok, error_msg)
    
    def test_cholesky_gradients_upper(self):
        """测试upper=True的Cholesky分解梯度"""
        # 测试实对称正定矩阵
        sizes = [2, 3, 5]
        for size in sizes:
            A = np.random.randn(size, size)
            A = A @ A.T + np.eye(size) * 0.1  # 确保正定
            case_name = f"upper=True 实对称正定矩阵梯度 {size}x{size}"
            grad_ok, error_msg = gradient_for_cholesky(A, case_name, upper=True)
            
            # 记录结果
            if IS_RUNNING_AS_SCRIPT:
                details = []
                if error_msg:
                    details.append(error_msg)
                stats.add_result(case_name, grad_ok, " | ".join(details) if details else None)
            
            # 验证梯度
            self.assertTrue(grad_ok, error_msg)

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}开始测试 Riemann Cholesky 函数{Colors.ENDC}")
    print("="*80)
    
    # 运行基本测试
    print(f"\n{Colors.BOLD}运行基本测试{Colors.ENDC}")
    basic_suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgCholeskyBasic)
    basic_runner = unittest.TextTestRunner(verbosity=0)
    basic_result = basic_runner.run(basic_suite)
    
    # 运行梯度测试
    print(f"\n{Colors.BOLD}运行梯度测试{Colors.ENDC}")
    gradient_suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgCholeskyGradients)
    gradient_runner = unittest.TextTestRunner(verbosity=0)
    gradient_result = gradient_runner.run(gradient_suite)
    
    # 打印统计摘要
    print(f"\n{Colors.BOLD}测试统计摘要{Colors.ENDC}")
    stats.print_summary()
    
    # 确定最终结果
    all_passed = stats.passed_cases == stats.total_cases
    print(f"\n{Colors.BOLD}最终测试结果: {Colors.OKGREEN if all_passed else Colors.FAIL}{'全部通过' if all_passed else '部分失败'}{Colors.ENDC}")
    
    # 退出
    sys.exit(0 if all_passed else 1)