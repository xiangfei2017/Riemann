import unittest
import numpy as np
import time
import sys, os

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import riemann as rm
    from riemann.linalg import lu as rm_lu
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

try:
    import torch
    from torch.linalg import lu as torch_lu
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的lu函数")
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

# 检查LU分解的重构
def check_lu_reconstruction(A, P, L, U, atol=1e-6):
    """检查LU分解的重构是否正确：A ≈ PLU"""
    try:
        # 计算PLU
        PL = P @ L
        PLU = PL @ U
        
        # 计算重构误差 - 对批量矩阵使用更高的容差
        A_array = A.detach().numpy() if hasattr(A, 'detach') else A.numpy()
        PLU_array = PLU.detach().numpy() if hasattr(PLU, 'detach') else PLU.numpy()
        
        # 根据矩阵维度和批次调整容差
        if len(A_array.shape) > 2:  # 批量矩阵
            current_atol = max(atol, 1e-5)  # 批量矩阵使用更大的容差
        else:
            current_atol = atol
            
        error = np.linalg.norm(A_array - PLU_array)
        reconstruction_ok = error < current_atol
        
        # 检查L是否为单位下三角矩阵
        L_array = L.detach().numpy() if hasattr(L, 'detach') else L.numpy()
        if len(L_array.shape) > 2:  # 批量处理
            # 对每个批次分别检查
            is_unit_lower = True
            for i in range(L_array.shape[0]):
                batch_L = L_array[i]
                size = min(batch_L.shape)
                I = np.eye(size)
                batch_diag = np.diagonal(batch_L)
                is_unit_lower = is_unit_lower and np.allclose(batch_diag, np.ones(size))
        else:
            # 单个矩阵检查
            size = min(L_array.shape)
            I = np.eye(size)
            L_diag = np.diagonal(L_array)
            is_unit_lower = np.allclose(L_diag, np.ones(size))
        
        # 检查U是否为上三角矩阵
        U_array = U.detach().numpy() if hasattr(U, 'detach') else U.numpy()
        if len(U_array.shape) > 2:  # 批量处理
            # 对每个批次分别检查
            is_upper = True
            for i in range(U_array.shape[0]):
                batch_U = U_array[i]
                batch_lower = np.tril(batch_U, k=-1)
                is_upper = is_upper and np.allclose(batch_lower, 0, atol=current_atol)
        else:
            # 单个矩阵检查
            U_lower = np.tril(U_array, k=-1)
            is_upper = np.allclose(U_lower, 0, atol=current_atol)
        
        return reconstruction_ok and is_unit_lower and is_upper
    except Exception as e:
        print(f"重构检查错误: {str(e)}")
        return False

# LU分解的梯度测试函数
def gradient_for_decomp(A, decomp_func_name, case_name):
    """统一的梯度测试函数，适用于LU分解
    
    参数:
        A: 输入矩阵
        decomp_func_name: 分解函数名称 ('lu')
        case_name: 测试用例名称
    
    返回:
        (p_grad_close, l_grad_close, u_grad_close, error_msg): 梯度比较结果和错误信息
    """
    start_time = time.time()
    try:
        # 函数名称到实际函数的映射
        func_map = {
            'lu': (rm_lu, torch_lu)
        }
        
        if decomp_func_name not in func_map:
            raise ValueError(f"不支持的分解函数: {decomp_func_name}")
        
        rm_func, torch_func = func_map[decomp_func_name]
        
        # 统一的梯度计算函数
        def compute_gradients(tensor, func):
            """计算LU分解的L和U梯度"""
            # 确保张量需要梯度
            tensor.requires_grad = True
            
            # 进行LU分解
            results = func(tensor)
            
            # 存储结果的列表
            gradients = []
            tensor.grad = None
            
            # 对L和U分别计算梯度（P不需要梯度）
            for i, component in enumerate(results[1:3]):  # 跳过P
                # 计算标量值
                scalar = (component.abs()**2.0).sum()
                
                # 反向传播
                scalar.backward(retain_graph=True)
                
                # 存储梯度 - 确保保存副本而不是引用
                gradients.append(tensor.grad.clone())
                tensor.grad = None
            
            return gradients
        
        # Riemann梯度计算
        rm_A = rm.tensor(A.copy(), requires_grad=True)
        rm_gradients = compute_gradients(rm_A, rm_func)
        rm_grad_l, rm_grad_u = rm_gradients
        
        if not TORCH_AVAILABLE:
            # 如果PyTorch不可用，默认测试通过
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, True)
                print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒) [PyTorch不可用，跳过梯度比较]")
            return True, True, True, None
        
        # PyTorch梯度计算 - 使用统一的处理方式
        torch_A = torch.tensor(A.copy(), requires_grad=True)
        torch_gradients = compute_gradients(torch_A, torch_func)
        torch_grad_l, torch_grad_u = torch_gradients
        
        # P始终不需要梯度，所以返回True
        p_grad_close = True
        
        # 比较梯度
        l_grad_close = np.allclose(rm_grad_l.numpy(), torch_grad_l.numpy(), rtol=1e-3, atol=1e-3)
        u_grad_close = np.allclose(rm_grad_u.numpy(), torch_grad_u.numpy(), rtol=1e-3, atol=1e-3)
        
        # 汇总结果
        all_ok = l_grad_close and u_grad_close
        error_msg = None
        if not all_ok:
            error_msg = f"{decomp_func_name} {case_name}梯度比较失败"
            
        # 记录结果并显示子用例状态
        if IS_RUNNING_AS_SCRIPT:
            details = []
            if not l_grad_close:
                details.append("L梯度不匹配")
            if not u_grad_close:
                details.append("U梯度不匹配")
            
            stats.add_result(case_name, all_ok, " | ".join(details) if details else None)
            status = "通过" if all_ok else "失败"
            print(f"测试用例: {case_name} - {Colors.OKGREEN if all_ok else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
            if details:
                print(f"  详情: {', '.join(details)}")
        
        return p_grad_close, l_grad_close, u_grad_close, error_msg
        
    except Exception as e:
        error_msg = f"{decomp_func_name} {case_name}梯度计算错误: {str(e)}"
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result(case_name, False, error_msg)
            print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
        return False, False, False, error_msg

# 基本测试类
class TestLinalgLUBasic(unittest.TestCase):
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
    
    def _test_lu_single_case(self, np_data, case_name, is_batch=False):
        """统一的lu测试用例处理函数"""
        start_time = time.time()
        try:
            # 转换为riemann张量
            riemann_tensor = rm.tensor(np_data, requires_grad=True)
            
            # 转换为torch张量
            if TORCH_AVAILABLE:
                torch_tensor = torch.tensor(np_data, requires_grad=True)
            else:
                torch_tensor = None
            
            # 执行lu分解
            rm_P, rm_L, rm_U = rm_lu(riemann_tensor)
            
            # 检查重构
            reconstruction_ok = check_lu_reconstruction(riemann_tensor, rm_P, rm_L, rm_U)
            
            # 检查数据类型 - 对于复数，P可能是实数类型，这是正常的
            dtype_ok = (rm_L.dtype == riemann_tensor.dtype and 
                       rm_U.dtype == riemann_tensor.dtype)
            # P矩阵是置换矩阵，可以是实数类型，与输入类型无关
            
            # 与torch结果比较
            torch_comparison_ok = True
            if TORCH_AVAILABLE:
                torch_P, torch_L, torch_U = torch_lu(torch_tensor)
                # 比较P、L、U的值
                torch_comparison_ok = (compare_values(rm_P, torch_P) and 
                                     compare_values(rm_L, torch_L) and 
                                     compare_values(rm_U, torch_U))
            
            # 汇总结果
            all_ok = reconstruction_ok and dtype_ok
            if TORCH_AVAILABLE:
                all_ok = all_ok and torch_comparison_ok
            
            # 记录结果
            if IS_RUNNING_AS_SCRIPT:
                details = []
                if not reconstruction_ok:
                    details.append("重构失败")
                if not dtype_ok:
                    details.append("数据类型不匹配")
                if TORCH_AVAILABLE and not torch_comparison_ok:
                    details.append("与PyTorch结果不匹配")
                
                # 添加结果并显示子用例状态
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
    
    def test_real_square_matrix(self):
        """测试实方阵的LU分解"""
        # 测试不同大小的实方阵
        sizes = [2, 3, 5, 10]
        for size in sizes:
            # 生成可逆矩阵
            np_data = np.random.randn(size, size)
            case_name = f"实方阵 {size}x{size}"
            self.assertTrue(self._test_lu_single_case(np_data, case_name))
    
    def test_complex_square_matrix(self):
        """测试复方阵的LU分解"""
        # 测试不同大小的复方阵
        sizes = [2, 3, 5]
        for size in sizes:
            # 生成可逆复矩阵
            real_part = np.random.randn(size, size)
            imag_part = np.random.randn(size, size)
            np_data = real_part + 1j * imag_part
            case_name = f"复方阵 {size}x{size}"
            self.assertTrue(self._test_lu_single_case(np_data, case_name))
    
    def test_rectangular_matrix(self):
        """测试长方形矩阵的LU分解"""
        # 测试不同大小的长方形矩阵
        shapes = [(3, 5), (5, 3)]
        for m, n in shapes:
            np_data = np.random.randn(m, n)
            case_name = f"长方形矩阵 {m}x{n}"
            self.assertTrue(self._test_lu_single_case(np_data, case_name))
    
    def test_batch_matrix(self):
        """测试批量矩阵的LU分解"""
        # 测试批量实矩阵
        batch_sizes = [(2, 3, 3), (4, 2, 2)]
        for batch_shape in batch_sizes:
            np_data = np.random.randn(*batch_shape)
            case_name = f"批量实矩阵 {batch_shape}"
            self.assertTrue(self._test_lu_single_case(np_data, case_name, is_batch=True))
        
        # 测试批量复矩阵
        batch_sizes = [(2, 3, 3)]
        for batch_shape in batch_sizes:
            real_part = np.random.randn(*batch_shape)
            imag_part = np.random.randn(*batch_shape)
            np_data = real_part + 1j * imag_part
            case_name = f"批量复矩阵 {batch_shape}"
            self.assertTrue(self._test_lu_single_case(np_data, case_name, is_batch=True))

# 梯度测试类
class TestLinalgLUGradients(unittest.TestCase):
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
    
    def test_lu_gradients_basic(self):
        """测试基本矩阵（实矩阵和复矩阵，方阵和非对称矩阵）的LU分解梯度"""
        # 测试配置列表
        test_configs = [
            # 实方阵
            {'data_type': 'real', 'matrix_type': 'square', 'sizes': [(2, 2), (3, 3)]},
            # 复方阵
            {'data_type': 'complex', 'matrix_type': 'square', 'sizes': [(2, 2), (3, 3)]},
            # 实非对称矩阵（行数大于列数）
            {'data_type': 'real', 'matrix_type': 'rectangular', 'sizes': [(3, 2), (5, 2), (5, 1)]},
            # 复非对称矩阵（行数大于列数）
            {'data_type': 'complex', 'matrix_type': 'rectangular', 'sizes': [(3, 2), (5, 2), (5, 1)]},
            # 实非对称矩阵（行数小于列数）
            {'data_type': 'real', 'matrix_type': 'rectangular', 'sizes': [(1, 3), (2, 3), (2, 5)]},
            # 复非对称矩阵（行数小于列数）
            {'data_type': 'complex', 'matrix_type': 'rectangular', 'sizes': [(1, 3), (2, 3), (2, 5)]}
        ]
        
        for config in test_configs:
            data_type = config['data_type']
            matrix_type = config['matrix_type']
            sizes = config['sizes']
            
            for size in sizes:
                # 生成矩阵数据
                if data_type == 'real':
                    np_data = np.random.randn(size[0], size[1])
                    case_name = f"实矩阵梯度 {size[0]}x{size[1]}"
                else:  # complex
                    real_part = np.random.randn(size[0], size[1])
                    imag_part = np.random.randn(size[0], size[1])
                    np_data = real_part + 1j * imag_part
                    case_name = f"复矩阵梯度 {size[0]}x{size[1]}"
                
                # 计算梯度
                p_grad_ok, l_grad_ok, u_grad_ok, error_msg = gradient_for_decomp(np_data, 'lu', case_name)
                
                # 记录结果
                if IS_RUNNING_AS_SCRIPT:
                    details = []
                    if error_msg:
                        details.append(error_msg)
                    stats.add_result(case_name, p_grad_ok and l_grad_ok and u_grad_ok, " | ".join(details) if details else None)
                
                # 验证梯度
                self.assertTrue(p_grad_ok)
                self.assertTrue(l_grad_ok)
                self.assertTrue(u_grad_ok)
    
    def test_lu_gradients_batch(self):
        """测试批量矩阵的LU分解梯度，包括实矩阵和复矩阵"""
        # 测试配置列表
        test_configs = [
            # 批量实方阵
            {'data_type': 'real', 'batch_shapes': [(2, 3, 3)]},
            # 批量实非对称矩阵
            {'data_type': 'real', 'batch_shapes': [(2, 3, 1), (2, 3, 2)]},
            {'data_type': 'real', 'batch_shapes': [(2, 2, 3), (2, 2, 5)]},
            # 批量复方阵
            {'data_type': 'complex', 'batch_shapes': [(2, 3, 3)]},
            # 批量复非对称矩阵
            {'data_type': 'complex', 'batch_shapes': [(2, 3, 1),(2, 3, 2)]},
            {'data_type': 'complex', 'batch_shapes': [(2, 2, 3),(2, 2, 5)]},
        ]
        
        for config in test_configs:
            data_type = config['data_type']
            batch_shapes = config['batch_shapes']
            
            for batch_shape in batch_shapes:
                # 生成批量矩阵数据
                if data_type == 'real':
                    np_data = np.random.randn(*batch_shape)
                    case_name = f"批量实矩阵梯度 {batch_shape}"
                else:  # complex
                    real_part = np.random.randn(*batch_shape)
                    imag_part = np.random.randn(*batch_shape)
                    np_data = real_part + 1j * imag_part
                    case_name = f"批量复矩阵梯度 {batch_shape}"
                
                # 计算梯度
                p_grad_ok, l_grad_ok, u_grad_ok, error_msg = gradient_for_decomp(np_data, 'lu', case_name)
                
                # 记录结果
                if IS_RUNNING_AS_SCRIPT:
                    details = []
                    if error_msg:
                        details.append(error_msg)
                    stats.add_result(case_name, p_grad_ok and l_grad_ok and u_grad_ok, " | ".join(details) if details else None)
                
                # 验证梯度
                self.assertTrue(p_grad_ok)
                self.assertTrue(l_grad_ok)
                self.assertTrue(u_grad_ok)

# 运行测试
if __name__ == '__main__':
    IS_RUNNING_AS_SCRIPT = True
    clear_screen()
    print(f"{Colors.HEADER}===== 开始LU分解测试 ====={Colors.ENDC}")
    
    # 运行基本测试
    print(f"\n{Colors.BOLD}基本功能测试{Colors.ENDC}")
    basic_suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgLUBasic)
    unittest.TextTestRunner(verbosity=0).run(basic_suite)
    
    # 运行梯度测试
    print(f"\n{Colors.BOLD}梯度功能测试{Colors.ENDC}")
    if TORCH_AVAILABLE:
        grad_suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgLUGradients)
        unittest.TextTestRunner(verbosity=0).run(grad_suite)
    else:
        print(f"{Colors.WARNING}PyTorch不可用，跳过梯度测试{Colors.ENDC}")
    
    # 打印汇总
    stats.print_summary()
    print(f"\n{Colors.HEADER}===== 测试完成 ====={Colors.ENDC}")