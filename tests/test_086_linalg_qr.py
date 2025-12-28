import unittest
import numpy as np
import time
import sys, os

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import riemann as rm
    from riemann.linalg import qr as rm_qr
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

try:
    import torch
    from torch.linalg import qr as torch_qr
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的qr函数")
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

# 检查QR分解的重构
def check_qr_reconstruction(A, Q, R, atol=1e-6):
    """
    检查QR分解的正确性
    
    参数:
    - A: 原始矩阵
    - Q: 分解后的正交/酉矩阵
    - R: 分解后的上三角矩阵
    - atol: 绝对误差容限
    
    返回:
    - tuple: (重构是否成功, 错误消息)
    """
    try:
        # 首先检查R是否为上三角矩阵（所有模式都需要检查）
        R_array = R.detach().numpy() if hasattr(R, 'detach') else R.numpy()
        
        # 检查上三角性质（统一处理批量和单个矩阵）
        is_upper = True
        if len(R_array.shape) > 2:  # 批量处理
            for i in range(R_array.shape[0]):
                batch_lower = np.tril(R_array[i], k=-1)
                is_upper = is_upper and np.allclose(batch_lower, 0, atol=atol)
        else:  # 单个矩阵
            R_lower = np.tril(R_array, k=-1)
            is_upper = np.allclose(R_lower, 0, atol=atol)
        
        # 检查Q是否为0维张量（mode='r'的情况）
        if not hasattr(Q, 'shape') or len(Q.shape) == 0:
            # mode='r'时，只需要检查R是上三角矩阵
            return is_upper, (None if is_upper else "R矩阵不是上三角矩阵")
        
        # 对于其他mode，执行完整检查
        # 计算QR
        QR = Q @ R
        
        # 计算重构误差
        A_array = A.detach().numpy() if hasattr(A, 'detach') else A.numpy()
        QR_array = QR.detach().numpy() if hasattr(QR, 'detach') else QR.numpy()
        
        # 对于批量矩阵使用更高的容差
        current_atol = max(atol, 1e-5) if len(A_array.shape) > 2 else atol
        
        error = np.linalg.norm(A_array - QR_array)
        reconstruction_ok = error < current_atol
        
        # 检查Q是否为正交/酉矩阵
        is_orthonormal = True
        
        # 定义检查单个矩阵正交性的辅助函数
        def check_single_orthonormal(q_array, atol_val):
            # 计算 Q^H * Q
            if np.iscomplexobj(q_array):
                qtq = q_array.conj().T @ q_array
            else:
                qtq = q_array.T @ q_array
            # 与单位矩阵比较
            return np.allclose(qtq, np.eye(qtq.shape[0]), atol=atol_val)
        
        # 批量处理或单个矩阵处理
        if len(QR_array.shape) > 2:  # 批量处理
            for i in range(QR_array.shape[0]):
                batch_Q = Q[i].detach().numpy() if hasattr(Q[i], 'detach') else Q[i].numpy()
                is_orthonormal = is_orthonormal and check_single_orthonormal(batch_Q, current_atol)
        else:  # 单个矩阵
            Q_array = Q.detach().numpy() if hasattr(Q, 'detach') else Q.numpy()
            is_orthonormal = check_single_orthonormal(Q_array, current_atol)
        
        # 综合所有检查结果
        all_ok = reconstruction_ok and is_orthonormal and is_upper
        
        if not all_ok:
            error_msg = []
            if not reconstruction_ok:
                error_msg.append(f"重构失败，误差: {error:.8f}")
            if not is_orthonormal:
                error_msg.append("Q矩阵不是正交/酉矩阵")
            if not is_upper:
                error_msg.append("R矩阵不是上三角矩阵")
            return False, ", ".join(error_msg)
        
        return True, None
    except Exception as e:
        return False, f"重构检查错误: {str(e)}"

# QR分解的梯度测试函数
def gradient_for_decomp(A, decomp_func_name, case_name, mode='reduced'):
    """统一的梯度测试函数，适用于QR分解
    
    参数:
        A: 输入矩阵
        decomp_func_name: 分解函数名称 ('qr')
        case_name: 测试用例名称
        mode: QR分解的mode参数
    
    返回:
        (q_grad_close, r_grad_close, error_msg): 梯度比较结果和错误信息
    """
    start_time = time.time()
    try:
        # 特殊情况处理：mode='r'时不支持梯度跟踪
        if mode == 'r':
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, True)
                print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒) [mode='r'不支持梯度跟踪]")
            return True, True, None
        
        # 特殊情况处理：mode='complete'且m>n时torch不支持梯度跟踪
        # 对于批量矩阵，需要检查最后两个维度
        if mode == 'complete':
            # 获取矩阵维度（处理批量情况）
            if len(A.shape) > 2:  # 批量矩阵
                m, n = A.shape[-2], A.shape[-1]
            else:
                m, n = A.shape[0], A.shape[1]
            
            if m > n:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, True)
                    print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒) [mode='complete'且m>n时不支持梯度跟踪]")
                return True, True, None
        
        # 函数名称到实际函数的映射
        func_map = {
            'qr': (rm_qr, torch_qr)
        }
        
        if decomp_func_name not in func_map:
            raise ValueError(f"不支持的分解函数: {decomp_func_name}")
        
        rm_func, torch_func = func_map[decomp_func_name]
        
        # 统一的梯度计算函数
        def compute_gradients(tensor, func):
            """计算QR分解的Q和R梯度"""
            # 确保张量需要梯度
            tensor.requires_grad = True
            
            # 进行QR分解
            results = func(tensor, mode=mode)
            
            # 存储结果的列表
            gradients = []
            
            # 对Q和R分别计算梯度
            for component in results:
                # 重置梯度
                if tensor.grad is not None:
                    tensor.grad.zero_()
                else:
                    tensor.grad = None
                
                # 对于空张量或标量，跳过梯度计算
                if component is None or (hasattr(component, 'shape') and len(component.shape) == 0):
                    gradients.append(None)
                    continue
                
                # 计算标量值 - 确保正确处理批量矩阵
                scalar = (component.abs()**2.0).sum()
                
                # 反向传播 - 对于最后一个组件，不需要retain_graph
                retain_graph = (component is not results[-1])
                scalar.backward(retain_graph=retain_graph)
                
                # 存储梯度 - 确保保存副本而不是引用
                if tensor.grad is not None:
                    gradients.append(tensor.grad.clone())
                else:
                    gradients.append(None)
            
            return gradients
        
        # Riemann梯度计算
        rm_A = rm.tensor(A.copy(), requires_grad=True)
        rm_gradients = compute_gradients(rm_A, rm_func)
        
        # 确保我们有梯度结果
        if len(rm_gradients) < 2:
            raise ValueError("梯度计算结果不足")
            
        rm_grad_q, rm_grad_r = rm_gradients[0], rm_gradients[1]
        
        if not TORCH_AVAILABLE:
            # 如果PyTorch不可用，默认测试通过
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, True)
                print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒) [PyTorch不可用，跳过梯度比较]")
            return True, True, None
        
        # PyTorch梯度计算
        torch_A = torch.tensor(A.copy(), requires_grad=True)
        torch_gradients = compute_gradients(torch_A, torch_func)
        
        if len(torch_gradients) < 2:
            raise ValueError("PyTorch梯度计算结果不足")
            
        torch_grad_q, torch_grad_r = torch_gradients[0], torch_gradients[1]
        
        # 定义梯度比较辅助函数，减少重复代码
        def compare_gradients(rm_grad, torch_grad, component_name):
            """比较两个梯度张量是否接近"""
            if rm_grad is None or torch_grad is None:
                return True
                
            rm_grad_np = rm_grad.numpy()
            torch_grad_np = torch_grad.numpy()
            
            # 检查形状是否匹配
            if rm_grad_np.shape != torch_grad_np.shape:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  {component_name}梯度形状不匹配: Riemann={rm_grad_np.shape}, PyTorch={torch_grad_np.shape}")
                return False
            
            # 对于批量矩阵，适当放宽容差
            is_batch = len(rm_grad_np.shape) > 2
            rtol = 1e-2 if is_batch else 1e-3
            atol = 1e-2 if is_batch else 1e-3
            
            grad_close = np.allclose(rm_grad_np, torch_grad_np, rtol=rtol, atol=atol)
            
            # 如果不匹配，打印一些统计信息帮助调试（仅对Q梯度）
            if not grad_close and IS_RUNNING_AS_SCRIPT and component_name == "Q":
                diff = np.abs(rm_grad_np - torch_grad_np)
                print(f"  {component_name}梯度不匹配详情:")
                print(f"    最大差异: {np.max(diff):.8f}")
                print(f"    平均差异: {np.mean(diff):.8f}")
                print(f"    梯度范围 - Riemann: [{np.min(rm_grad_np):.8f}, {np.max(rm_grad_np):.8f}]")
                print(f"    梯度范围 - PyTorch: [{np.min(torch_grad_np):.8f}, {np.max(torch_grad_np):.8f}]")
            
            return grad_close
        
        # 比较梯度
        q_grad_close = compare_gradients(rm_grad_q, torch_grad_q, "Q")
        r_grad_close = compare_gradients(rm_grad_r, torch_grad_r, "R")
        
        # 汇总结果
        all_ok = q_grad_close and r_grad_close
        error_msg = None
        if not all_ok:
            error_msg = f"{decomp_func_name} {case_name}梯度比较失败"
            
        # 记录结果并显示子用例状态
        if IS_RUNNING_AS_SCRIPT:
            details = []
            if not q_grad_close:
                details.append("Q梯度不匹配")
            if not r_grad_close:
                details.append("R梯度不匹配")
            
            stats.add_result(case_name, all_ok, " | ".join(details) if details else None)
            status = "通过" if all_ok else "失败"
            print(f"测试用例: {case_name} - {Colors.OKGREEN if all_ok else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
            if details:
                print(f"  详情: {', '.join(details)}")
        
        return q_grad_close, r_grad_close, error_msg
        
    except Exception as e:
        error_msg = f"{decomp_func_name} {case_name}梯度计算错误: {str(e)}"
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result(case_name, False, error_msg)
            print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
        return False, False, error_msg

# 基本测试类
class TestLinalgQRBasic(unittest.TestCase):
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
    
    def _test_qr_single_case(self, np_data, case_name, mode='reduced', is_batch=False):
        """统一的qr测试用例处理函数"""
        start_time = time.time()
        try:
            # 转换为riemann张量
            riemann_tensor = rm.tensor(np_data, requires_grad=True)
            
            # 转换为torch张量
            if TORCH_AVAILABLE:
                torch_tensor = torch.tensor(np_data, requires_grad=True)
            else:
                torch_tensor = None
            
            # 执行qr分解
            rm_Q, rm_R = rm_qr(riemann_tensor, mode=mode)
            
            # 检查重构
            reconstruction_ok = check_qr_reconstruction(riemann_tensor, rm_Q, rm_R)
            
            # 检查数据类型
            # 对于mode='r'，Q是0维张量，不需要检查其数据类型
            if hasattr(rm_Q, 'shape') and len(rm_Q.shape) == 0:
                dtype_ok = True
            else:
                # 确保Q不是0维张量再检查数据类型
                if rm_Q is not None:
                    dtype_ok = (rm_Q.dtype == riemann_tensor.dtype)
                # 检查R的数据类型
                if rm_R is not None and not (hasattr(rm_R, 'shape') and (len(rm_R.shape) == 0 or rm_R.shape == ())):
                    dtype_ok = dtype_ok and (rm_R.dtype == riemann_tensor.dtype)
            
            # 与torch结果比较
            torch_comparison_ok = True
            if TORCH_AVAILABLE:
                try:
                    torch_Q, torch_R = torch_qr(torch_tensor, mode=mode)
                    # 比较Q和R的值
                    # 对于mode='r'，Q为空张量，只需比较R
                    if mode == 'r':
                        torch_comparison_ok = compare_values(rm_R, torch_R)
                    else:
                        torch_comparison_ok = (compare_values(rm_Q, torch_Q) and 
                                             compare_values(rm_R, torch_R))
                except Exception as e:
                    print(f"PyTorch QR分解错误: {str(e)}")
                    torch_comparison_ok = False
            
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
    
    def _run_basic_test(self, data_generator, sizes, modes, prefix, is_batch=False):
        """统一的基本测试运行函数，减少代码重复"""
        for mode in modes:
            for size in sizes:
                # 生成数据
                if isinstance(size, int):  # 方阵的特殊情况
                    np_data = data_generator(size, size)
                    case_name = f"{prefix} {size}x{size} (mode={mode})"
                else:  # 普通情况（长方形或批量矩阵）
                    np_data = data_generator(*size)
                    case_name = f"{prefix} {size} (mode={mode})"
                
                self.assertTrue(self._test_qr_single_case(np_data, case_name, mode=mode, is_batch=is_batch))
    
    def test_real_square_matrix(self):
        """测试实方阵的QR分解"""
        modes = ['reduced', 'complete', 'r']
        sizes = [2, 3, 5]
        self._run_basic_test(np.random.randn, sizes, modes, "实方阵")
    
    def test_complex_square_matrix(self):
        """测试复方阵的QR分解"""
        modes = ['reduced', 'complete', 'r']
        sizes = [2, 3, 5]
        
        # 定义复数矩阵生成器
        def complex_matrix_generator(m, n):
            real_part = np.random.randn(m, n)
            imag_part = np.random.randn(m, n)
            return real_part + 1j * imag_part
        
        self._run_basic_test(complex_matrix_generator, sizes, modes, "复方阵")
    
    def test_rectangular_matrix(self):
        """测试长方形矩阵的QR分解"""
        modes = ['reduced', 'complete', 'r']
        shapes = [(3, 5), (5, 3)]  # 不同的长方形矩阵
        self._run_basic_test(np.random.randn, shapes, modes, "长方形矩阵")
    
    def test_batch_matrix(self):
        """测试批量矩阵的QR分解"""
        modes = ['reduced', 'complete', 'r']
        
        # 测试批量实矩阵
        batch_sizes = [(2, 3, 3), (4, 2, 2), (2, 3, 5), (2, 5, 3)]
        self._run_basic_test(np.random.randn, batch_sizes, modes, "批量实矩阵", is_batch=True)
        
        # 测试批量复矩阵
        batch_sizes = [(2, 3, 3), (2, 3, 5)]
        self._run_basic_test(
            lambda *shape: np.random.randn(*shape) + 1j * np.random.randn(*shape),
            batch_sizes, modes, "批量复矩阵", is_batch=True
        )

# 梯度测试类
class TestLinalgQRGradients(unittest.TestCase):
    # 复用setUp和tearDown方法，与TestLinalgQRBasic相同
    setUp = TestLinalgQRBasic.setUp
    tearDown = TestLinalgQRBasic.tearDown
    
    def _run_gradient_test(self, data_generator, sizes, modes, prefix):
        """统一的梯度测试运行函数，减少代码重复"""
        for mode in modes:
            for size in sizes:
                # 生成数据
                if isinstance(size, tuple) and len(size) == 2:  # 单个矩阵
                    np_data = data_generator(*size)
                    case_name = f"{prefix} {size[0]}x{size[1]} (mode={mode})"
                else:  # 批量矩阵
                    np_data = data_generator(*size)
                    case_name = f"{prefix} {size} (mode={mode})"
                
                # 计算梯度
                q_grad_ok, r_grad_ok, error_msg = gradient_for_decomp(np_data, 'qr', case_name, mode=mode)
                
                # 记录结果
                if IS_RUNNING_AS_SCRIPT:
                    details = [error_msg] if error_msg else []
                    stats.add_result(case_name, q_grad_ok and r_grad_ok, " | ".join(details) if details else None)
                
                # 验证梯度
                self.assertTrue(q_grad_ok)
                self.assertTrue(r_grad_ok)
    
    def test_qr_gradients_real(self):
        """测试实矩阵的QR分解梯度"""
        # 测试不同大小的实矩阵和不同的mode
        modes = ['reduced', 'complete']
        sizes = [(2, 2), (2, 3), (3, 2), (5, 5)]  # 方阵和长方形矩阵
        
        self._run_gradient_test(np.random.randn, sizes, modes, "实矩阵梯度")
        
        # 测试mode='r'的情况（不支持梯度跟踪）
        np_data = np.random.randn(3, 3)
        case_name = f"实矩阵梯度 3x3 (mode=r)"
        q_grad_ok, r_grad_ok, _ = gradient_for_decomp(np_data, 'qr', case_name, mode='r')
        self.assertTrue(q_grad_ok)
        self.assertTrue(r_grad_ok)
    
    def test_qr_gradients_complex(self):
        """测试复矩阵的QR分解梯度"""
        # 测试不同大小的复矩阵和不同的mode
        modes = ['reduced', 'complete']
        sizes = [(2, 2), (3, 3), (2, 3), (3, 2)]
        
        # 定义复数矩阵生成器
        def complex_matrix_generator(m, n):
            real_part = np.random.randn(m, n)
            imag_part = np.random.randn(m, n)
            return real_part + 1j * imag_part
        
        self._run_gradient_test(complex_matrix_generator, sizes, modes, "复矩阵梯度")
    
    def test_qr_gradients_batch(self):
        """测试批量矩阵的QR分解梯度"""
        # 测试批量实矩阵
        modes = ['reduced', 'complete']
        batch_shapes = [(2, 3, 3), (2, 2, 3), (2, 3, 2)]
        
        self._run_gradient_test(np.random.randn, batch_shapes, modes, "批量实矩阵梯度")

# 运行测试
if __name__ == '__main__':
    IS_RUNNING_AS_SCRIPT = True
    clear_screen()
    print(f"{Colors.HEADER}===== 开始QR分解测试 ====={Colors.ENDC}")
    
    # 运行基本测试
    print(f"\n{Colors.BOLD}基本功能测试{Colors.ENDC}")
    basic_suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgQRBasic)
    unittest.TextTestRunner(verbosity=0).run(basic_suite)
    
    # 运行梯度测试
    print(f"\n{Colors.BOLD}梯度功能测试{Colors.ENDC}")
    if TORCH_AVAILABLE:
        grad_suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgQRGradients)
        unittest.TextTestRunner(verbosity=0).run(grad_suite)
    else:
        print(f"{Colors.WARNING}PyTorch不可用，跳过梯度测试{Colors.ENDC}")
    
    # 打印汇总
    stats.print_summary()
    print(f"\n{Colors.HEADER}===== 测试完成 ====={Colors.ENDC}")