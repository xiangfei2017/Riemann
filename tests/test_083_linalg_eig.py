import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.linalg import eig as rm_eig
    from riemann.linalg import eigh as rm_eigh
    # 检查CUDA可用性
    CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
    if CUDA_AVAILABLE:
        cp = rm.cuda.cp
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    from torch.linalg import eig as torch_eig
    from torch.linalg import eigh as torch_eigh
    TORCH_AVAILABLE = True
    # 检查PyTorch CUDA可用性
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的eig和eigh函数")
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

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
    """比较Riemann和PyTorch的eig/eigh结果是否接近"""
    # 处理None值的情况
    if not TORCH_AVAILABLE:
        # 如果没有PyTorch，只检查riemann结果是否存在
        return rm_result is not None
    
    if rm_result is None and torch_result is None:
        return True
    if rm_result is None or torch_result is None:
        return False
    
    # 处理元组的情况
    if isinstance(rm_result, tuple) and isinstance(torch_result, tuple):
        if len(rm_result) != len(torch_result):
            return False
        
        all_passed = True
        for i, (r, t) in enumerate(zip(rm_result, torch_result)):
            try:
                # 获取Riemann数据，处理CUDA情况
                if hasattr(r.data, 'get'):
                    # 如果是CUDA张量，先移动到CPU
                    r_data = r.data.get()
                else:
                    r_data = r.data
                
                # 获取PyTorch张量的numpy数据，处理可能的复数格式和CUDA情况
                if hasattr(t, 'detach'):
                    if hasattr(t, 'is_cuda') and t.is_cuda:
                        # 如果是CUDA张量，先移动到CPU
                        t_numpy = t.detach().cpu().numpy()
                    else:
                        t_numpy = t.detach().numpy()
                else:
                    t_numpy = t
                
                # 调试信息
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  比较元素 {i}: Riemann形状 {r.shape}, PyTorch形状 {t_numpy.shape}")
                    print(f"    Riemann数据类型: {r_data.dtype}, PyTorch数据类型: {t_numpy.dtype}")
                
                # 检查是否是特征向量（二维方阵）
                is_eigenvector = len(r_data.shape) >= 2 and r_data.shape[-2] == r_data.shape[-1]
                
                # 对于复数结果，比较实部和虚部
                if np.iscomplexobj(r_data) or np.iscomplexobj(t_numpy):
                    r_real = r_data.real if np.iscomplexobj(r_data) else r_data
                    r_imag = r_data.imag if np.iscomplexobj(r_data) else np.zeros_like(r_data)
                    t_real = t_numpy.real if np.iscomplexobj(t_numpy) else t_numpy
                    t_imag = t_numpy.imag if np.iscomplexobj(t_numpy) else np.zeros_like(t_numpy)
                    
                    if is_eigenvector:
                        # 特征向量矩阵，比较绝对值来处理符号不确定性
                        np.testing.assert_allclose(np.abs(r_real), np.abs(t_real), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(np.abs(r_imag), np.abs(t_imag), rtol=rtol, atol=atol)
                    else:
                        # 特征值，直接比较
                        np.testing.assert_allclose(r_real, t_real, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(r_imag, t_imag, rtol=rtol, atol=atol)
                else:
                    # 对于实数结果，如果是特征向量（二维数组），使用绝对值比较处理符号不确定性
                    if is_eigenvector:
                        # 特征向量矩阵，比较绝对值
                        np.testing.assert_allclose(np.abs(r_data), np.abs(t_numpy), rtol=rtol, atol=atol)
                    else:
                        # 特征值或其他数据，直接比较
                        np.testing.assert_allclose(r_data, t_numpy, rtol=rtol, atol=atol)
            except AssertionError as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  比较失败: {e}")
                    print(f"    Riemann数据: {r_data}")
                    print(f"    PyTorch数据: {t_numpy}")
                all_passed = False
                break
        
        return all_passed
    
    return False

# 检查特征分解重构误差
def check_eigen_reconstruction(A, w, V, atol=1e-6):
    """检查使用特征分解结果重构原始矩阵的误差"""
    try:
        # 获取数据，处理CUDA情况
        if hasattr(A.data, 'get'):
            A_data = A.data.get()
            V_data = V.data.get()
            w_data = w.data.get()
        else:
            A_data = A.data
            V_data = V.data
            w_data = w.data
        
        # 计算 A @ V
        AV = A_data @ V_data
        
        # 计算 V @ diag(w)
        V_diag_w = V_data @ np.diag(w_data)
        
        # 计算重构误差
        error = np.linalg.norm(AV - V_diag_w)
        return error < atol
    except:
        return False

# 检查特征分解重构误差（对于eigh）
def check_eigh_reconstruction(A, w, V, atol=1e-6):
    """检查使用特征分解结果重构原始矩阵的误差（对于Hermitian矩阵）"""
    try:
        # 获取数据，处理CUDA情况
        if hasattr(A.data, 'get'):
            A_data = A.data.get()
            V_data = V.data.get()
            w_data = w.data.get()
            # 对于CUDA计算，使用更大的误差阈值
            current_atol = 1e-4
        else:
            A_data = A.data
            V_data = V.data
            w_data = w.data
            current_atol = atol
        
        # 计算 A @ V
        AV = A_data @ V_data
        
        # 计算 V @ diag(w)
        V_diag_w = V_data @ np.diag(w_data)
        
        # 计算重构误差
        error = np.linalg.norm(AV - V_diag_w)
        
        # 检查正交性：V^H @ V = I
        VhV = V_data.T.conj() @ V_data
        identity = np.eye(V.shape[-1])
        orthogonality_error = np.linalg.norm(VhV - identity)
        
        return error < current_atol and orthogonality_error < current_atol
    except:
        return False

def gradient_for_decomp(A, decomp_func_name, case_name, device='cpu'):
    """统一的梯度测试函数，适用于eig和eigh分解
    
    参数:
        A: 输入矩阵
        decomp_func_name: 分解函数名称 ('eig' 或 'eigh')
        case_name: 测试用例名称
        device: 设备名称 ('cpu' 或 'cuda')
    
    返回:
        (w_grad_close, v_grad_close, error_msg): 梯度比较结果和错误信息
    """
    try:
        # 函数名称到实际函数的映射
        func_map = {
            'eig': (rm_eig, torch_eig),
            'eigh': (rm_eigh, torch_eigh)
        }
        
        if decomp_func_name not in func_map:
            raise ValueError(f"不支持的分解函数: {decomp_func_name}")
        
        rm_func, torch_func = func_map[decomp_func_name]
        
        # 统一的梯度计算函数
        def compute_gradients(tensor, func):
            """计算特征值分解的w和V梯度"""
            # 确保张量需要梯度
            tensor.requires_grad = True
            
            # 进行特征值分解
            w, V = func(tensor)
            
            # 存储结果的列表
            gradients = []
            tensor.grad = None
            
            # 对w和V分别计算梯度
            for component in [w, V]:
                # 计算标量值
                scalar = (component.abs()**2.0).sum()
                
                # 反向传播
                scalar.backward(retain_graph=True)
                
                # 存储梯度 - 确保保存副本而不是引用
                gradients.append(tensor.grad.clone())
                tensor.grad = None
                
            return tuple(gradients)
        
        # Riemann梯度计算
        rm_A = rm.tensor(A.copy(), requires_grad=True, device=device)
        rm_grad_w, rm_grad_V = compute_gradients(rm_A, rm_func)
        
        if not TORCH_AVAILABLE:
            return True, True, None
        
        # PyTorch梯度计算 - 使用统一的处理方式
        torch_device = device if TORCH_CUDA_AVAILABLE else 'cpu'
        torch_A = torch.tensor(A.copy(), requires_grad=True, device=torch_device)
        torch_grad_w, torch_grad_V = compute_gradients(torch_A, torch_func)
        
        # 比较梯度，处理CUDA情况
        if hasattr(rm_grad_w, 'get'):
            # 如果是CUDA张量，先移动到CPU
            rm_grad_w_np = rm_grad_w.get()
            rm_grad_V_np = rm_grad_V.get()
        else:
            rm_grad_w_np = rm_grad_w.numpy()
            rm_grad_V_np = rm_grad_V.numpy()
        
        if hasattr(torch_grad_w, 'is_cuda') and torch_grad_w.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            torch_grad_w_np = torch_grad_w.cpu().numpy()
            torch_grad_V_np = torch_grad_V.cpu().numpy()
        else:
            torch_grad_w_np = torch_grad_w.numpy()
            torch_grad_V_np = torch_grad_V.numpy()
        
        w_grad_close = np.allclose(rm_grad_w_np, torch_grad_w_np, rtol=1e-3, atol=1e-3)
        v_grad_close = np.allclose(rm_grad_V_np, torch_grad_V_np, rtol=1e-3, atol=1e-3)
        
        error_msg = None
        if not w_grad_close or not v_grad_close:
            error_msg = f"{decomp_func_name} {case_name}梯度比较失败 - {device}"
        
        return w_grad_close, v_grad_close, error_msg
        
    except Exception as e:
        return False, False, f"{decomp_func_name} {case_name}梯度计算错误 - {device}: {str(e)}"

class TestLinalgEig(unittest.TestCase):
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
    
    def _test_eig_single_case(self, np_data, case_name, is_batch=False, device='cpu'):
        """统一的eig测试用例处理函数"""
        try:
            # 转换为riemann张量
            riemann_tensor = rm.tensor(np_data, requires_grad=True, device=device)
            
            # 转换为torch张量
            if TORCH_AVAILABLE:
                torch_device = device if TORCH_CUDA_AVAILABLE else 'cpu'
                torch_tensor = torch.tensor(np_data, requires_grad=True, device=torch_device)
            else:
                torch_tensor = None
            
            # 执行eig分解
            rm_w, rm_V = rm_eig(riemann_tensor)
            
            if TORCH_AVAILABLE:
                torch_w, torch_V = torch_eig(torch_tensor)
                # 处理PyTorch返回的复数格式（批量测试需要）
                if is_batch and torch_w.dim() == 2 and torch_w.shape[1] == 2:  # [实部, 虚部]格式
                    torch_w_complex = torch.view_as_complex(torch_w)
                    torch_w = torch_w_complex
                if is_batch and torch_V.dim() == 3 and torch_V.shape[-1] == 2:  # 复数特征向量
                    torch_V_complex = torch.view_as_complex(torch_V)
                    torch_V = torch_V_complex
            else:
                torch_w, torch_V = None, None
            
            # 比较结果
            passed = compare_values((rm_w, rm_V), (torch_w, torch_V))
            
            # 检查重构误差
            if is_batch:
                # 批量测试只检查第一个批次
                reconstruction_passed = check_eigen_reconstruction(
                    rm.tensor(np_data[0], device=device), rm_w[0], rm_V[0]
                )
            else:
                reconstruction_passed = check_eigen_reconstruction(riemann_tensor, rm_w, rm_V)
            
            return passed, reconstruction_passed, None
            
        except Exception as e:
            return False, False, f"{case_name}测试错误 - {device}: {str(e)}"

    def test_eig_decomposition(self):
        """测试eig分解（合并基本、复数、批量测试）"""
        case_name = "eig分解综合测试"
        start_time = time.time()
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            # 检查CuPy是否支持eig函数
            try:
                import cupy as cp
                if hasattr(cp.linalg, 'eig'):
                    devices.append("cuda")
                else:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"\n{Colors.YELLOW}警告: CuPy不支持eig函数，跳过CUDA测试{Colors.ENDC}")
            except:
                pass
        
        # 定义测试用例列表
        test_cases = [
            {
                'name': '基本eig分解',
                'matrix': np.array([[1, 2], [3, 4]], dtype=np.float32),
                'is_batch': False
            },
            {
                'name': '复数矩阵eig分解',
                'matrix': (np.array([[1, 2], [3, 4]], dtype=np.float32) +
                          1j * np.array([[0, 1], [-1, 0]], dtype=np.float32)),
                'is_batch': False
            },
            {
                'name': '批量eig分解',
                'matrix': np.random.randn(2, 3, 3).astype(np.float32),
                'is_batch': True
            }
        ]
        
        all_passed = True
        error_messages = []
        subcase_results = []  # 存储子用例结果
        
        try:
            for device in devices:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"\n{Colors.BOLD}测试设备: {device}{Colors.ENDC}")
                
                for i, test_case in enumerate(test_cases):
                    subcase_name = test_case['name']
                    device_subcase_name = f"{subcase_name} - {device}"
                    subcase_start_time = time.time()
                    
                    # 使用统一的测试函数处理
                    passed, reconstruction_passed, error_msg = self._test_eig_single_case(
                        test_case['matrix'], subcase_name, test_case['is_batch'], device=device
                    )
                    
                    subcase_time_taken = time.time() - subcase_start_time
                    subcase_passed = passed and reconstruction_passed
                    subcase_results.append({
                        'name': device_subcase_name,
                        'passed': subcase_passed,
                        'time_taken': subcase_time_taken,
                        'value_compare_passed': passed,
                        'reconstruction_passed': reconstruction_passed
                    })
                    
                    if not subcase_passed:
                        all_passed = False
                        if error_msg:
                            error_messages.append(error_msg)
                    
                    # 显示子用例结果
                    if IS_RUNNING_AS_SCRIPT and TORCH_AVAILABLE:
                        status = "通过" if subcase_passed else "失败"
                        color = Colors.OKGREEN if subcase_passed else Colors.FAIL
                        print(f"  子用例: {device_subcase_name} - {color}{status}{Colors.ENDC} ({subcase_time_taken:.4f}秒)")
                        if not passed:
                            print(f"    值比较: 失败")
                        if not reconstruction_passed:
                            print(f"    重构误差: 失败")
                        print()  # 添加空行分隔
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                # 添加子用例统计到总用例表
                for result in subcase_results:
                    subcase_full_name = f"{case_name} - {result['name']}"
                    stats.add_result(subcase_full_name, result['passed'])
                
                stats.add_result(case_name, all_passed)
                status = "通过" if all_passed else "失败"
                color = Colors.OKGREEN if all_passed else Colors.FAIL
                print(f"测试用例: {case_name} - {color}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                print(f"  子用例统计: {sum(r['passed'] for r in subcase_results)}/{len(subcase_results)} 通过")
            
            # 断言确保测试通过
            if not all_passed and TORCH_AVAILABLE:
                error_msg = "; ".join(error_messages) if error_messages else "eig分解测试失败"
                self.assertTrue(False, f"{error_msg}: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    

    
    def _test_eigh_single_case(self, np_data, case_name, is_batch=False, is_complex=False, device='cpu'):
        """统一的eigh测试逻辑"""
        start_time = time.time()
        try:
            # 转换为riemann张量
            riemann_tensor = rm.tensor(np_data, requires_grad=True, device=device)
            
            # 转换为torch张量
            if TORCH_AVAILABLE:
                torch_device = device if TORCH_CUDA_AVAILABLE else 'cpu'
                torch_tensor = torch.tensor(np_data, requires_grad=True, device=torch_device)
            else:
                torch_tensor = None
            
            # 执行eigh分解
            rm_w, rm_V = rm_eigh(riemann_tensor)
            
            if TORCH_AVAILABLE:
                torch_w, torch_V = torch_eigh(torch_tensor)
            else:
                torch_w, torch_V = None, None
            
            # 比较结果
            passed = compare_values((rm_w, rm_V), (torch_w, torch_V))
            
            # 检查重构误差
            if is_batch:
                # 批量测试只检查第一个批次
                reconstruction_passed = check_eigh_reconstruction(
                    rm.tensor(np_data[0], device=device), rm_w[0], rm_V[0]
                )
            else:
                reconstruction_passed = check_eigh_reconstruction(riemann_tensor, rm_w, rm_V)
            
            time_taken = time.time() - start_time
            
            device_case_name = f"{case_name} - {device}"
            if IS_RUNNING_AS_SCRIPT:
                overall_passed = passed and reconstruction_passed
                stats.add_result(device_case_name, overall_passed)
                status = "通过" if overall_passed else "失败"
                print(f"测试用例: {device_case_name} - {Colors.OKGREEN if overall_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
                if not reconstruction_passed:
                    print(f"  重构误差: 失败")
                print()  # 添加空行分隔
            
            # 断言确保测试通过
            self.assertTrue(passed, f"eigh值比较失败: {device_case_name}")
            self.assertTrue(reconstruction_passed, f"eigh重构误差检查失败: {device_case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            device_case_name = f"{case_name} - {device}"
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(device_case_name, False, [str(e)])
                print(f"测试用例: {device_case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_eigh_decomposition(self):
        """统一的eigh分解测试"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        test_cases = [
            {
                'name': '实对称矩阵eigh分解测试',
                'create_data': lambda: np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=np.float32),
                'is_batch': False,
                'is_complex': False
            },
            {
                'name': '复Hermitian矩阵eigh分解测试',
                'create_data': lambda: self._create_hermitian_matrix(3),
                'is_batch': False,
                'is_complex': True
            },
            {
                'name': '批量eigh分解测试',
                'create_data': lambda: self._create_batch_symmetric_matrix(),
                'is_batch': True,
                'is_complex': False
            }
        ]
        
        for device in devices:
            if IS_RUNNING_AS_SCRIPT:
                print(f"\n{Colors.BOLD}测试设备: {device}{Colors.ENDC}")
            
            for test_case in test_cases:
                np_data = test_case['create_data']()
                self._test_eigh_single_case(
                    np_data, 
                    test_case['name'],
                    is_batch=test_case['is_batch'],
                    is_complex=test_case['is_complex'],
                    device=device
                )
    
    def _create_batch_symmetric_matrix(self):
        """创建批量对称矩阵"""
        np_data = np.random.randn(2, 3, 3).astype(np.float32)
        # 使其对称
        return (np_data + np_data.transpose(0, 2, 1)) / 2
    
    def _process_subcase_results(self, case_name, subcase_results, start_time):
        """统一的子用例结果处理和显示函数"""
        time_taken = time.time() - start_time
        all_passed = all(r['passed'] for r in subcase_results)
        
        if IS_RUNNING_AS_SCRIPT:
            # 添加子用例统计到总用例表
            for result in subcase_results:
                subcase_full_name = f"{case_name} - {result['name']}"
                stats.add_result(subcase_full_name, result['passed'])
            
            stats.add_result(case_name, all_passed)
            status = "通过" if all_passed else "失败"
            color = Colors.OKGREEN if all_passed else Colors.FAIL
            print(f"测试用例: {case_name} - {color}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            print(f"  子用例统计: {sum(r['passed'] for r in subcase_results)}/{len(subcase_results)} 通过")
        
        return all_passed, time_taken
    
    def test_eigh_gradient(self):
        """测试eigh函数的梯度计算"""
        case_name = "eigh函数梯度测试"
        start_time = time.time()
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        # 定义测试用例列表
        test_cases = [
            {
                'name': '实对称矩阵',
                'matrix': np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=np.float32)
            },
            {
                'name': '复数Hermitian矩阵',
                'matrix': self._create_hermitian_matrix(3)
            }
        ]
        
        all_passed = True
        error_messages = []
        subcase_results = []  # 存储子用例结果
        
        try:
            for device in devices:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"\n{Colors.BOLD}测试设备: {device}{Colors.ENDC}")
                
                for i, test_case in enumerate(test_cases):
                    subcase_name = test_case['name']
                    device_subcase_name = f"{subcase_name} - {device}"
                    subcase_start_time = time.time()
                    
                    # 使用统一的梯度测试函数
                    w_grad_close, v_grad_close, error_msg = gradient_for_decomp(
                        test_case['matrix'], 'eigh', subcase_name, device=device
                    )
                    
                    subcase_time_taken = time.time() - subcase_start_time
                    subcase_passed = w_grad_close and v_grad_close
                    subcase_results.append({
                        'name': device_subcase_name,
                        'passed': subcase_passed,
                        'time_taken': subcase_time_taken,
                        'w_grad_close': w_grad_close,
                        'v_grad_close': v_grad_close
                    })
                    
                    if not subcase_passed:
                        all_passed = False
                        if error_msg:
                            error_messages.append(error_msg)
                    
                    # 显示子用例结果
                    if IS_RUNNING_AS_SCRIPT and TORCH_AVAILABLE:
                        status = "通过" if subcase_passed else "失败"
                        color = Colors.OKGREEN if subcase_passed else Colors.FAIL
                        print(f"  子用例: {device_subcase_name} - {color}{status}{Colors.ENDC} ({subcase_time_taken:.4f}秒)")
                        if not w_grad_close:
                            print(f"    w梯度比较: 失败")
                        if not v_grad_close:
                            print(f"    V梯度比较: 失败")
            
            # 使用统一的子用例结果处理
            all_passed, time_taken = self._process_subcase_results(case_name, subcase_results, start_time)
            
            # 断言确保测试通过
            if not all_passed and TORCH_AVAILABLE:
                error_msg = "; ".join(error_messages) if error_messages else "eigh梯度比较失败"
                self.assertTrue(False, f"{error_msg}: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_eig_gradient(self):
        """测试eig函数的梯度计算"""
        case_name = "eig函数梯度测试"
        start_time = time.time()
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            # 检查CuPy是否支持eig函数
            try:
                import cupy as cp
                if hasattr(cp.linalg, 'eig'):
                    devices.append("cuda")
                else:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"\n{Colors.YELLOW}警告: CuPy不支持eig函数，跳过CUDA测试{Colors.ENDC}")
            except:
                pass
        
        # 定义测试用例列表
        test_cases = [
            {
                'name': '实数非对称矩阵',
                'matrix': np.array([[1, 2], [3, 4]], dtype=np.float32)
            },
            {
                'name': '复数非对称矩阵', 
                'matrix': (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32) +
                          1j * np.array([[0, 1, -2], [3, 0, 4], [-5, 6, 0]], dtype=np.float32))
            }
        ]
        
        all_passed = True
        error_messages = []
        subcase_results = []  # 存储子用例结果
        
        try:
            for device in devices:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"\n{Colors.BOLD}测试设备: {device}{Colors.ENDC}")
                
                for i, test_case in enumerate(test_cases):
                    subcase_name = test_case['name']
                    device_subcase_name = f"{subcase_name} - {device}"
                    subcase_start_time = time.time()
                    
                    # 使用统一的梯度测试函数
                    w_grad_close, v_grad_close, error_msg = gradient_for_decomp(
                        test_case['matrix'], 'eig', subcase_name, device=device
                    )
                    
                    subcase_time_taken = time.time() - subcase_start_time
                    subcase_passed = w_grad_close and v_grad_close
                    subcase_results.append({
                        'name': device_subcase_name,
                        'passed': subcase_passed,
                        'time_taken': subcase_time_taken,
                        'w_grad_close': w_grad_close,
                        'v_grad_close': v_grad_close
                    })
                    
                    if not subcase_passed:
                        all_passed = False
                        if error_msg:
                            error_messages.append(error_msg)
                    
                    # 显示子用例结果
                    if IS_RUNNING_AS_SCRIPT and TORCH_AVAILABLE:
                        status = "通过" if subcase_passed else "失败"
                        color = Colors.OKGREEN if subcase_passed else Colors.FAIL
                        print(f"  子用例: {device_subcase_name} - {color}{status}{Colors.ENDC} ({subcase_time_taken:.4f}秒)")
                        if not w_grad_close:
                            print(f"    w梯度比较: 失败")
                        if not v_grad_close:
                            print(f"    V梯度比较: 失败")
            
            # 使用统一的子用例结果处理
            all_passed, time_taken = self._process_subcase_results(case_name, subcase_results, start_time)
            
            # 断言确保测试通过
            if not all_passed and TORCH_AVAILABLE:
                error_msg = "; ".join(error_messages) if error_messages else "eig梯度比较失败"
                self.assertTrue(False, f"{error_msg}: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def _create_hermitian_matrix(self, n):
        """创建一个Hermitian矩阵用于测试"""
        np.random.seed(42)  # 确保可重复性
        real_part = np.random.randn(n, n).astype(np.float32)
        imag_part = np.random.randn(n, n).astype(np.float32)
        # 创建复数矩阵
        complex_matrix = real_part + 1j * imag_part
        # 确保是Hermitian: A = A^H
        return (complex_matrix + complex_matrix.conj().T) / 2
        
    def test_eig_output_shape(self):
        """测试eig函数的输出形状"""
        case_name = "eig输出形状测试"
        start_time = time.time()
        try:
            # 定义要测试的设备列表
            devices = ["cpu"]
            if CUDA_AVAILABLE:
                # 检查CuPy是否支持eig函数
                try:
                    import cupy as cp
                    if hasattr(cp.linalg, 'eig'):
                        devices.append("cuda")
                    else:
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"\n{Colors.YELLOW}警告: CuPy不支持eig函数，跳过CUDA测试{Colors.ENDC}")
                except:
                    pass
            
            # 测试不同形状的矩阵
            test_cases = [
                (3, 3),    # 3x3矩阵
                (5, 5),    # 5x5矩阵
                (2, 4, 4), # 批量4x4矩阵
            ]
            
            subcase_results = []
            
            for device in devices:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"\n{Colors.BOLD}测试设备: {device}{Colors.ENDC}")
                
                for shape in test_cases:
                    subcase_start_time = time.time()
                    subcase_name = f"形状{shape} - {device}"
                    
                    try:
                        # 创建测试数据
                        if len(shape) == 2:
                            np_data = np.random.randn(*shape).astype(np.float32)
                        else:
                            np_data = np.random.randn(*shape).astype(np.float32)
                        
                        riemann_tensor = rm.tensor(np_data, device=device)
                        
                        # 执行eig分解
                        w, V = rm_eig(riemann_tensor)
                        
                        # 检查w的形状
                        expected_w_shape = shape[:-1]
                        self.assertEqual(w.shape, expected_w_shape, f"特征值w的形状不匹配: {w.shape} vs {expected_w_shape}")
                        
                        # 检查V的形状
                        expected_V_shape = shape
                        self.assertEqual(V.shape, expected_V_shape, f"特征向量V的形状不匹配: {V.shape} vs {expected_V_shape}")
                        
                        subcase_time_taken = time.time() - subcase_start_time
                        subcase_results.append(True)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name + " - " + subcase_name, True)
                            print(f"  子用例: {subcase_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({subcase_time_taken:.4f}秒)")
                            
                    except Exception as sub_e:
                        subcase_time_taken = time.time() - subcase_start_time
                        subcase_results.append(False)
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name + " - " + subcase_name, False, [str(sub_e)])
                            print(f"  子用例: {subcase_name} - {Colors.FAIL}失败{Colors.ENDC} ({subcase_time_taken:.4f}秒) - {str(sub_e)}")
                        raise
            
            time_taken = time.time() - start_time
            
            # 统计子用例结果
            passed_count = sum(subcase_results)
            total_count = len(subcase_results)
            
            if IS_RUNNING_AS_SCRIPT:
                overall_passed = passed_count == total_count
                stats.add_result(case_name, overall_passed)
                status = "通过" if overall_passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if overall_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                print(f"  子用例统计: {passed_count}/{total_count} 通过")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_eigh_output_shape(self):
        """测试eigh函数的输出形状"""
        case_name = "eigh输出形状测试"
        start_time = time.time()
        try:
            # 定义要测试的设备列表
            devices = ["cpu"]
            if CUDA_AVAILABLE:
                devices.append("cuda")
            
            # 测试不同形状的矩阵
            test_cases = [
                (3, 3),    # 3x3矩阵
                (5, 5),    # 5x5矩阵
                (2, 4, 4), # 批量4x4矩阵
            ]
            
            subcase_results = []
            
            for device in devices:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"\n{Colors.BOLD}测试设备: {device}{Colors.ENDC}")
                
                for shape in test_cases:
                    subcase_start_time = time.time()
                    subcase_name = f"形状{shape} - {device}"
                    
                    try:
                        # 创建对称测试数据
                        if len(shape) == 2:
                            np_data = np.random.randn(*shape).astype(np.float32)
                            np_data = (np_data + np_data.T) / 2  # 使其对称
                        else:
                            np_data = np.random.randn(*shape).astype(np.float32)
                            np_data = (np_data + np_data.transpose(0, 2, 1)) / 2  # 使其对称
                        
                        riemann_tensor = rm.tensor(np_data, device=device)
                        
                        # 执行eigh分解
                        w, V = rm_eigh(riemann_tensor)
                        
                        # 检查w的形状
                        expected_w_shape = shape[:-1]
                        self.assertEqual(w.shape, expected_w_shape, f"特征值w的形状不匹配: {w.shape} vs {expected_w_shape}")
                        
                        # 检查V的形状
                        expected_V_shape = shape
                        self.assertEqual(V.shape, expected_V_shape, f"特征向量V的形状不匹配: {V.shape} vs {expected_V_shape}")
                        
                        subcase_time_taken = time.time() - subcase_start_time
                        subcase_results.append(True)
                        
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name + " - " + subcase_name, True)
                            print(f"  子用例: {subcase_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({subcase_time_taken:.4f}秒)")
                            
                    except Exception as sub_e:
                        subcase_time_taken = time.time() - start_time
                        subcase_results.append(False)
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case_name + " - " + subcase_name, False, [str(sub_e)])
                            print(f"  子用例: {subcase_name} - {Colors.FAIL}失败{Colors.ENDC} ({subcase_time_taken:.4f}秒) - {str(sub_e)}")
                        raise
            
            time_taken = time.time() - start_time
            
            # 统计子用例结果
            passed_count = sum(subcase_results)
            total_count = len(subcase_results)
            
            if IS_RUNNING_AS_SCRIPT:
                overall_passed = passed_count == total_count
                stats.add_result(case_name, overall_passed)
                status = "通过" if overall_passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if overall_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                print(f"  子用例统计: {passed_count}/{total_count} 通过")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise

if __name__ == "__main__":
    clear_screen()

    # 标记为独立脚本运行
    IS_RUNNING_AS_SCRIPT = True
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgEig)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    # 打印统计信息
    stats.print_summary()
    
    # 根据测试结果设置退出代码
    sys.exit(0 if result.wasSuccessful() else 1)