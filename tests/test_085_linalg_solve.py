import unittest
import os
import sys
import time
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 导入Riemann库
import riemann as rm
from riemann import tensor
# 尝试导入PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 判断是否作为独立脚本运行
IS_RUNNING_AS_SCRIPT = __name__ == "__main__"

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# 颜色类，用于美化输出
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# 如果环境不支持颜色，将颜色代码替换为空字符串
try:
    if not sys.stdout.isatty():
        for attr in dir(Colors):
            if not attr.startswith('__'):
                setattr(Colors, attr, '')
except:
    for attr in dir(Colors):
        if not attr.startswith('__'):
            setattr(Colors, attr, '')

# 统计收集器，用于收集测试结果
class StatisticsCollector:
    def __init__(self):
        self.total_cases = 0
        self.passed_cases = 0
        self.total_time = 0.0
        self.function_stats = {}
        self.current_function = None
        self.current_function_start_time = 0
    
    def start_function(self, function_name):
        self.current_function = function_name
        self.current_function_start_time = time.time()
        if function_name not in self.function_stats:
            self.function_stats[function_name] = {"total": 0, "passed": 0, "time": 0.0}
    
    def add_result(self, test_name, passed, errors=None):
        self.total_cases += 1
        if passed:
            self.passed_cases += 1
        
        if self.current_function:
            self.function_stats[self.current_function]["total"] += 1
            if passed:
                self.function_stats[self.current_function]["passed"] += 1
    
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

# 验证AX=B的函数
def verify_solution(A, X, B, rtol=1e-5, atol=1e-5):
    """验证AX=B的解是否正确（left=True的情况）"""
    try:
        # 只有批量1D向量需要特殊处理，其他情况都直接使用 A @ X
        
        # 情况1：批量1D向量 - B是(batch, n) 且 A是3D批量矩阵
        if len(B.shape) >= 2 and B.shape[-1] != 1:
            # 使用(A @ X.unsqueeze(-1)).squeeze(-1)验证
            verification_batch = (A @ X.unsqueeze(-1)).squeeze(-1)
            result = rm.allclose(verification_batch, B, rtol=rtol, atol=atol)
            if not result and IS_RUNNING_AS_SCRIPT:
                print(f"A shape: {A.shape}, X shape: {X.shape}, B shape: {B.shape}")
                print(f"verification_batch shape: {verification_batch.shape}")
                print(f"verification_batch: {verification_batch.data}")
                print(f"B: {B.data}")
                print(f"差值: {rm.abs(verification_batch - B).data}")
            return result
        
        # 情况2：其他所有情况 - 直接使用 A @ X 验证
        else:
            # 包括：
            # - 单个1D向量 (n,): A(n,n) @ X(n,) = (n,)  
            # - 单个2D列向量 (n,1): A(n,n) @ X(n,1) = (n,1)
            # - 批量2D列向量 (batch, n,1): A(batch, n,n) @ X(batch, n,1) = (batch, n,1)
            # - 批量2D矩阵 (batch, n,m): A(batch, n,n) @ X(batch, n,m) = (batch, n,m)
            AX = A @ X
            result = rm.allclose(AX, B, rtol=rtol, atol=atol)
            if not result and IS_RUNNING_AS_SCRIPT:
                print(f"A shape: {A.shape}, X shape: {X.shape}, B shape: {B.shape}")
                print(f"AX shape: {AX.shape}")
                print(f"AX: {AX.data}")
                print(f"B: {B.data}")
                print(f"差值: {rm.abs(AX - B).data}")
            return result
            
    except Exception as e:
        print(f"验证解时出错: {e}")
        return False

# 验证XA=B的函数
def verify_solution_right(A, X, B, rtol=1e-5, atol=1e-5):
    """验证XA=B的解是否正确（left=False的情况）"""
    try:
        # 只有批量1D向量需要特殊处理，其他情况都直接使用 X @ A
        
        # 情况1：批量1D向量 - B是(batch, n)
        if len(B.shape) >=2 and B.shape[-2] != 1:
            # 使用(X.unsqueeze(-2) @ A).squeeze(-2)验证
            verification_batch = (X.unsqueeze(-2) @ A).squeeze(-2)
            result = rm.allclose(verification_batch, B, rtol=rtol, atol=atol)
            if not result and IS_RUNNING_AS_SCRIPT:
                print(f"A shape: {A.shape}, X shape: {X.shape}, B shape: {B.shape}")
                print(f"verification_batch shape: {verification_batch.shape}")
                print(f"verification_batch: {verification_batch.data}")
                print(f"B: {B.data}")
                print(f"差值: {rm.abs(verification_batch - B).data}")
            return result
        
        # 情况2：其他所有情况 - 直接使用 X @ A 验证
        else:
            # 包括：
            # - 批量2D行向量 (batch, 1, n): X(batch, 1, n) @ A(batch, n, n) = (batch, 1, n)
            # - 单个2D行向量 (1, n): X(1, n) @ A(n, n) = (1, n)  
            # - 其他批量矩阵 (batch, m, n): X(batch, m, n) @ A(batch, n, n) = (batch, m, n)
            XA = X @ A
            result = rm.allclose(XA, B, rtol=rtol, atol=atol)
            if not result and IS_RUNNING_AS_SCRIPT:
                print(f"A shape: {A.shape}, X shape: {X.shape}, B shape: {B.shape}")
                print(f"XA shape: {XA.shape}")
                print(f"XA: {XA.data}")
                print(f"B: {B.data}")
                print(f"差值: {rm.abs(XA - B).data}")
            return result
            
    except Exception as e:
        print(f"验证解时出错: {e}")
        return False

# 计算梯度的函数
def compute_gradients(A_data, B_data, left=True, use_out=False):
    """计算solve函数的梯度并返回Riemann和PyTorch的结果"""
    # try:
    # Riemann计算
    A_rm = tensor(A_data, requires_grad=True)
    B_rm = tensor(B_data, requires_grad=True)
    
    if use_out:
        out_rm = rm.zero_like(B_rm)
        X_rm = rm.linalg.solve(A_rm, B_rm, left=left, out=out_rm)
        # 当使用out参数时，返回的X不支持梯度跟踪，所以不计算梯度
        A_grad_rm = None
        B_grad_rm = None
    else:
        X_rm = rm.linalg.solve(A_rm, B_rm, left=left)
        # 对解求和以便计算梯度
        loss_rm = X_rm.sum()
        loss_rm.backward()
        A_grad_rm = A_rm.grad
        B_grad_rm = B_rm.grad
    
    # PyTorch计算
    A_grad_torch = None
    B_grad_torch = None
    
    if TORCH_AVAILABLE:
        if use_out:
            A_torch = torch.tensor(A_data)
            B_torch = torch.tensor(B_data)
            out_torch = torch.zeros(out_shape, dtype=torch.float64)
            X_torch = torch.linalg.solve(A_torch, B_torch, left=left, out=out_torch)
        else:
            A_torch = torch.tensor(A_data, requires_grad=True)
            B_torch = torch.tensor(B_data, requires_grad=True)
            X_torch = torch.linalg.solve(A_torch, B_torch, left=left)
        
            loss_torch = X_torch.sum()
            loss_torch.backward()
            
            A_grad_torch = A_torch.grad
            B_grad_torch = B_torch.grad
    
    return {
        'riemann': {'A_grad': A_grad_rm, 'B_grad': B_grad_rm, 'X': X_rm},
        'torch': {'A_grad': A_grad_torch, 'B_grad': B_grad_torch, 'X': X_torch if TORCH_AVAILABLE else None}
    }
        
    # except Exception as e:
    #     print(f"计算梯度时出错: {e}")
    #     return None

class TestLinalgSolve(unittest.TestCase):
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
            print(f"测试描述: {self._testMethodDoc}")
    
    def tearDown(self):
        # 如果是独立脚本运行，则结束函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_solve_left_1d_vector(self):
        """测试left=True时B为1D向量的场景"""
        # 创建测试数据
        A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        B = np.array([5.0, 10.0], dtype=np.float64)
        
        # Riemann计算
        A_rm = tensor(A, requires_grad=True)
        B_rm = tensor(B, requires_grad=True)
        X_rm = rm.linalg.solve(A_rm, B_rm)
        
        # 验证解
        verification_passed = verify_solution(A_rm, X_rm, B_rm)
        
        # 计算梯度
        gradients = compute_gradients(A, B, left=True)
        
        # 比较梯度
        if TORCH_AVAILABLE and gradients:
            A_grad_match = np.allclose(gradients['riemann']['A_grad'].data, 
                                     gradients['torch']['A_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
            B_grad_match = np.allclose(gradients['riemann']['B_grad'].data, 
                                     gradients['torch']['B_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
        else:
            A_grad_match = True
            B_grad_match = True
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            passed = verification_passed and A_grad_match and B_grad_match
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            stats.add_result("left=True 1D向量", passed)
            print(f"解验证: {status_color}{'通过' if verification_passed else '失败'}{Colors.ENDC}")
            print(f"A梯度匹配: {status_color}{'通过' if A_grad_match else '失败'}{Colors.ENDC}")
            print(f"B梯度匹配: {status_color}{'通过' if B_grad_match else '失败'}{Colors.ENDC}")
        
        # 断言
        self.assertTrue(verification_passed, f"解验证失败：AX != B, A.shape={A.shape}, B.shape={B.shape}, X.shape={X_rm.shape}")
        if TORCH_AVAILABLE:
            self.assertTrue(A_grad_match, "A梯度不匹配")
            self.assertTrue(B_grad_match, "B梯度不匹配")
    
    def test_solve_left_2d_column_vector(self):
        """测试left=True时B为2D列向量的场景"""
        # 创建测试数据
        A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        B = np.array([[5.0], [10.0]], dtype=np.float64)
        
        # Riemann计算
        A_rm = tensor(A, requires_grad=True)
        B_rm = tensor(B, requires_grad=True)
        X_rm = rm.linalg.solve(A_rm, B_rm)
        
        # 验证解
        verification_passed = verify_solution(A_rm, X_rm, B_rm)
        
        # 计算梯度
        gradients = compute_gradients(A, B, left=True)
        
        # 比较梯度
        if TORCH_AVAILABLE and gradients:
            A_grad_match = np.allclose(gradients['riemann']['A_grad'].data, 
                                     gradients['torch']['A_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
            B_grad_match = np.allclose(gradients['riemann']['B_grad'].data, 
                                     gradients['torch']['B_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
        else:
            A_grad_match = True
            B_grad_match = True
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            passed = verification_passed and A_grad_match and B_grad_match
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            stats.add_result("left=True 2D列向量", passed)
            print(f"解验证: {status_color}{'通过' if verification_passed else '失败'}{Colors.ENDC}")
            print(f"A梯度匹配: {status_color}{'通过' if A_grad_match else '失败'}{Colors.ENDC}")
            print(f"B梯度匹配: {status_color}{'通过' if B_grad_match else '失败'}{Colors.ENDC}")
        
        # 断言
        self.assertTrue(verification_passed, f"解验证失败：AX != B, A.shape={A.shape}, B.shape={B.shape}, X.shape={X_rm.shape}")
        if TORCH_AVAILABLE:
            self.assertTrue(A_grad_match, "A梯度不匹配")
            self.assertTrue(B_grad_match, "B梯度不匹配")
    
    def test_solve_left_batch_vectors(self):
        """测试left=True时A、B为批量向量的场景"""
        # 创建测试数据 - 批量向量（A是2x2x2，B是2x2x1）
        A = np.array([[[2.0, 1.0], [1.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
        B = np.array([[[5.0], [10.0]], [[7.0], [15.0]]], dtype=np.float64)  # 批量列向量 (2,2,1)

        # Riemann计算
        A_rm = tensor(A, requires_grad=True)
        B_rm = tensor(B, requires_grad=True)
        X_rm = rm.linalg.solve(A_rm, B_rm)

        # 验证解
        verification_passed = verify_solution(A_rm, X_rm, B_rm)
        
        # 计算梯度
        gradients = compute_gradients(A, B, left=True)
        
        # 比较梯度
        if TORCH_AVAILABLE and gradients:
            A_grad_match = np.allclose(gradients['riemann']['A_grad'].data, 
                                     gradients['torch']['A_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
            B_grad_match = np.allclose(gradients['riemann']['B_grad'].data, 
                                     gradients['torch']['B_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
        else:
            A_grad_match = True
            B_grad_match = True
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            passed = verification_passed and A_grad_match and B_grad_match
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            stats.add_result("left=True 批量向量", passed)
            print(f"解验证: {status_color}{'通过' if verification_passed else '失败'}{Colors.ENDC}")
            print(f"A梯度匹配: {status_color}{'通过' if A_grad_match else '失败'}{Colors.ENDC}")
            print(f"B梯度匹配: {status_color}{'通过' if B_grad_match else '失败'}{Colors.ENDC}")
        
        # 断言
        self.assertTrue(verification_passed, "解验证失败")
        if TORCH_AVAILABLE:
            self.assertTrue(A_grad_match, "A梯度不匹配")
            self.assertTrue(B_grad_match, "B梯度不匹配")
    
    def test_solve_left_batch_1d_vector(self):
        """测试left=True时B为批量1D向量的场景"""
        # 创建测试数据 - 批量1D向量（shape中-2维不为1）
        A = np.array([[[2.0, 1.0], [1.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)  # (2,2,2)
        B = np.array([[5.0, 10.0], [7.0, 15.0]], dtype=np.float64)  # (2,2) - 批量1D向量
        
        # Riemann计算
        A_rm = tensor(A, requires_grad=True)
        B_rm = tensor(B, requires_grad=True)
        X_rm = rm.linalg.solve(A_rm, B_rm)
        
        # 验证解 - AX = B
        verification_passed = verify_solution(A_rm, X_rm, B_rm)
        
        # 计算梯度
        gradients = compute_gradients(A, B, left=True)
        
        # 比较梯度
        if TORCH_AVAILABLE and gradients:
            A_grad_match = np.allclose(gradients['riemann']['A_grad'].data, 
                                     gradients['torch']['A_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
            B_grad_match = np.allclose(gradients['riemann']['B_grad'].data, 
                                     gradients['torch']['B_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
        else:
            A_grad_match = True
            B_grad_match = True
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            passed = verification_passed and A_grad_match and B_grad_match
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            stats.add_result("left=True 批量1D向量", passed)
            print(f"解验证: {status_color}{'通过' if verification_passed else '失败'}{Colors.ENDC}")
            print(f"A梯度匹配: {status_color}{'通过' if A_grad_match else '失败'}{Colors.ENDC}")
            print(f"B梯度匹配: {status_color}{'通过' if B_grad_match else '失败'}{Colors.ENDC}")
        
        # 断言
        self.assertTrue(verification_passed, "解验证失败")
        if TORCH_AVAILABLE:
            self.assertTrue(A_grad_match, "A梯度不匹配")
            self.assertTrue(B_grad_match, "B梯度不匹配")



    def test_solve_right_2d_row_vector(self):
        """测试left=False时B为2D行向量的场景"""
        # 创建测试数据 - 2D行向量（A是2x2，B是1x2）
        A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        B = np.array([[1.0, 3.0]], dtype=np.float64)  # 2D行向量 (1,2) - 使用与try.py一致的数据
        
        # Riemann计算
        A_rm = tensor(A, requires_grad=True)
        B_rm = tensor(B, requires_grad=True)
        X_rm = rm.linalg.solve(A_rm, B_rm, left=False)
        
        # 验证解 - XA = B
        verification_passed = verify_solution_right(A_rm, X_rm, B_rm)
        
        # 计算梯度
        gradients = compute_gradients(A, B, left=False)
        
        # 比较梯度
        if TORCH_AVAILABLE and gradients:
            A_grad_match = np.allclose(gradients['riemann']['A_grad'].data, 
                                     gradients['torch']['A_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
            B_grad_match = np.allclose(gradients['riemann']['B_grad'].data, 
                                     gradients['torch']['B_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
        else:
            A_grad_match = True
            B_grad_match = True
        
        # 计算期望结果
        # XA = B => X = B @ A^(-1)
        A_inv = np.linalg.inv(A)
        X_expected = B @ A_inv  # (1,2) @ (2,2) = (1,2)
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            passed = verification_passed and A_grad_match and B_grad_match
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            stats.add_result("left=False 2D行向量", passed)
            print(f"解验证: {status_color}{'通过' if verification_passed else '失败'}{Colors.ENDC}")
            print(f"A梯度匹配: {status_color}{'通过' if A_grad_match else '失败'}{Colors.ENDC}")
            print(f"B梯度匹配: {status_color}{'通过' if B_grad_match else '失败'}{Colors.ENDC}")
            print(f"期望解: {X_expected}")
            print(f"实际解: {X_rm.data}")
        
        # 断言
        self.assertTrue(verification_passed, "解验证失败")
        if TORCH_AVAILABLE:
            self.assertTrue(A_grad_match, "A梯度不匹配")
            self.assertTrue(B_grad_match, "B梯度不匹配")

    def test_solve_right_batch_2d_row_vector(self):
        """测试left=False时B为批量2D行向量的场景"""
        # 创建测试数据 - 批量2D行向量（A是批量2x2，B是批量1x2）
        A = np.array([[[2.0, 1.0], [1.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)  # (2,2,2)
        B = np.array([[[5.0, 10.0]], [[7.0, 15.0]]], dtype=np.float64)  # (2,1,2) - 批量2D行向量
        
        # Riemann计算
        A_rm = tensor(A, requires_grad=True)
        B_rm = tensor(B, requires_grad=True)
        X_rm = rm.linalg.solve(A_rm, B_rm, left=False)
        
        # 验证解 - XA = B
        verification_passed = verify_solution_right(A_rm, X_rm, B_rm)
        
        # 计算梯度
        gradients = compute_gradients(A, B, left=False)
        
        # 比较梯度
        if TORCH_AVAILABLE and gradients:
            A_grad_match = np.allclose(gradients['riemann']['A_grad'].data, 
                                     gradients['torch']['A_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
            B_grad_match = np.allclose(gradients['riemann']['B_grad'].data, 
                                     gradients['torch']['B_grad'].detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
        else:
            A_grad_match = True
            B_grad_match = True
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            passed = verification_passed and A_grad_match and B_grad_match
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            stats.add_result("left=False 批量2D行向量", passed)
            print(f"解验证: {status_color}{'通过' if verification_passed else '失败'}{Colors.ENDC}")
            print(f"A梯度匹配: {status_color}{'通过' if A_grad_match else '失败'}{Colors.ENDC}")
            print(f"B梯度匹配: {status_color}{'通过' if B_grad_match else '失败'}{Colors.ENDC}")
        
        # 断言
        self.assertTrue(verification_passed, "解验证失败")
        if TORCH_AVAILABLE:
            self.assertTrue(A_grad_match, "A梯度不匹配")
            self.assertTrue(B_grad_match, "B梯度不匹配")
    
    def test_solve_out_parameter(self):
        """测试out参数非None的场景"""
        # 创建测试数据
        A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        B = np.array([5.0, 10.0], dtype=np.float64)
        
        # Riemann计算 with out parameter (A和B不需要梯度，因为out参数不支持自动微分)
        A_rm = tensor(A, requires_grad=False)
        B_rm = tensor(B, requires_grad=False)
        out_rm = tensor(np.zeros_like(B))
        X_rm = rm.linalg.solve(A_rm, B_rm, out=out_rm)
        
        # 验证解
        verification_passed = verify_solution(A_rm, X_rm, B_rm)
        
        # 验证out参数被正确使用
        out_correct = np.allclose(X_rm.data, out_rm.data)
        
        # 验证返回的X不支持梯度跟踪（与PyTorch行为一致）
        x_no_grad = not X_rm.requires_grad
        
        # 由于A和B不需要梯度，我们不需要计算梯度比较
        # 但我们仍然可以验证PyTorch在相同情况下的行为
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            passed = verification_passed and out_correct and x_no_grad
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            stats.add_result("out参数测试", passed)
            print(f"解验证: {status_color}{'通过' if verification_passed else '失败'}{Colors.ENDC}")
            print(f"out参数使用: {status_color}{'通过' if out_correct else '失败'}{Colors.ENDC}")
            print(f"X无梯度跟踪: {status_color}{'通过' if x_no_grad else '失败'}{Colors.ENDC}")
        
        # 断言
        self.assertTrue(verification_passed, "解验证失败")
        self.assertTrue(out_correct, "out参数使用不正确")
        self.assertTrue(x_no_grad, "返回的X应该不支持梯度跟踪")

    def test_solve_right_1d_vector(self):
        """测试left=False时B为单个1D向量的场景（仅Riemann，不与PyTorch比较）"""
        # 创建测试数据 - 1D向量（A是2x2，B是2维1D向量）
        A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        B = np.array([1.0, 3.0], dtype=np.float64)  # 1D向量 (2,)
        
        # Riemann计算
        A_rm = tensor(A, requires_grad=True)
        B_rm = tensor(B, requires_grad=True)
        X_rm = rm.linalg.solve(A_rm, B_rm, left=False)
        
        # 验证解 - XA = B（对于1D向量，B会被视为行向量）
        # X应该是1D向量，形状与B相同
        verification_passed = verify_solution_right(A_rm, X_rm, B_rm)
        
        # 验证X的形状和梯度跟踪
        shape_correct = X_rm.shape == B_rm.shape
        grad_tracking = X_rm.requires_grad
        
        # 计算期望结果（手动验证）
        # XA = B => X = B @ A^(-1)
        A_inv = np.linalg.inv(A)
        X_expected = B @ A_inv  # (2,) @ (2,2) = (2,)
        solution_correct = np.allclose(X_rm.data, X_expected, rtol=1e-5, atol=1e-5)
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            passed = verification_passed and shape_correct and grad_tracking and solution_correct
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            stats.add_result("left=False 1D向量(Riemann only)", passed)
            print(f"解验证: {status_color}{'通过' if verification_passed else '失败'}{Colors.ENDC}")
            print(f"形状正确: {status_color}{'通过' if shape_correct else '失败'}{Colors.ENDC}")
            print(f"梯度跟踪: {status_color}{'通过' if grad_tracking else '失败'}{Colors.ENDC}")
            print(f"解正确: {status_color}{'通过' if solution_correct else '失败'}{Colors.ENDC}")
            print(f"期望解: {X_expected}")
            print(f"实际解: {X_rm.data}")
        
        # 断言
        self.assertTrue(verification_passed, "解验证失败")
        self.assertTrue(shape_correct, "X形状不正确")
        self.assertTrue(grad_tracking, "X应该支持梯度跟踪")
        self.assertTrue(solution_correct, "解不正确")

    def test_solve_right_batch_1d_vectors(self):
        """测试left=False时B为批量1D向量的场景（仅Riemann，不与PyTorch比较）"""
        # 创建测试数据 - 批量1D向量（A是批量2x2，B是批量2维1D向量）
        A = np.array([[[2.0, 1.0], [1.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)  # (2,2,2)
        B = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64)  # (2,2) - 批量1D向量
        
        # Riemann计算
        A_rm = tensor(A, requires_grad=True)
        B_rm = tensor(B, requires_grad=True)
        X_rm = rm.linalg.solve(A_rm, B_rm, left=False)
        
        # 验证解 - XA = B（对于批量1D向量，每个B[i]被视为行向量）
        verification_passed = verify_solution_right(A_rm, X_rm, B_rm)
        
        # 验证X的形状和梯度跟踪
        shape_correct = X_rm.shape == B_rm.shape
        grad_tracking = X_rm.requires_grad
        
        # 计算期望结果（手动验证）
        # XA = B => X = B @ A^(-1)（批量计算）
        X_expected = np.zeros_like(B)
        for i in range(A.shape[0]):
            A_inv = np.linalg.inv(A[i])
            X_expected[i] = B[i] @ A_inv  # (2,) @ (2,2) = (2,)
        solution_correct = np.allclose(X_rm.data, X_expected, rtol=1e-5, atol=1e-5)
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            passed = verification_passed and shape_correct and grad_tracking and solution_correct
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            stats.add_result("left=False 批量1D向量(Riemann only)", passed)
            print(f"解验证: {status_color}{'通过' if verification_passed else '失败'}{Colors.ENDC}")
            print(f"形状正确: {status_color}{'通过' if shape_correct else '失败'}{Colors.ENDC}")
            print(f"梯度跟踪: {status_color}{'通过' if grad_tracking else '失败'}{Colors.ENDC}")
            print(f"解正确: {status_color}{'通过' if solution_correct else '失败'}{Colors.ENDC}")
            print(f"期望解: {X_expected}")
            print(f"实际解: {X_rm.data}")
        
        # 断言
        self.assertTrue(verification_passed, "解验证失败")
        self.assertTrue(shape_correct, "X形状不正确")
        self.assertTrue(grad_tracking, "X应该支持梯度跟踪")
        self.assertTrue(solution_correct, "解不正确")
    
    def test_solve_singular_matrix(self):
        """测试矩阵A不可逆的异常场景"""
        # 创建奇异矩阵
        A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)  # 奇异矩阵
        B = np.array([5.0, 10.0], dtype=np.float64)
        
        # Riemann计算 - 应该抛出异常
        A_rm = tensor(A)
        B_rm = tensor(B)
        
        with self.assertRaises(Exception):
            rm.linalg.solve(A_rm, B_rm)
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result("奇异矩阵异常", True)
            print(f"奇异矩阵异常处理: {Colors.OKGREEN}通过{Colors.ENDC}")
    
    def test_solve_shape_mismatch(self):
        """测试A和B形状不匹配的异常场景"""
        # 创建形状不匹配的数据
        A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)  # 2x2
        B = np.array([5.0, 10.0, 15.0], dtype=np.float64)  # 3维向量，不匹配
        
        # Riemann计算 - 应该抛出异常
        A_rm = tensor(A)
        B_rm = tensor(B)
        
        with self.assertRaises(Exception):
            rm.linalg.solve(A_rm, B_rm)
        
        # 记录结果
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result("形状不匹配异常", True)
            print(f"形状不匹配异常处理: {Colors.OKGREEN}通过{Colors.ENDC}")

# 运行测试
if __name__ == "__main__":
    IS_RUNNING_AS_SCRIPT = True
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行线性代数求解函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgSolve)
    # 创建运行器并设置为禁用默认输出
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    # 运行测试
    result = runner.run(test_suite)
    
    # 打印统计信息
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)