import unittest
import numpy as np
import time
import sys, os

from torch import set_default_device

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.linalg import svd as rm_svd
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
    from torch.linalg import svd as torch_svd
    TORCH_AVAILABLE = True
    # 检查PyTorch CUDA可用性
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的svd函数")
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

# 设备列表
devices = ['cpu']
if CUDA_AVAILABLE:
    devices.append('cuda')

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
    """比较Riemann和PyTorch的SVD结果是否接近"""
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
        
        # 检查是否在CUDA上运行
        is_cuda = False
        if rm_result and hasattr(rm_result[0].data, 'get'):
            is_cuda = True
        
        # 在CUDA上使用更大的误差阈值
        if is_cuda:
            atol = max(atol, 1e-4)
            rtol = max(rtol, 1e-4)
        
        all_passed = True
        for i, (r, t) in enumerate(zip(rm_result, torch_result)):
            # 对于奇异值，我们直接比较
            if i == 1:  # S (奇异值)
                try:
                    # 确保在CPU上比较
                    r_data = r.data
                    # 检查是否是cupy数组
                    if hasattr(r_data, 'get'):
                        r_data = r_data.get()
                    t_data = t.detach().cpu().numpy() if TORCH_CUDA_AVAILABLE else t.detach().numpy()
                    np.testing.assert_allclose(r_data, t_data, rtol=rtol, atol=atol)
                except AssertionError:
                    all_passed = False
                    break
            else:  # U 和 Vh (奇异向量)
                # 对于奇异向量，由于可能存在符号差异，我们检查它们是否张成相同的空间
                # 通过检查重构误差来验证
                if i == 0:  # U
                    # 我们只比较前k列，其中k是奇异值的数量
                    k = min(r.shape[-2], r.shape[-1])
                    try:
                        # 确保在CPU上比较
                        r_data = r.data
                        # 检查是否是cupy数组
                        if hasattr(r_data, 'get'):
                            r_data = r_data.get()
                        t_data = t.detach().cpu().numpy() if TORCH_CUDA_AVAILABLE else t.detach().numpy()
                        np.testing.assert_allclose(
                            np.abs(r_data[..., :k]), 
                            np.abs(t_data[..., :k]), 
                            rtol=rtol, 
                            atol=atol
                        )
                    except AssertionError:
                        all_passed = False
                        break
                else:  # Vh
                    # 对于Vh，我们比较前k行
                    k = min(r.shape[-2], r.shape[-1])
                    try:
                        # 确保在CPU上比较
                        r_data = r.data
                        # 检查是否是cupy数组
                        if hasattr(r_data, 'get'):
                            r_data = r_data.get()
                        t_data = t.detach().cpu().numpy() if TORCH_CUDA_AVAILABLE else t.detach().numpy()
                        np.testing.assert_allclose(
                            np.abs(r_data[:k]), 
                            np.abs(t_data[:k]), 
                            rtol=rtol, 
                            atol=atol
                        )
                    except AssertionError:
                        all_passed = False
                        break
        
        return all_passed
    
    return False

# 检查重构误差
def check_reconstruction(A, U, S, Vh, atol=1e-6):
    """检查使用SVD结果重构原始矩阵的误差"""
    # 检查是否在CUDA上运行
    is_cuda = False
    if hasattr(S.data, 'get'):
        is_cuda = True
    
    # 在CUDA上使用更大的误差阈值
    if is_cuda:
        atol = max(atol, 1.5e-6)
    
    # 对于不同维度情况进行处理
    if U.ndim == 2 and S.ndim == 1 and Vh.ndim == 2:
        # 2D矩阵情况
        m, n = A.shape[-2], A.shape[-1]
        k = min(m, n)
        
        # 获取U和Vh的实际形状，而不是假设完整矩阵
        u_shape = U.shape[-1]  # U的最后一个维度大小
        vh_shape = Vh.shape[0]  # Vh的第一个维度大小
        
        # 创建与U和Vh兼容的对角矩阵
        sigma = np.zeros((u_shape, vh_shape))
        # 只填充我们有的奇异值
        actual_k = min(k, u_shape, vh_shape)
        
        # 确保在CPU上操作
        S_data = S.data
        # 检查是否是cupy数组
        if hasattr(S_data, 'get'):
            S_data = S_data.get()
        sigma[:actual_k, :actual_k] = np.diag(S_data[:actual_k])
        
        # 重构矩阵
        U_data = U.data
        # 检查是否是cupy数组
        if hasattr(U_data, 'get'):
            U_data = U_data.get()
        Vh_data = Vh.data
        # 检查是否是cupy数组
        if hasattr(Vh_data, 'get'):
            Vh_data = Vh_data.get()
        A_data = A.data
        # 检查是否是cupy数组
        if hasattr(A_data, 'get'):
            A_data = A_data.get()
        
        reconstructed = U_data @ sigma @ Vh_data
        
        # 计算重构误差
        error = np.linalg.norm(A_data - reconstructed)
        return error < atol
    elif U.ndim > 2:
        # 批处理情况
        # 简化版：只检查第一个批次
        m, n = A.shape[-2], A.shape[-1]
        k = min(m, n)
        
        # 获取U和Vh的实际形状
        u_shape = U.shape[-1]  # U的最后一个维度大小
        vh_shape = Vh.shape[-2]  # Vh的倒数第二个维度大小
        
        # 对于第一个批次进行重构检查
        sigma = np.zeros((u_shape, vh_shape))
        actual_k = min(k, u_shape, vh_shape)
        
        # 确保在CPU上操作
        S_data = S.data
        # 检查是否是cupy数组
        if hasattr(S_data, 'get'):
            S_data = S_data.get()
        sigma[:actual_k, :actual_k] = np.diag(S_data[0, :actual_k])
        
        U_data = U.data
        # 检查是否是cupy数组
        if hasattr(U_data, 'get'):
            U_data = U_data.get()
        Vh_data = Vh.data
        # 检查是否是cupy数组
        if hasattr(Vh_data, 'get'):
            Vh_data = Vh_data.get()
        A_data = A.data
        # 检查是否是cupy数组
        if hasattr(A_data, 'get'):
            A_data = A_data.get()
        
        reconstructed = U_data[0] @ sigma @ Vh_data[0]
        error = np.linalg.norm(A_data[0] - reconstructed)
        return error < atol
    
    return False

class TestLinalgSVD(unittest.TestCase):
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
    
    def test_basic_svd(self):
        """测试基本的SVD分解"""
        case_name = "基本SVD分解测试"
        start_time = time.time()
        try:
            # 创建测试数据 - 使用float32精度
            np_data = np.random.randn(5, 4).astype(np.float32)
            
            # 遍历设备
            for device in devices:
                device_case_name = case_name + f" - {device}"
                
                # 转换为riemann张量
                riemann_tensor = rm.tensor(np_data, requires_grad=True, device=device)
                
                # 转换为torch张量
                if TORCH_AVAILABLE:
                    # 确保PyTorch也在相应设备上
                    torch_device = device if TORCH_CUDA_AVAILABLE else 'cpu'
                    torch_tensor = torch.tensor(np_data, requires_grad=True, device=torch_device)
                else:
                    torch_tensor = None
                
                # 测试full_matrices=True情况
                case_subname = "full_matrices=True"
                rm_U, rm_S, rm_Vh = rm_svd(riemann_tensor, full_matrices=True)
                
                if TORCH_AVAILABLE:
                    torch_U, torch_S, torch_Vh = torch_svd(torch_tensor, full_matrices=True)
                else:
                    torch_U, torch_S, torch_Vh = None, None, None
                
                # 比较结果
                passed = compare_values((rm_U, rm_S, rm_Vh), (torch_U, torch_S, torch_Vh))
                
                # 检查重构误差
                reconstruction_passed = check_reconstruction(riemann_tensor, rm_U, rm_S, rm_Vh, atol=1e-6)
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    overall_passed = passed and reconstruction_passed
                    stats.add_result(device_case_name + " - " + case_subname, overall_passed)
                    status = "通过" if overall_passed else "失败"
                    print(f"测试用例: {device_case_name} - {case_subname} - {Colors.OKGREEN if overall_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  值比较: 失败")
                    if not reconstruction_passed:
                        print(f"  重构误差: 失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"SVD值比较失败: {device_case_name} - {case_subname}")
                self.assertTrue(reconstruction_passed, f"SVD重构误差检查失败: {device_case_name} - {case_subname}")
                
                # 测试full_matrices=False情况
                case_subname = "full_matrices=False"
                rm_U, rm_S, rm_Vh = rm_svd(riemann_tensor, full_matrices=False)
                
                if TORCH_AVAILABLE:
                    torch_U, torch_S, torch_Vh = torch_svd(torch_tensor, full_matrices=False)
                else:
                    torch_U, torch_S, torch_Vh = None, None, None
                
                # 比较结果
                passed = compare_values((rm_U, rm_S, rm_Vh), (torch_U, torch_S, torch_Vh))
                
                # 检查重构误差，在CUDA上使用更大的误差阈值
                cuda_atol = 1e-1 if device == 'cuda' else 1e-6
                reconstruction_passed = check_reconstruction(riemann_tensor, rm_U, rm_S, rm_Vh, atol=cuda_atol)
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    overall_passed = passed and reconstruction_passed
                    stats.add_result(device_case_name + " - " + case_subname, overall_passed)
                    status = "通过" if overall_passed else "失败"
                    print(f"测试用例: {device_case_name} - {case_subname} - {Colors.OKGREEN if overall_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  值比较: 失败")
                    if not reconstruction_passed:
                        print(f"  重构误差: 失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"SVD值比较失败: {device_case_name} - {case_subname}")
                self.assertTrue(reconstruction_passed, f"SVD重构误差检查失败: {device_case_name} - {case_subname}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_batch_svd(self):
        """测试批量SVD分解"""
        case_name = "批量SVD分解测试"
        start_time = time.time()
        try:
            # 创建测试数据 - 使用float32精度，批量大小为2
            np_data = np.random.randn(2, 3, 4).astype(np.float32)
            
            # 遍历设备
            for device in devices:
                device_case_name = case_name + f" - {device}"
                
                # 转换为riemann张量
                riemann_tensor = rm.tensor(np_data, requires_grad=True, device=device)
                
                # 转换为torch张量
                if TORCH_AVAILABLE:
                    # 确保PyTorch也在相应设备上
                    torch_device = device if TORCH_CUDA_AVAILABLE else 'cpu'
                    torch_tensor = torch.tensor(np_data, requires_grad=True, device=torch_device)
                else:
                    torch_tensor = None
                
                # 执行SVD分解
                rm_U, rm_S, rm_Vh = rm_svd(riemann_tensor, full_matrices=False)
                
                if TORCH_AVAILABLE:
                    torch_U, torch_S, torch_Vh = torch_svd(torch_tensor, full_matrices=False)
                else:
                    torch_U, torch_S, torch_Vh = None, None, None
                
                # 比较结果
                passed = compare_values((rm_U, rm_S, rm_Vh), (torch_U, torch_S, torch_Vh))    
                
                # 检查重构误差
                reconstruction_passed = check_reconstruction(riemann_tensor, rm_U, rm_S, rm_Vh, atol=1e-6)
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    overall_passed = passed and reconstruction_passed
                    stats.add_result(device_case_name, overall_passed)
                    status = "通过" if overall_passed else "失败"
                    print(f"测试用例: {device_case_name} - {Colors.OKGREEN if overall_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  值比较: 失败")
                    if not reconstruction_passed:
                        print(f"  重构误差: 失败")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"批量SVD值比较失败: {device_case_name}")
                self.assertTrue(reconstruction_passed, f"批量SVD重构误差检查失败: {device_case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_svd_gradients(self):
        """测试SVD的梯度计算"""
        case_name = "SVD梯度计算测试"
        start_time = time.time()
        try:
            # 仅当PyTorch可用时进行此测试
            if not TORCH_AVAILABLE:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"{Colors.WARNING}跳过梯度测试：PyTorch不可用{Colors.ENDC}")
                return
            
            # 创建小规模确定性测试矩阵（确保是良态的，条件数不大）
            test_matrices = [
                # 1. 2x2对称正定矩阵（良态）
                np.array([[2.0, 0.5], 
                          [0.5, 1.0]], dtype=np.float32),
                
                # 2. 3x3对角矩阵，含0奇异值
                np.array([[3.0, 0.0, 0.0], 
                          [0.0, 2.0, 0.0], 
                          [0.0, 0.0, 0.0]], dtype=np.float32),

                # 3. 4x4矩阵，对角元素差距不大
                np.array(  [[3.0, 0.0, -1.0, 5.0], 
                            [1.0, -2.0, 2.0, 0.0], 
                            [0.0, -1.0, 1.5, 0.0],
                            [3.0, 1.0,  0.0, 1.0]], dtype=np.float32),
                
                
                # 4. 3x2的秩2矩阵
                np.array([[1.0, 2.0], 
                        [0.0, 1.0], 
                        [2., 1.]], dtype=np.float32),
                
                # 5. 2x3的秩2矩阵
                np.array([[1.0,1.0, 2.0], 
                        [2.0,2.0, 5.0]], dtype=np.float32),
                
                # 6. 2x2复数矩阵
                np.array([[1.0+1.0j, 0.0-1.0j], 
                        [0.0+1.0j, 1.0+0.0j]], dtype=np.complex64),
                
                # 7. 3x2复数矩阵
                np.array([[1.0+1.0j, 0.0-1.0j], 
                        [0.0+1.0j, 1.0+0.0j],
                        [1.0-1.0j, 0.0+1.0j]], dtype=np.complex64),
                
                # 8. 2x3复数矩阵
                np.array([[1.0+1.0j, 0.0-1.0j, 1.0+0.0j], 
                        [0.0+1.0j, 1.0+0.0j, 1.0-1.0j]], dtype=np.complex64),
            ]
            
            # 矩阵名称
            matrix_names = ["2x2对称正定矩阵", 
                            "3x3对角矩阵，含0奇异值", 
                            "4x4矩阵", 
                            "3x2秩2矩阵",
                            "2x3秩2矩阵", 
                            "2x2复数矩阵",
                            "3x2复数矩阵", 
                            "2x3复数矩阵"]
            
            # 初始化计数器
            test_count = 0
            all_passed = True
            
            # 遍历设备
            for device in devices:
                device_case_name = case_name + f" - {device}"
                if IS_RUNNING_AS_SCRIPT:
                    print(f"\n{Colors.BOLD}测试设备: {device}{Colors.ENDC}")
                
                device_all_passed = True
                
                for matrix_idx, np_data in enumerate(test_matrices):
                    matrix_name = matrix_names[matrix_idx]
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"\n测试矩阵 {matrix_idx+1}/{len(test_matrices)}: {matrix_name}")
                                    
                    # 测试U的梯度
                    subcase_name = f"U梯度测试 ({matrix_name})"
                    try:
                        # Riemann计算
                        riemann_tensor_u = rm.tensor(np_data.copy(), requires_grad=True, device=device)
                        rm_U, _, _ = rm_svd(riemann_tensor_u, full_matrices=False)
                        # SVD分解中，S中奇异值始终非负，U的i列向量方向，Vh的i行向量方向，同时变化时，U@S@Vh的积保持不变
                        # 所以比较Rieman和Torch的U、Vh的反向梯度时，对U、Vh取绝对值后求和，才能确保梯度计算一致
                        rm_U_sum = rm_U.abs().sum()
                        rm_U_sum.backward()
                        # 确保在CPU上比较
                        rm_grad_u = riemann_tensor_u.grad.data
                        # 检查是否是cupy数组
                        if hasattr(rm_grad_u, 'get'):
                            rm_grad_u = rm_grad_u.get()
                        else:
                            rm_grad_u = rm_grad_u.copy()
                        
                        # PyTorch计算
                        # 确保PyTorch也在相应设备上
                        torch_device = device if TORCH_CUDA_AVAILABLE else 'cpu'
                        torch_tensor_u = torch.tensor(np_data.copy(), requires_grad=True, device=torch_device)
                        torch_U, _, _ = torch_svd(torch_tensor_u, full_matrices=False)
                        # SVD分解中，S中奇异值始终非负，U的i列向量方向，Vh的i行向量方向，同时变化时，U@S@Vh的积保持不变
                        # 所以比较Rieman和Torch的U、Vh的反向梯度时，对U、Vh取绝对值后求和，才能确保梯度计算一致                    
                        torch_U_sum = torch_U.abs().sum()
                        torch_U_sum.backward()
                        torch_grad_u = torch_tensor_u.grad.detach().cpu().numpy()
                        
                        # 比较梯度值，在CUDA上使用更大的误差阈值
                        cuda_rtol = 1e-2 if device == 'cuda' else 1e-3
                        cuda_atol = 1e-2 if device == 'cuda' else 1e-3
                        u_grad_close = np.allclose(rm_grad_u, torch_grad_u, rtol=cuda_rtol, atol=cuda_atol)
                        if u_grad_close:
                            # 测试通过时输出状态信息
                            if IS_RUNNING_AS_SCRIPT:
                                print(f"  U梯度比较{Colors.OKGREEN}通过{Colors.ENDC} ({matrix_name})")
                        else:
                            device_all_passed = False
                            if IS_RUNNING_AS_SCRIPT:
                                print(f"  {Colors.FAIL}U梯度比较失败{Colors.ENDC} ({matrix_name})")
                                print(f"    最大绝对误差: {np.max(np.abs(rm_grad_u - torch_grad_u)):.6f}")
                                print(f"    Riemann梯度:\n{rm_grad_u}")
                                print(f"    PyTorch梯度:\n{torch_grad_u}")
                        
                        # 记录子测试结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(device_case_name + " - " + subcase_name, u_grad_close)
                            test_count += 1
                    except Exception as e:
                        device_all_passed = False
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"  {Colors.FAIL}U梯度计算异常{Colors.ENDC} ({matrix_name}): {str(e)}")
                            stats.add_result(device_case_name + " - " + subcase_name, False, f"异常: {str(e)}")
                    
                    # 测试S的梯度
                    subcase_name = f"S梯度测试 ({matrix_name})"
                    try:
                        # Riemann计算
                        riemann_tensor_s = rm.tensor(np_data.copy(), requires_grad=True, device=device)
                        _, rm_S, _ = rm_svd(riemann_tensor_s, full_matrices=False)
                        rm_S_sum = rm_S.sum()
                        rm_S_sum.backward()
                        # 确保在CPU上比较
                        rm_grad_s = riemann_tensor_s.grad.data
                        # 检查是否是cupy数组
                        if hasattr(rm_grad_s, 'get'):
                            rm_grad_s = rm_grad_s.get()
                        else:
                            rm_grad_s = rm_grad_s.copy()
                        
                        # PyTorch计算
                        torch_tensor_s = torch.tensor(np_data.copy(), requires_grad=True, device=torch_device)
                        _, torch_S, _ = torch_svd(torch_tensor_s, full_matrices=False)
                        torch_S_sum = torch_S.sum()
                        torch_S_sum.backward()
                        torch_grad_s = torch_tensor_s.grad.detach().cpu().numpy()
                        
                        # 比较梯度值，在CUDA上使用更大的误差阈值
                        cuda_rtol = 1e-2 if device == 'cuda' else 1e-3
                        cuda_atol = 1e-2 if device == 'cuda' else 1e-3
                        s_grad_close = np.allclose(rm_grad_s, torch_grad_s, rtol=cuda_rtol, atol=cuda_atol)
                        if s_grad_close:
                            # 测试通过时输出状态信息
                            if IS_RUNNING_AS_SCRIPT:
                                print(f"  S梯度比较{Colors.OKGREEN}通过{Colors.ENDC} ({matrix_name})")
                        else:
                            device_all_passed = False
                            if IS_RUNNING_AS_SCRIPT:
                                print(f"  {Colors.FAIL}S梯度比较失败{Colors.ENDC} ({matrix_name})")
                                print(f"    最大绝对误差: {np.max(np.abs(rm_grad_s - torch_grad_s)):.6f}")
                                print(f"    Riemann梯度:\n{rm_grad_s}")
                                print(f"    PyTorch梯度:\n{torch_grad_s}")
                        
                        # 记录子测试结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(device_case_name + " - " + subcase_name, s_grad_close)
                            test_count += 1
                    except Exception as e:
                        device_all_passed = False
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"  {Colors.FAIL}S梯度计算异常{Colors.ENDC} ({matrix_name}): {str(e)}")
                            stats.add_result(device_case_name + " - " + subcase_name, False, f"异常: {str(e)}")
                    
                    # 测试Vh的梯度
                    subcase_name = f"Vh梯度测试 ({matrix_name})"
                    try:
                        # Riemann计算
                        riemann_tensor_vh = rm.tensor(np_data.copy(), requires_grad=True, device=device)
                        _, _, rm_Vh = rm_svd(riemann_tensor_vh, full_matrices=False)
                        # SVD分解中，S中奇异值始终非负，U的i列向量方向，Vh的i行向量方向，同时变化时，U@S@Vh的积保持不变
                        # 所以比较Rieman和Torch的U、Vh的反向梯度时，对U、Vh取绝对值后求和，才能确保梯度计算一致                    
                        rm_Vh_sum = rm_Vh.abs().sum()
                        rm_Vh_sum.backward()
                        # 确保在CPU上比较
                        rm_grad_vh = riemann_tensor_vh.grad.data
                        # 检查是否是cupy数组
                        if hasattr(rm_grad_vh, 'get'):
                            rm_grad_vh = rm_grad_vh.get()
                        else:
                            rm_grad_vh = rm_grad_vh.copy()
                        
                        # PyTorch计算
                        torch_tensor_vh = torch.tensor(np_data.copy(), requires_grad=True, device=torch_device)
                        _, _, torch_Vh = torch_svd(torch_tensor_vh, full_matrices=False)
                        # SVD分解中，S中奇异值始终非负，U的i列向量方向，Vh的i行向量方向，同时变化时，U@S@Vh的积保持不变
                        # 所以比较Rieman和Torch的U、Vh的反向梯度时，对U、Vh取绝对值后求和，才能确保梯度计算一致                    
                        torch_Vh_sum = torch_Vh.abs().sum()
                        torch_Vh_sum.backward()
                        torch_grad_vh = torch_tensor_vh.grad.detach().cpu().numpy()
                        
                        # 比较梯度值，在CUDA上使用更大的误差阈值
                        cuda_rtol = 1e-2 if device == 'cuda' else 1e-3
                        cuda_atol = 1e-2 if device == 'cuda' else 1e-3
                        vh_grad_close = np.allclose(rm_grad_vh, torch_grad_vh, rtol=cuda_rtol, atol=cuda_atol)
                        if vh_grad_close:
                            # 测试通过时输出状态信息
                            if IS_RUNNING_AS_SCRIPT:
                                print(f"  Vh梯度比较{Colors.OKGREEN}通过{Colors.ENDC} ({matrix_name})")
                        else:
                            device_all_passed = False
                            if IS_RUNNING_AS_SCRIPT:
                                print(f"  {Colors.FAIL}Vh梯度比较失败{Colors.ENDC} ({matrix_name})")
                                print(f"    最大绝对误差: {np.max(np.abs(rm_grad_vh - torch_grad_vh)):.6f}")
                                print(f"    Riemann梯度:\n{rm_grad_vh}")
                                print(f"    PyTorch梯度:\n{torch_grad_vh}")
                        
                        # 记录子测试结果
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(device_case_name + " - " + subcase_name, vh_grad_close)
                            test_count += 1
                    except Exception as e:
                        device_all_passed = False
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"  {Colors.FAIL}Vh梯度计算异常{Colors.ENDC} ({matrix_name}): {str(e)}")
                            stats.add_result(device_case_name + " - " + subcase_name, False, f"异常: {str(e)}")
                
                if not device_all_passed:
                    all_passed = False
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                # 不再添加额外的总测试结果，避免重复计数
                status = "通过" if all_passed else "失败"
                print(f"\n测试用例: {case_name} - {Colors.OKGREEN if all_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                print(f"  执行了 {test_count} 个子测试")
            
            # 断言确保所有梯度比较通过
            self.assertTrue(all_passed, "SVD梯度测试失败：梯度值与PyTorch结果不匹配")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_svd_output_shape(self):
        """测试SVD的输出形状"""
        case_name = "SVD输出形状测试"
        start_time = time.time()
        try:
            # 测试不同形状的矩阵
            test_cases = [
                ((5, 4), True),   # m > n, full_matrices=True
                ((5, 4), False),  # m > n, full_matrices=False
                ((4, 5), True),   # m < n, full_matrices=True
                ((4, 5), False),  # m < n, full_matrices=False
                ((3, 3), True),   # m = n, full_matrices=True
                ((3, 3), False),  # m = n, full_matrices=False
            ]
            
            # 遍历设备
            for device in devices:
                device_case_name = case_name + f" - {device}"
                if IS_RUNNING_AS_SCRIPT:
                    print(f"\n{Colors.BOLD}测试设备: {device}{Colors.ENDC}")
                
                for shape, full_matrices in test_cases:
                    subcase_name = f"形状{shape}, full_matrices={full_matrices}"
                    
                    # 创建测试数据
                    np_data = np.random.randn(*shape).astype(np.float32)
                    riemann_tensor = rm.tensor(np_data, device=device)
                    
                    # 执行SVD
                    U, S, Vh = rm_svd(riemann_tensor, full_matrices=full_matrices)
                    
                    # 计算预期形状
                    m, n = shape
                    k = min(m, n)
                    
                    # 检查U的形状
                    expected_U_shape = (m, m) if full_matrices else (m, k)
                    self.assertEqual(U.shape, expected_U_shape, f"U的形状不匹配: {U.shape} vs {expected_U_shape}")
                    
                    # 检查S的形状
                    expected_S_shape = (k,)
                    self.assertEqual(S.shape, expected_S_shape, f"S的形状不匹配: {S.shape} vs {expected_S_shape}")
                    
                    # 检查Vh的形状
                    expected_Vh_shape = (n, n) if full_matrices else (k, n)
                    self.assertEqual(Vh.shape, expected_Vh_shape, f"Vh的形状不匹配: {Vh.shape} vs {expected_Vh_shape}")
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(device_case_name + " - " + subcase_name, True)
                        print(f"测试用例: {device_case_name} - {subcase_name} - {Colors.OKGREEN}通过{Colors.ENDC}")
            
            time_taken = time.time() - start_time
            
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
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgSVD)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    # 打印统计信息
    stats.print_summary()
    
    # 根据测试结果设置退出代码
    sys.exit(0 if result.wasSuccessful() else 1)