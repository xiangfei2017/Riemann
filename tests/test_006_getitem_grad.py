import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    from riemann.autograd.functional import hessian as rm_hessian  # 添加hessian函数导入
    from riemann import tensor as rm_tensor
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
    from torch import autograd as torch_autograd
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的索引操作梯度")
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
            torch_data = torch_result.detach().numpy()
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

class TestGetitemGradFunctions(unittest.TestCase):
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
    
    def _test_getitem_grad(self, case_name, input_np, index, device="cpu"):
        """测试索引操作的梯度计算"""
        start_time = time.time()
        try:
            # 测试Riemann
            if device == "cpu":
                x_riemann = rm_tensor(input_np, requires_grad=True)
            else:  # cuda
                x_riemann = rm_tensor(input_np, requires_grad=True, device=device)
            indexed_riemann = x_riemann[index]
            
            # 确保只对标量调用backward()
            if indexed_riemann.ndim > 0:
                # 对于非标量结果，求和后再调用backward()
                scalar_output = indexed_riemann.sum()
                scalar_output.backward()
            else:
                indexed_riemann.backward()
            
            riemann_grad = x_riemann.grad
            
            # 测试PyTorch
            if TORCH_AVAILABLE:
                if device == "cpu":
                    x_torch = torch.tensor(input_np, requires_grad=True, dtype=torch.float32)
                else:  # cuda
                    x_torch = torch.tensor(input_np, requires_grad=True, dtype=torch.float32, device=device)
                indexed_torch = x_torch[index]
                
                # 与Riemann相同的处理方式
                if len(indexed_torch.shape) > 0:
                    scalar_output_torch = indexed_torch.sum()
                    scalar_output_torch.backward()
                else:
                    indexed_torch.backward()
                
                torch_grad = x_torch.grad
            else:
                torch_grad = None
            
            # 比较结果
            passed = compare_values(riemann_grad, torch_grad)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"索引操作梯度计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    # 然后修改_test_getitem_second_grad方法
    def _test_getitem_second_grad(self, case_name, input_np, device="cpu"):
        """测试索引操作的二阶导数计算"""
        start_time = time.time()
        try:
            # 定义函数 f(x) = x[0]^2 + x[1]^3
            def f_riemann(x):
                # 计算x[0]^2 + x[1]^3
                y = x[0] ** 2. + x[1] ** 3.
                return y
            
            def f_torch(x):
                y = x[0] ** 2. + x[1] ** 3.
                return y
            
            # 测试Riemann - 使用hessian函数直接计算Hessian矩阵，与PyTorch处理方式一致
            if device == "cpu":
                x_riemann = rm_tensor(input_np, requires_grad=True)
            else:  # cuda
                x_riemann = rm_tensor(input_np, requires_grad=True, device=device)
            hessian_riemann = rm_hessian(f_riemann, x_riemann)
            
            # 测试PyTorch - 使用torch.autograd.functional.hessian
            if TORCH_AVAILABLE:
                if device == "cpu":
                    x_torch = torch.tensor(input_np, requires_grad=True, dtype=torch.float32)
                else:  # cuda
                    x_torch = torch.tensor(input_np, requires_grad=True, dtype=torch.float32, device=device)
                hessian_torch = torch.autograd.functional.hessian(f_torch, x_torch)
            else:
                hessian_torch = None
            
            # 比较结果
            passed = compare_values(hessian_riemann, hessian_torch)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"索引操作二阶导数计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_scalar_indexing(self):
        """测试场景1: 标量索引"""
        input_np = np.random.randn(5).astype(np.float32)
        index = 2
        
        # CPU场景测试
        self._test_getitem_grad("标量索引 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("标量索引 CUDA", input_np, index, device="cuda")
    
    def test_slice_indexing(self):
        """测试场景2: 切片索引"""
        input_np = np.random.randn(5).astype(np.float32)
        index = slice(1, 4)
        
        # CPU场景测试
        self._test_getitem_grad("切片索引 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("切片索引 CUDA", input_np, index, device="cuda")
    
    def test_array_indexing(self):
        """测试场景3: 整数数组索引"""
        input_np = np.random.randn(5).astype(np.float32)
        index = [0, 2, 4]
        
        # CPU场景测试
        self._test_getitem_grad("整数数组索引 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("整数数组索引 CUDA", input_np, index, device="cuda")
    
    def test_boolean_indexing(self):
        """测试场景4: 布尔索引"""
        input_np = np.random.randn(5).astype(np.float32)
        index = input_np > 0  # 创建布尔掩码
        
        # CPU场景测试
        self._test_getitem_grad("布尔索引 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("布尔索引 CUDA", input_np, index, device="cuda")
    
    def test_2d_single_indexing(self):
        """测试场景5: 二维数组的单索引"""
        input_np = np.random.randn(3, 4).astype(np.float32)
        index = 1  # 选择第二行
        
        # CPU场景测试
        self._test_getitem_grad("二维单索引 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("二维单索引 CUDA", input_np, index, device="cuda")
    
    def test_2d_tuple_indexing(self):
        """测试场景6: 二维数组的元组索引"""
        input_np = np.random.randn(3, 4).astype(np.float32)
        index = (1, 2)  # 选择第2行第3列
        
        # CPU场景测试
        self._test_getitem_grad("二维元组索引 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("二维元组索引 CUDA", input_np, index, device="cuda")
    
    def test_2d_mixed_indexing(self):
        """测试场景7: 二维数组的混合索引（整数+切片）"""
        input_np = np.random.randn(3, 4).astype(np.float32)
        index = (1, slice(0, 3))  # 第2行，前3列
        
        # CPU场景测试
        self._test_getitem_grad("二维混合索引 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("二维混合索引 CUDA", input_np, index, device="cuda")
    
    def test_3d_multi_dim_indexing(self):
        """测试场景8: 三维数组的多维度索引"""
        input_np = np.random.randn(2, 3, 4).astype(np.float32)
        index = (0, 1, 2)
        
        # CPU场景测试
        self._test_getitem_grad("三维数组的多维度索引 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("三维数组的多维度索引 CUDA", input_np, index, device="cuda")
    
    def test_repeated_indexing(self):
        """测试场景9: 重复索引位置（检查梯度累加）"""
        input_np = np.random.randn(5).astype(np.float32)
        index = [0, 0, 2, 2]
        
        # CPU场景测试
        self._test_getitem_grad("重复索引位置 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("重复索引位置 CUDA", input_np, index, device="cuda")
    
    def test_negative_slice_indexing(self):
        """测试场景10: 负切片索引"""
        input_np = np.random.randn(10).astype(np.float32)
        index = slice(-5, -1)
        
        # CPU场景测试
        self._test_getitem_grad("负切片索引 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("负切片索引 CUDA", input_np, index, device="cuda")
    
    def test_dots_indexing(self):
        """测试场景11: dots索引测试"""
        # 测试三维数组的dots索引
        input_np = np.random.randn(2, 3, 4).astype(np.float32)
        index = (..., 2)  # 相当于 :, :, 2
        
        # CPU场景测试
        self._test_getitem_grad("dots索引测试 CPU", input_np, index, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_grad("dots索引测试 CUDA", input_np, index, device="cuda")
    
    def _test_3level_view_indexing_device(self, case_name, input_np, device="cpu"):
        """测试特定设备上的3级视图索引"""
        start_time = time.time()
        try:
            # 测试Riemann
            if device == "cpu":
                x_riemann = rm_tensor(input_np, requires_grad=True)
            else:  # cuda
                x_riemann = rm_tensor(input_np, requires_grad=True, device=device)
            
            # 创建3级视图
            view1 = x_riemann[:, 1:4, :]  # 第1级视图
            view2 = view1[1:3, :, 1:4]     # 第2级视图
            view3 = view2[:, 1:3, :]       # 第3级视图
            
            # 在3级视图上进行索引操作
            indexed_riemann = view3[0, 0, 0]
            indexed_riemann.backward()
            riemann_grad = x_riemann.grad
            
            # 测试PyTorch
            if TORCH_AVAILABLE:
                if device == "cpu":
                    x_torch = torch.tensor(input_np, requires_grad=True)
                else:  # cuda
                    x_torch = torch.tensor(input_np, requires_grad=True, device=device)
                
                # 创建3级视图
                torch_view1 = x_torch[:, 1:4, :]
                torch_view2 = torch_view1[1:3, :, 1:4]
                torch_view3 = torch_view2[:, 1:3, :]
                
                # 在3级视图上进行索引操作
                indexed_torch = torch_view3[0, 0, 0]
                indexed_torch.backward()
                torch_grad = x_torch.grad
            else:
                torch_grad = None
            
            # 比较结果
            passed = compare_values(riemann_grad, torch_grad)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"3级视图索引操作梯度计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_3level_view_indexing(self):
        """测试场景12: 3级视图上的索引测试"""
        # 创建输入数据
        input_np = np.random.randn(5, 5, 5).astype(np.float32)
        
        # CPU场景测试
        self._test_3level_view_indexing_device("3级视图索引测试 CPU", input_np, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_3level_view_indexing_device("3级视图索引测试 CUDA", input_np, device="cuda")
    
    def test_second_derivative_indexing(self):
        """测试场景13: 索引操作的二阶导数计算 (Hessian矩阵)"""
        input_np = np.array([1.0, 2.0], dtype=np.float32)
        
        # CPU场景测试
        self._test_getitem_second_grad("索引操作的二阶导数计算 CPU", input_np, device="cpu")
        
        # CUDA场景测试（如果可用）
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            self._test_getitem_second_grad("索引操作的二阶导数计算 CUDA", input_np, device="cuda")

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行索引操作梯度测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestGetitemGradFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)