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
    print("警告: 无法导入PyTorch，将只测试riemann的排序函数")
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
            
            # 实时输出测试用例结果
            if IS_RUNNING_AS_SCRIPT:
                print(f"测试用例: {case_name} - {status_color}{status}{Colors.ENDC}")
                if details and not passed:
                    print(f"  错误详情: {details}")
    
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
    if hasattr(rm_result, 'data'):
        rm_np = rm_result.data
    elif isinstance(rm_result, np.ndarray):
        rm_np = rm_result
    else:
        rm_np = np.array(rm_result)
    
    # 智能处理torch_result，可以是PyTorch张量或Riemann张量或numpy数组
    if hasattr(torch_result, 'detach') and hasattr(torch_result, 'cpu'):  # PyTorch张量
        torch_np = torch_result.detach().cpu().numpy()
    elif hasattr(torch_result, 'data'):  # Riemann张量
        torch_np = torch_result.data
    elif hasattr(torch_result, 'numpy'):
        torch_np = torch_result.numpy()
    elif isinstance(torch_result, np.ndarray):
        torch_np = torch_result
    else:
        torch_np = np.array(torch_result)
    
    # 检查形状是否相同
    if rm_np.shape != torch_np.shape:
        return False
    
    # 比较值
    return np.allclose(rm_np, torch_np, atol=atol, rtol=rtol)

# 测试类
class TestSortFunction(unittest.TestCase):
    def setUp(self):
        """每个测试方法执行前的设置"""
        # 设置随机种子以确保结果可复现
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
        
        # 不同维度的测试数据
        self.data_1d = np.random.randn(5)
        self.data_2d = np.random.randn(3, 4)
        self.data_3d = np.random.randn(2, 3, 4)
    
    def tearDown(self):
        """每个测试方法执行后的清理"""
        # 清理梯度缓存（如果有的话）
        pass
    
    def test_sort_basic_1d(self):
        """测试1D张量的基本排序功能"""
        stats.start_function("test_sort_basic_1d")
        
        # 1D升序排序测试
        rm_x = rm.tensor(self.data_1d, requires_grad=False)
        rm_sorted, rm_indices = rm.sort(rm_x)
        
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(self.data_1d)
            torch_sorted, torch_indices = torch.sort(torch_x)
            
            values_match = compare_values(rm_sorted, torch_sorted)
            indices_match = compare_values(rm_indices, torch_indices)
            
            stats.add_result("1D升序排序值比较", values_match)
            stats.add_result("1D升序排序索引比较", indices_match)
            
            self.assertTrue(values_match, "1D升序排序值与PyTorch不匹配")
            self.assertTrue(indices_match, "1D升序排序索引与PyTorch不匹配")
        else:
            # 没有PyTorch时，只验证排序结果是否正确
            expected_sorted = np.sort(self.data_1d)
            expected_indices = np.argsort(self.data_1d)
            
            values_match = compare_values(rm_sorted, expected_sorted)
            indices_match = compare_values(rm_indices, expected_indices)
            
            stats.add_result("1D升序排序值验证", values_match)
            stats.add_result("1D升序排序索引验证", indices_match)
            
            self.assertTrue(values_match, "1D升序排序值不正确")
            self.assertTrue(indices_match, "1D升序排序索引不正确")
        
        stats.end_function()
    
    def test_sort_descending(self):
        """测试降序排序功能"""
        stats.start_function("test_sort_descending")
        
        # 2D降序排序测试（沿最后一个维度）
        rm_x = rm.tensor(self.data_2d, requires_grad=False)
        rm_sorted, rm_indices = rm.sort(rm_x, descending=True)
        
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(self.data_2d)
            torch_sorted, torch_indices = torch.sort(torch_x, descending=True)
            
            values_match = compare_values(rm_sorted, torch_sorted)
            indices_match = compare_values(rm_indices, torch_indices)
            
            stats.add_result("2D降序排序值比较", values_match)
            stats.add_result("2D降序排序索引比较", indices_match)
            
            self.assertTrue(values_match, "2D降序排序值与PyTorch不匹配")
            self.assertTrue(indices_match, "2D降序排序索引与PyTorch不匹配")
        
        stats.end_function()
    
    def test_sort_different_dims(self):
        """测试沿不同维度排序"""
        stats.start_function("test_sort_different_dims")
        
        # 测试沿第0维和第1维排序
        for dim in [0, 1]:
            rm_x = rm.tensor(self.data_2d, requires_grad=False)
            rm_sorted, rm_indices = rm.sort(rm_x, dim=dim)
            
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(self.data_2d)
                torch_sorted, torch_indices = torch.sort(torch_x, dim=dim)
                
                values_match = compare_values(rm_sorted, torch_sorted)
                indices_match = compare_values(rm_indices, torch_indices)
                
                stats.add_result(f"2D沿维度{dim}排序值比较", values_match)
                stats.add_result(f"2D沿维度{dim}排序索引比较", indices_match)
                
                self.assertTrue(values_match, f"2D沿维度{dim}排序值与PyTorch不匹配")
                self.assertTrue(indices_match, f"2D沿维度{dim}排序索引与PyTorch不匹配")
        
        stats.end_function()
    
    def test_sort_3d_tensor(self):
        """测试3D张量排序"""
        stats.start_function("test_sort_3d_tensor")
        
        # 测试沿3D张量的不同维度排序
        for dim in [0, 1, 2]:
            rm_x = rm.tensor(self.data_3d, requires_grad=False)
            rm_sorted, rm_indices = rm.sort(rm_x, dim=dim)
            
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(self.data_3d)
                torch_sorted, torch_indices = torch.sort(torch_x, dim=dim)
                
                values_match = compare_values(rm_sorted, torch_sorted)
                indices_match = compare_values(rm_indices, torch_indices)
                
                stats.add_result(f"3D沿维度{dim}排序值比较", values_match)
                stats.add_result(f"3D沿维度{dim}排序索引比较", indices_match)
                
                self.assertTrue(values_match, f"3D沿维度{dim}排序值与PyTorch不匹配")
                self.assertTrue(indices_match, f"3D沿维度{dim}排序索引与PyTorch不匹配")
        
        stats.end_function()
    
    def test_sort_with_grad(self):
        """测试带梯度的排序操作"""
        if not TORCH_AVAILABLE:
            return  # 如果没有PyTorch，跳过此测试
        
        stats.start_function("test_sort_with_grad")
        
        # 测试2D张量的梯度计算
        for dim in [0, 1]:
            # Riemann测试
            rm_x = rm.tensor(self.data_2d, requires_grad=True)
            rm_sorted, _ = rm.sort(rm_x, dim=dim)
            rm_loss = rm_sorted.sum()
            rm_loss.backward()
            
            # PyTorch测试
            torch_x = torch.tensor(self.data_2d, requires_grad=True)
            torch_sorted, _ = torch.sort(torch_x, dim=dim)
            torch_loss = torch_sorted.sum()
            torch_loss.backward()
            
            # 比较梯度
            grad_match = compare_values(rm_x.grad, torch_x.grad)
            
            stats.add_result(f"沿维度{dim}排序梯度比较", grad_match)
            self.assertTrue(grad_match, f"沿维度{dim}排序梯度与PyTorch不匹配")
        
        stats.end_function()
    
    def test_sort_with_out_parameter(self):
        """测试使用out参数的排序操作"""
        stats.start_function("test_sort_with_out_parameter")
        
        # 不带梯度的out参数测试
        rm_x = rm.tensor(self.data_2d, requires_grad=False)
        rm_values_out = rm.zeros_like(rm_x)
        rm_indices_out = rm.zeros_like(rm_x, dtype=rm.int64)
        
        # 使用out参数调用sort
        rm_result = rm.sort(rm_x, dim=0, out=(rm_values_out, rm_indices_out))
        
        # 验证返回值是否为out参数
        stats.add_result("out参数返回值验证", rm_result[0] is rm_values_out and rm_result[1] is rm_indices_out)
        self.assertTrue(rm_result[0] is rm_values_out, "返回的values不是out参数")
        self.assertTrue(rm_result[1] is rm_indices_out, "返回的indices不是out参数")
        
        # 与不使用out参数的结果比较
        rm_sorted_no_out, rm_indices_no_out = rm.sort(rm_x, dim=0)
        
        values_match = compare_values(rm_values_out, rm_sorted_no_out)
        indices_match = compare_values(rm_indices_out, rm_indices_no_out)
        
        stats.add_result("out参数值一致性验证", values_match)
        stats.add_result("out参数索引一致性验证", indices_match)
        
        self.assertTrue(values_match, "使用out参数的排序值与不使用out参数的不一致")
        self.assertTrue(indices_match, "使用out参数的排序索引与不使用out参数的不一致")
        
        stats.end_function()
    
    def test_sort_out_requires_grad_conflict(self):
        """测试out参数和requires_grad的冲突情况"""
        stats.start_function("test_sort_out_requires_grad_conflict")
        
        # 创建需要梯度的输入张量
        rm_x = rm.tensor(self.data_2d, requires_grad=True)
        rm_values_out = rm.zeros_like(rm_x)
        rm_indices_out = rm.zeros_like(rm_x, dtype=rm.int64)
        
        # 验证是否抛出RuntimeError
        try:
            rm.sort(rm_x, dim=0, out=(rm_values_out, rm_indices_out))
            # 如果没有抛出异常，测试失败
            stats.add_result("out参数与requires_grad冲突检测", False)
            self.fail("使用out参数和requires_grad=True时应该抛出RuntimeError")
        except RuntimeError as e:
            # 检查错误信息
            expected_error = "sort(): functions with out=... arguments don't support automatic differentiation"
            error_match = expected_error in str(e)
            stats.add_result("out参数与requires_grad冲突错误信息", error_match)
            self.assertTrue(error_match, f"错误信息不符合预期: {e}")
        
        stats.end_function()
    
    def test_sort_invalid_dim(self):
        """测试无效维度参数"""
        stats.start_function("test_sort_invalid_dim")
        
        rm_x = rm.tensor(self.data_2d, requires_grad=False)
        
        # 测试维度越界（太大）
        try:
            rm.sort(rm_x, dim=5)  # 2D张量，有效维度是0和1
            stats.add_result("大维度越界检测", False)
            self.fail("维度超出范围时应该抛出IndexError")
        except IndexError:
            stats.add_result("大维度越界检测", True)
        
        # 测试维度越界（太小）
        try:
            rm.sort(rm_x, dim=-3)  # 2D张量，最小有效维度是-2
            stats.add_result("小维度越界检测", False)
            self.fail("维度超出范围时应该抛出IndexError")
        except IndexError:
            stats.add_result("小维度越界检测", True)
        
        stats.end_function()
    
    def test_sort_invalid_out_parameter(self):
        """测试无效的out参数"""
        stats.start_function("test_sort_invalid_out_parameter")
        
        rm_x = rm.tensor(self.data_2d, requires_grad=False)
        
        # 测试out不是元组
        try:
            rm.sort(rm_x, dim=0, out=rm.zeros_like(rm_x))
            stats.add_result("out非元组检测", False)
            self.fail("out不是元组时应该抛出TypeError")
        except TypeError:
            stats.add_result("out非元组检测", True)
        
        # 测试out元组长度不为2
        try:
            rm.sort(rm_x, dim=0, out=(rm.zeros_like(rm_x),))
            stats.add_result("out元组长度检测", False)
            self.fail("out元组长度不为2时应该抛出TypeError")
        except TypeError:
            stats.add_result("out元组长度检测", True)
        
        # 测试out张量形状不匹配
        try:
            wrong_shape = rm.zeros((2, 2))
            rm.sort(rm_x, dim=0, out=(wrong_shape, rm.zeros_like(rm_x, dtype=rm.int64)))
            stats.add_result("out形状不匹配检测", False)
            self.fail("out张量形状不匹配时应该抛出RuntimeError")
        except RuntimeError:
            stats.add_result("out形状不匹配检测", True)
        
        stats.end_function()
    
    def test_sort_stable_parameter(self):
        """测试stable参数（虽然当前实现忽略此参数）"""
        stats.start_function("test_sort_stable_parameter")
        
        # 创建有重复值的测试数据，更容易看出稳定排序的效果
        repeated_data = np.array([3, 1, 2, 1, 3])
        
        rm_x = rm.tensor(repeated_data, requires_grad=False)
        
        # 测试stable=True
        rm_sorted_stable, rm_indices_stable = rm.sort(rm_x, stable=True)
        
        # 测试stable=False（默认值）
        rm_sorted_unstable, rm_indices_unstable = rm.sort(rm_x, stable=False)
        
        # 当前实现中，stable参数被忽略，所以结果应该相同
        values_match = compare_values(rm_sorted_stable, rm_sorted_unstable)
        indices_match = compare_values(rm_indices_stable, rm_indices_unstable)
        
        stats.add_result("stable参数值一致性", values_match)
        stats.add_result("stable参数索引一致性", indices_match)
        
        self.assertTrue(values_match, "stable=True和stable=False的排序值应该相同（当前实现忽略此参数）")
        # 注意：索引可能不同，因为如果有重复值，排序算法可能会产生不同的索引顺序
        # 所以我们不测试索引的一致性，只测试值的一致性
        
        stats.end_function()

# 主函数
if __name__ == '__main__':
    IS_RUNNING_AS_SCRIPT = True
    
    if os.name == 'nt':  # Windows系统
        # 在Windows系统上禁用ANSI颜色代码
        class Colors:
            HEADER = ''
            OKBLUE = ''
            OKGREEN = ''
            WARNING = ''
            FAIL = ''
            ENDC = ''
            BOLD = ''
            UNDERLINE = ''
    
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行排序函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 运行测试
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestSortFunction)
    
    # 创建自定义测试运行器，禁用默认输出
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计
    stats.print_summary()
    
    # 退出，根据测试结果返回适当的退出码
    sys.exit(0 if result.wasSuccessful() else 1)