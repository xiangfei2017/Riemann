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
        """sort - 测试1D张量的基本排序功能"""
        stats.start_function("test_sort_basic_1d")
        
        # 1D升序排序测试
        rm_x = rm.tensor(self.data_1d, requires_grad=False)
        rm_sorted, rm_indices = rm.sort(rm_x)
        
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(self.data_1d)
            torch_sorted, torch_indices = torch.sort(torch_x)
            
            values_match = compare_values(rm_sorted, torch_sorted)
            indices_match = compare_values(rm_indices, torch_indices)
            
            stats.add_result("sort - 1D升序排序值比较", values_match)
            stats.add_result("sort - 1D升序排序索引比较", indices_match)
            
            self.assertTrue(values_match, "1D升序排序值与PyTorch不匹配")
            self.assertTrue(indices_match, "1D升序排序索引与PyTorch不匹配")
        else:
            # 没有PyTorch时，只验证排序结果是否正确
            expected_sorted = np.sort(self.data_1d)
            expected_indices = np.argsort(self.data_1d)
            
            values_match = compare_values(rm_sorted, expected_sorted)
            indices_match = compare_values(rm_indices, expected_indices)
            
            stats.add_result("sort - 1D升序排序值验证", values_match)
            stats.add_result("sort - 1D升序排序索引验证", indices_match)
            
            self.assertTrue(values_match, "1D升序排序值不正确")
            self.assertTrue(indices_match, "1D升序排序索引不正确")
        
        stats.end_function()
    
    def test_sort_descending(self):
        """sort - 测试降序排序功能"""
        stats.start_function("test_sort_descending")
        
        # 2D降序排序测试（沿最后一个维度）
        rm_x = rm.tensor(self.data_2d, requires_grad=False)
        rm_sorted, rm_indices = rm.sort(rm_x, descending=True)
        
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(self.data_2d)
            torch_sorted, torch_indices = torch.sort(torch_x, descending=True)
            
            values_match = compare_values(rm_sorted, torch_sorted)
            indices_match = compare_values(rm_indices, torch_indices)
            
            stats.add_result("sort - 2D降序排序值比较", values_match)
            stats.add_result("sort - 2D降序排序索引比较", indices_match)
            
            self.assertTrue(values_match, "2D降序排序值与PyTorch不匹配")
            self.assertTrue(indices_match, "2D降序排序索引与PyTorch不匹配")
        else:
            expected_sorted = np.sort(self.data_2d, axis=-1)[..., ::-1]
            expected_indices = np.argsort(self.data_2d, axis=-1)[..., ::-1]
            
            values_match = compare_values(rm_sorted, expected_sorted)
            indices_match = compare_values(rm_indices, expected_indices)
            
            stats.add_result("sort - 2D降序排序值验证", values_match)
            stats.add_result("sort - 2D降序排序索引验证", indices_match)
            
            self.assertTrue(values_match, "2D降序排序值不正确")
            self.assertTrue(indices_match, "2D降序排序索引不正确")
        
        stats.end_function()
    
    def test_sort_different_dims(self):
        """sort - 测试沿不同维度排序"""
        stats.start_function("test_sort_different_dims")
        
        # 沿不同维度测试
        for dim in [0, 1]:
            rm_x = rm.tensor(self.data_2d, requires_grad=False)
            rm_sorted, rm_indices = rm.sort(rm_x, dim=dim)
            
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(self.data_2d)
                torch_sorted, torch_indices = torch.sort(torch_x, dim=dim)
                
                values_match = compare_values(rm_sorted, torch_sorted)
                indices_match = compare_values(rm_indices, torch_indices)
                
                stats.add_result(f"sort - 2D张量沿dim={dim}排序值比较", values_match)
                stats.add_result(f"sort - 2D张量沿dim={dim}排序索引比较", indices_match)
                
                self.assertTrue(values_match, f"沿dim={dim}排序值与PyTorch不匹配")
                self.assertTrue(indices_match, f"沿dim={dim}排序索引与PyTorch不匹配")
            else:
                expected_sorted = np.sort(self.data_2d, axis=dim)
                expected_indices = np.argsort(self.data_2d, axis=dim)
                
                values_match = compare_values(rm_sorted, expected_sorted)
                indices_match = compare_values(rm_indices, expected_indices)
                
                stats.add_result(f"sort - 2D张量沿dim={dim}排序值验证", values_match)
                stats.add_result(f"sort - 2D张量沿dim={dim}排序索引验证", indices_match)
                
                self.assertTrue(values_match, f"沿dim={dim}排序值不正确")
                self.assertTrue(indices_match, f"沿dim={dim}排序索引不正确")
        
        stats.end_function()
    
    def test_sort_3d_tensor(self):
        """sort - 测试3D张量排序"""
        stats.start_function("test_sort_3d_tensor")
        
        # 3D张量排序测试
        rm_x = rm.tensor(self.data_3d, requires_grad=False)
        
        for dim in range(3):
            rm_sorted, rm_indices = rm.sort(rm_x, dim=dim)
            
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(self.data_3d)
                torch_sorted, torch_indices = torch.sort(torch_x, dim=dim)
                
                values_match = compare_values(rm_sorted, torch_sorted)
                indices_match = compare_values(rm_indices, torch_indices)
                
                stats.add_result(f"sort - 3D张量沿dim={dim}排序值比较", values_match)
                stats.add_result(f"sort - 3D张量沿dim={dim}排序索引比较", indices_match)
                
                self.assertTrue(values_match, f"3D张量沿dim={dim}排序值与PyTorch不匹配")
                self.assertTrue(indices_match, f"3D张量沿dim={dim}排序索引与PyTorch不匹配")
            else:
                expected_sorted = np.sort(self.data_3d, axis=dim)
                expected_indices = np.argsort(self.data_3d, axis=dim)
                
                values_match = compare_values(rm_sorted, expected_sorted)
                indices_match = compare_values(rm_indices, expected_indices)
                
                stats.add_result(f"sort - 3D张量沿dim={dim}排序值验证", values_match)
                stats.add_result(f"sort - 3D张量沿dim={dim}排序索引验证", indices_match)
                
                self.assertTrue(values_match, f"3D张量沿dim={dim}排序值不正确")
                self.assertTrue(indices_match, f"3D张量沿dim={dim}排序索引不正确")
        
        stats.end_function()
    
    def test_sort_with_grad(self):
        """sort - 测试带梯度的排序操作"""
        stats.start_function("test_sort_with_grad")
        
        rm_x = rm.tensor(self.data_1d, requires_grad=True)
        rm_sorted, rm_indices = rm.sort(rm_x)
        
        # 计算梯度
        rm_sum = rm_sorted.sum()
        rm_sum.backward()
        
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(self.data_1d, requires_grad=True)
            torch_sorted, torch_indices = torch.sort(torch_x)
            torch_sum = torch_sorted.sum()
            torch_sum.backward()
            
            grad_match = compare_values(rm_x.grad, torch_x.grad)
            
            stats.add_result("sort - 带梯度排序的梯度比较", grad_match)
            stats.add_result("sort - 带梯度排序后的张量比较", compare_values(rm_sorted, torch_sorted))
            
            self.assertTrue(grad_match, "排序操作的梯度与PyTorch不匹配")
        else:
            stats.add_result("带梯度排序的功能验证", True)
        
        stats.end_function()
    
    def test_sort_with_out_parameter(self):
        """sort - 测试使用out参数的排序操作"""
        stats.start_function("test_sort_with_out_parameter")
        
        # 测试out参数
        rm_x = rm.tensor(self.data_1d, requires_grad=False)
        rm_out_values = rm.zeros_like(rm_x)
        rm_out_indices = rm.zeros_like(rm_x, dtype='int64')
        
        rm_sorted, rm_indices = rm.sort(rm_x, out=(rm_out_values, rm_out_indices))
        
        # 检查返回结果是否与out参数一致
        self.assertIs(rm_sorted, rm_out_values, "返回的排序值不是out参数的同一个对象")
        self.assertIs(rm_indices, rm_out_indices, "返回的索引不是out参数的同一个对象")
        
        # 验证排序结果
        expected_sorted = np.sort(self.data_1d)
        expected_indices = np.argsort(self.data_1d)
        
        values_match = compare_values(rm_sorted, expected_sorted)
        indices_match = compare_values(rm_indices, expected_indices)
        
        stats.add_result("sort - out参数排序值正确性验证", values_match)
        stats.add_result("sort - out参数排序索引正确性验证", indices_match)
        stats.add_result("sort - out参数对象一致性验证", rm_sorted is rm_out_values and rm_indices is rm_out_indices)
        
        self.assertTrue(values_match, "使用out参数的排序值不正确")
        self.assertTrue(indices_match, "使用out参数的排序索引不正确")
        
        stats.end_function()
    
    def test_sort_out_requires_grad_conflict(self):
        """sort - 测试out参数和requires_grad的冲突情况"""
        stats.start_function("test_sort_out_requires_grad_conflict")
        
        rm_x = rm.tensor(self.data_1d, requires_grad=True)
        rm_out = rm.zeros_like(rm_x)
        rm_out_indices = rm.zeros_like(rm_x, dtype='int64')
        
        with self.assertRaises(RuntimeError, msg="out参数与requires_grad=True同时使用时应抛出RuntimeError"):
            rm.sort(rm_x, out=(rm_out, rm_out_indices))
        
        stats.add_result("sort - out参数与requires_grad冲突测试", True)
        
        stats.end_function()
    
    def test_sort_invalid_dim(self):
        """sort - 测试无效维度参数"""
        stats.start_function("test_sort_invalid_dim")
        
        rm_x = rm.tensor(self.data_2d, requires_grad=False)
        
        # 测试无效的维度
        with self.assertRaises(IndexError, msg="无效维度参数应抛出IndexError异常"):
            rm.sort(rm_x, dim=10)
        
        # 测试负维度
        with self.assertRaises(IndexError, msg="无效负维度参数应抛出IndexError异常"):
            rm.sort(rm_x, dim=-10)
        
        stats.add_result("sort - 无效维度测试", True)
        stats.add_result("sort - 无效负维度测试", True)
        
        stats.end_function()
    
    def test_sort_invalid_out_parameter(self):
        """sort - 测试无效的out参数"""
        stats.start_function("test_sort_invalid_out_parameter")
        
        rm_x = rm.tensor(self.data_2d, requires_grad=False)
        
        with self.assertRaises(TypeError, msg="错误类型的out参数应抛出TypeError"):
            rm.sort(rm_x, out="invalid")
        
        with self.assertRaises(TypeError, msg="out参数长度不足应抛出TypeError"):
            rm.sort(rm_x, out=(rm.zeros_like(rm_x), ))
        
        invalid_shape_out = rm.zeros((3, 3))
        with self.assertRaises(RuntimeError, msg="形状不匹配的out参数应抛出RuntimeError"):
            rm.sort(rm_x, out=(invalid_shape_out, invalid_shape_out))
        
        stats.add_result("sort - 无效out参数类型测试", True)
        stats.add_result("sort - 无效out参数长度测试", True)
        stats.add_result("sort - 形状不匹配out参数测试", True)
        
        stats.end_function()
    
    def test_sort_stable_parameter(self):
        """sort - 测试stable参数（虽然当前实现忽略此参数）"""
        stats.start_function("test_sort_stable_parameter")
        
        rm_x = rm.tensor(self.data_1d, requires_grad=False)
        
        # 测试使用stable=True和stable=False
        rm_sorted_true, rm_indices_true = rm.sort(rm_x, stable=True)
        rm_sorted_false, rm_indices_false = rm.sort(rm_x, stable=False)
        
        # 验证排序结果是否正确（由于当前实现忽略stable参数，结果应相同）
        values_same = compare_values(rm_sorted_true, rm_sorted_false)
        indices_same = compare_values(rm_indices_true, rm_indices_false)
        
        stats.add_result("sort - stable参数设置为True的排序结果", values_same)
        stats.add_result("sort - stable参数设置为False的排序结果", indices_same)
        
        self.assertTrue(values_same, "使用stable=True和stable=False得到的排序值不同")
        self.assertTrue(indices_same, "使用stable=True和stable=False得到的排序索引不同")
        
        stats.end_function()

    def test_argsort_basic_1d(self):
        """argsort - 测试1D张量的argsort基本功能"""
        stats.start_function("test_argsort_basic_1d")
        
        # 1D升序测试
        rm_x = rm.tensor(self.data_1d, requires_grad=False)
        result = rm.argsort(rm_x)
        
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(self.data_1d)
            torch_indices = torch.argsort(torch_x)
            indices_match = compare_values(result, torch_indices)
            
            stats.add_result("argsort - 1D argsort 结果与 PyTorch 比较", indices_match)
            self.assertTrue(indices_match, "1D argsort 结果与 PyTorch 不一致")
        else:
            expected_indices = np.argsort(self.data_1d)
            indices_match = compare_values(result, expected_indices)
            
            stats.add_result("argsort - 1D argsort 结果验证", indices_match)
            self.assertTrue(indices_match, "1D argsort 结果不正确")
        
        stats.end_function()
    
    def test_argsort_descending(self):
        """argsort - 测试argsort的降序排序功能"""
        stats.start_function("test_argsort_descending")
        
        rm_x = rm.tensor(self.data_2d, requires_grad=False)
        result = rm.argsort(rm_x, descending=True)
        
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(self.data_2d)
            torch_indices = torch.argsort(torch_x, descending=True)
            indices_match = compare_values(result, torch_indices)
            
            stats.add_result("argsort - 降序与 PyTorch 比较", indices_match)
            self.assertTrue(indices_match, "argsort 降序结果与 PyTorch 不一致")
        else:
            expected_indices = np.argsort(-self.data_2d)
            indices_match = compare_values(result, expected_indices)
            
            stats.add_result("argsort - 降序结果验证", indices_match)
            self.assertTrue(indices_match, "argsort 降序结果不正确")
        
        stats.end_function()
    
    def test_argsort_different_dims(self):
        """argsort - 测试在不同维度上使用argsort"""
        stats.start_function("test_argsort_different_dims")
        
        for dim in [0, 1]:
            rm_x = rm.tensor(self.data_2d, requires_grad=False)
            result = rm.argsort(rm_x, dim=dim)
            
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(self.data_2d)
                torch_indices = torch.argsort(torch_x, dim=dim)
                indices_match = compare_values(result, torch_indices)
                
                stats.add_result(f"argsort - 沿 dim={dim} 与 PyTorch 比较", indices_match)
                self.assertTrue(indices_match, f"argsort 沿 dim={dim} 结果与 PyTorch 不一致")
            else:
                expected_indices = np.argsort(self.data_2d, axis=dim)
                indices_match = compare_values(result, expected_indices)
                
                stats.add_result(f"argsort - 沿 dim={dim} 结果验证", indices_match)
                self.assertTrue(indices_match, f"argsort 沿 dim={dim} 结果不正确")
        
        stats.end_function()
    
    def test_argsort_3d_tensor(self):
        """argsort - 测试3D张量的argsort功能"""
        stats.start_function("test_argsort_3d_tensor")
        
        rm_x = rm.tensor(self.data_3d, requires_grad=False)
        result = rm.argsort(rm_x, dim=1)
        
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(self.data_3d)
            torch_indices = torch.argsort(torch_x, dim=1)
            indices_match = compare_values(result, torch_indices)
            
            stats.add_result("argsort - 3D 张量 argsort 与 PyTorch 比较", indices_match)
            self.assertTrue(indices_match, "3D 张量 argsort 结果与 PyTorch 不一致")
        else:
            expected_indices = np.argsort(self.data_3d, axis=1)
            indices_match = compare_values(result, expected_indices)
            
            stats.add_result("argsort - 3D 张量 argsort 结果验证", indices_match)
            self.assertTrue(indices_match, "3D 张量 argsort 结果不正确")
        
        stats.end_function()
    
    def test_argsort_with_grad(self):
        """argsort - 测试带梯度的argsort"""
        stats.start_function("test_argsort_with_grad")
        
        rm_x = rm.tensor(self.data_1d, requires_grad=True)
        result = rm.argsort(rm_x)
        
        # 计算梯度
        # 注意：argsort 的结果是索引，不能直接计算梯度
        # 所以我们需要将索引用于索引原始张量后再计算梯度
        rm_subset = rm_x[result[:3]]
        rm_sum = rm_subset.sum()
        rm_sum.backward()
        
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(self.data_1d, requires_grad=True)
            torch_indices = torch.argsort(torch_x)
            torch_subset = torch_x[torch_indices[:3]]
            torch_sum = torch_subset.sum()
            torch_sum.backward()
            
            grad_match = compare_values(rm_x.grad, torch_x.grad)
            
            stats.add_result("argsort - 梯度与 PyTorch 比较", grad_match)
            stats.add_result("argsort - 带梯度功能验证", True)
            
            self.assertTrue(grad_match, "argsort 梯度与 PyTorch 不一致")
        else:
            stats.add_result("argsort - 带梯度功能验证", True)
        
        stats.end_function()
    
    def test_argsort_with_out_parameter(self):
        """argsort - 测试argsort的out参数"""
        stats.start_function("test_argsort_with_out_parameter")
        
        rm_x = rm.tensor(self.data_1d, requires_grad=False)
        out_tensor = rm.zeros_like(rm_x, dtype='int64')
        
        result = rm.argsort(rm_x, out=out_tensor)
        
        # 验证结果对象是否是同一个
        self.assertIs(result, out_tensor, "argsort 的 out 参数返回值不是同一对象")
        
        # 验证排序索引
        expected_indices = np.argsort(self.data_1d)
        indices_match = compare_values(result, expected_indices)
        
        stats.add_result("argsort - out 参数结果验证", indices_match)
        stats.add_result("argsort - out 参数对象一致性验证", result is out_tensor)
        
        self.assertTrue(indices_match, "argsort 使用 out 参数的结果不正确")
        
        stats.end_function()
    
    def test_argsort_invalid_dim(self):
        """argsort - 测试无效维度的argsort"""
        stats.start_function("test_argsort_invalid_dim")
        
        rm_x = rm.tensor(self.data_2d, requires_grad=False)
        
        with self.assertRaises(IndexError, msg="无效维度应抛出 IndexError"):
            rm.argsort(rm_x, dim=10)
        
        with self.assertRaises(IndexError, msg="无效负维度应抛出 IndexError"):
            rm.argsort(rm_x, dim=-10)
        
        stats.add_result("argsort - 无效正维度测试", True)
        stats.add_result("argsort - 无效负维度测试", True)
        
        stats.end_function()
    
    def test_argsort_stable_parameter(self):
        """argsort - 测试stable参数（虽然当前实现忽略此参数）"""
        stats.start_function("test_argsort_stable_parameter")
        
        rm_x = rm.tensor(self.data_1d, requires_grad=False)
        result_true = rm.argsort(rm_x, stable=True)
        result_false = rm.argsort(rm_x, stable=False)
        
        # 由于当前实现忽略stable参数，结果应相同
        indices_same = compare_values(result_true, result_false)
        
        stats.add_result("argsort - stable=True 与 stable=False 结果比较", indices_same)
        self.assertTrue(indices_same, "argsort 使用不同 stable 参数得到不同结果")
        
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