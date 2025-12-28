import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann import tensor as rm_tensor
    TENSOR_AVAILABLE = True
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的原地索引赋值操作")
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
    if isinstance(rm_result, rm.TN):
        rm_data = rm_result.detach().data if hasattr(rm_result, 'detach') else rm_result.data
    elif hasattr(rm_result, 'data'):
        rm_data = rm_result.data
    else:
        rm_data = rm_result
    
    if hasattr(torch_result, 'detach'):
        torch_data = torch_result.detach().cpu().numpy()
    else:
        torch_data = torch_result
    
    # 处理形状不匹配的情况
    try:
        # 增加容差参数
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

class TestInplaceSetitemFunctions(unittest.TestCase):
    def setUp(self):
        # 设置随机种子以确保结果可重复
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
    
    def tearDown(self):
        # 清理资源
        pass
    
    def _test_inplace_setitem(self, case_name, input_nps, indices_ops):
        """通用的原地索引赋值测试方法"""
        start_time = time.time()
        try:
            # 正确调用start_function，只传递函数名
            if IS_RUNNING_AS_SCRIPT:
                stats.start_function(case_name)
            
            # 测试Riemann - 创建输入张量
            riemann_tensors = []
            for input_np in input_nps:
                riemann_tensors.append(rm_tensor(input_np, requires_grad=True))
            
            # 创建结果张量
            result_riemann = rm_tensor(np.zeros_like(riemann_tensors[0].data), requires_grad=True)
            result_riemann = result_riemann.clone()  # 克隆以支持梯度计算
            
            # 执行原地索引赋值操作
            for i, (idx, input_idx, slice_idx) in enumerate(indices_ops):
                # 确保右侧的值与左侧索引形状匹配
                value_tensor = riemann_tensors[input_idx]
                if slice_idx is not None:
                    value_tensor = value_tensor[slice_idx]
                result_riemann[idx] = value_tensor
                if i < len(indices_ops) - 1:  # 不是最后一次操作，需要克隆以支持梯度计算
                    result_riemann = result_riemann.clone()
            
            # 计算损失
            loss_riemann = result_riemann.sum()
            loss_riemann.backward()
            
            # 获取梯度
            riemann_grads = [tensor.grad for tensor in riemann_tensors]
            
            # 测试PyTorch
            torch_grads = None
            if TORCH_AVAILABLE:
                torch_tensors = []
                for input_np in input_nps:
                    torch_tensors.append(torch.tensor(input_np, requires_grad=True, dtype=torch.float32))
                
                # 创建结果张量
                result_torch = torch.zeros_like(torch_tensors[0], requires_grad=True)
                result_torch = result_torch.clone()  # 克隆以支持梯度计算
                
                # 执行原地索引赋值操作 - 确保与Riemann语法一致
                for i, (idx, input_idx, slice_idx) in enumerate(indices_ops):
                    # 确保右侧的值与左侧索引形状匹配
                    value_tensor = torch_tensors[input_idx]
                    if slice_idx is not None:
                        value_tensor = value_tensor[slice_idx]
                    result_torch[idx] = value_tensor
                    if i < len(indices_ops) - 1:  # 不是最后一次操作，需要克隆以支持梯度计算
                        result_torch = result_torch.clone()
                
                # 计算损失
                loss_torch = result_torch.sum()
                loss_torch.backward()
                
                # 获取梯度
                torch_grads = [tensor.grad for tensor in torch_tensors]
            
            # 比较结果
            passed = compare_values(riemann_grads, torch_grads)
            time_taken = time.time() - start_time
            
            # 正确调用end_function，只传递self参数
            if IS_RUNNING_AS_SCRIPT:
                stats.end_function()
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  值比较: 失败")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"原地索引赋值梯度计算结果不匹配: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            # 在异常处理中也正确调用end_function
            if IS_RUNNING_AS_SCRIPT:
                stats.end_function()
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_scalar_indexing(self):
        """测试场景1: 标量索引"""
        input_nps = [
            np.array([1.0, 2.0, 3.0]).astype(np.float32)
        ]
        # 修改为(索引, 输入张量索引, 输入张量切片索引)的格式
        indices_ops = [(0, 0, 0), (1, 0, 1), (2, 0, 2)]  # 标量索引
        self._test_inplace_setitem("标量索引", input_nps, indices_ops)
    
    def test_slice_indexing(self):
        """测试场景2: 切片索引"""
        input_nps = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
        ]
        # 确保切片形状匹配
        indices_ops = [(slice(0, 2), 0, slice(0, 2)), (slice(2, 5), 0, slice(2, 5))]
        self._test_inplace_setitem("切片索引", input_nps, indices_ops)
    
    def test_array_indexing(self):
        """测试场景3: 整数数组索引"""
        input_nps = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
        ]
        # 确保索引和切片形状匹配
        indices_ops = [([0, 2, 4], 0, [0, 2, 4]), ([1, 3], 0, [1, 3])]
        self._test_inplace_setitem("整数数组索引", input_nps, indices_ops)
    
    def test_basic_overlapping_indexing(self):
        """测试场景4: 基本重叠索引场景"""
        input_nps = [
            np.array([1.0, 2.0, 3.0]).astype(np.float32),
            np.array([4.0, 5.0]).astype(np.float32),
            np.array([6.0]).astype(np.float32)
        ]
        # 确保索引和切片形状匹配
        indices_ops = [([0, 1], 0, [0, 1]), ([1, 2], 1, None), ([2], 2, None)]
        self._test_inplace_setitem("基本重叠索引场景", input_nps, indices_ops)
    
    def test_complex_overlapping_indexing(self):
        """测试场景5: 复杂重叠索引场景"""
        input_nps = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32),
            np.array([5.0, 6.0, 7.0]).astype(np.float32),
            np.array([8.0, 9.0]).astype(np.float32)
        ]
        # 确保索引和切片形状匹配
        indices_ops = [
            ([0, 1, 2], 0, [0, 1, 2]),
            ([1, 2, 3], 1, None),
            ([2, 3, 4], 0, [1, 2, 3]),  # 确保形状为(3,)
            ([3, 4], 2, None)
        ]
        self._test_inplace_setitem("复杂重叠索引场景", input_nps, indices_ops)
    
    def test_2d_indexing(self):
        """测试场景6: 二维张量索引"""
        input_nps = [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 2).astype(np.float32)
        ]
        # 确保索引和切片形状匹配
        indices_ops = [
            ((slice(0, 2), 0), 0, (slice(0, 2), 0)),  # 提取第一列
            ((0, slice(0, 2)), 1, (0, slice(0, 2)))   # 提取第一行的前两列
        ]
        self._test_inplace_setitem("二维张量索引", input_nps, indices_ops)
    
    def test_multiple_overlapping_updates(self):
        """测试场景7: 多次重叠更新"""
        input_nps = [
            np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32),
            np.array([5.0, 6.0, 7.0]).astype(np.float32),
            np.array([8.0, 9.0]).astype(np.float32)
        ]
        # 确保索引和切片形状匹配
        indices_ops = [
            ([0, 1], 0, [0, 1]),
            ([1, 2], 1, [0, 1]),  # 使用前两个元素
            ([2, 3], 2, None),
            ([0, 2, 3], 0, [0, 2, 3])  # 使用第0,2,3个元素
        ]
        self._test_inplace_setitem("多次重叠更新", input_nps, indices_ops)
    
    def test_mixed_indexing_types(self):
        """测试场景8: 混合索引类型"""
        input_nps = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32),
            np.array([6.0, 7.0]).astype(np.float32)
        ]
        # 确保索引和切片形状匹配
        indices_ops = [
            (0, 0, 0),  # 标量索引
            (slice(1, 3), 0, slice(1, 3)),  # 切片索引
            ([3, 4], 1, None)  # 数组索引
        ]
        self._test_inplace_setitem("混合索引类型", input_nps, indices_ops)
    
    def test_repeated_overlapping_indices(self):
        """测试场景9: 重复重叠索引位置"""
        input_nps = [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0]).astype(np.float32),
            np.array([4.0]).astype(np.float32)
        ]
        # 确保索引和切片形状匹配
        indices_ops = [
            (0, 0, 0),  # 标量索引
            (0, 1, 0),  # 标量索引
            (0, 2, 0)   # 标量索引
        ]
        self._test_inplace_setitem("重复重叠索引位置", input_nps, indices_ops)
    
    def test_2d_mixed_indices_overlap(self):
        """测试场景10: 二维张量上的混合索引类型和重叠更新"""
        # 参考try.py中的测试场景，创建包含混合索引操作的测试用例
        input_nps = [
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).astype(np.float32),  # 3x3 张量
            np.array(0.0).astype(np.float32),                        # 标量v
            np.array([0.0, 0.0, 0.0]).astype(np.float32),            # 一维数组v1
            np.array([0.0, 0.0, 0.0]).astype(np.float32)             # 一维数组v2
        ]
        # 实现混合索引操作: 标量索引、行索引和列索引
        indices_ops = [
            ((1, 1), 1, None),       # y[1,1] = v (标量索引赋值)
            ((1, slice(None)), 2, None),  # y[1] = v1 (行索引赋值)
            ((slice(None), 1), 3, None)   # y[:,1] = v2 (列索引赋值)
        ]
        self._test_inplace_setitem("二维张量混合索引重叠更新", input_nps, indices_ops)
        
    def test_high_order_gradient(self):
        """测试场景11: 高阶导数"""
        start_time = time.time()
        try:
            # 创建输入张量
            input_np = np.array([2.0]).astype(np.float32)
            x_riemann = rm_tensor(input_np, requires_grad=True)
            
            # 使用原地索引操作构建计算图
            result_riemann = rm_tensor(np.zeros(2).astype(np.float32), requires_grad=True)
            result_riemann = result_riemann.clone()
            result_riemann[0] = x_riemann[0]  # 确保语法一致，使用标量索引
            result_riemann[1] = x_riemann[0]**2.
            
            # 计算一阶导数
            first_grad = rm.autograd.grad(result_riemann.sum(), x_riemann, create_graph=True, allow_unused=True)[0]
            
            # 计算二阶导数
            grad_outputs = rm_tensor([1.0])
            second_grad = rm.autograd.grad(first_grad, x_riemann, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
            
            # 测试PyTorch
            if TORCH_AVAILABLE:
                x_torch = torch.tensor(input_np, requires_grad=True, dtype=torch.float32)
                
                # 使用相同的操作 - 确保语法完全一致
                result_torch = torch.zeros(2, requires_grad=True)
                result_torch = result_torch.clone()
                result_torch[0] = x_torch[0]  # 使用标量索引
                result_torch[1] = x_torch[0]**2.
                
                # 计算一阶导数
                first_grad_torch = torch.autograd.grad(result_torch.sum(), x_torch, create_graph=True, allow_unused=True)[0]
                
                # 计算二阶导数
                second_grad_torch = torch.autograd.grad(first_grad_torch, x_torch, 
                                                    grad_outputs=torch.ones_like(first_grad_torch), 
                                                    create_graph=True,
                                                    allow_unused=True)[0]
                
                # 比较一阶导数
                passed_first = compare_values(first_grad, first_grad_torch)
                # 比较二阶导数
                passed_second = compare_values(second_grad, second_grad_torch)
                
                passed = passed_first and passed_second
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result("高阶导数-一阶", passed_first)
                    stats.add_result("高阶导数-二阶", passed_second)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: 高阶导数 - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                
                # 断言确保测试通过
                self.assertTrue(passed, "原地索引高阶导数计算结果不匹配")
            else:
                # 没有PyTorch时，至少确保Riemann能计算高阶导数
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result("高阶导数", True)
                    print(f"测试用例: 高阶导数 - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("高阶导数", False, [str(e)])
                print(f"测试用例: 高阶导数 - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise

# 主函数 - 作为独立脚本运行时执行
if __name__ == "__main__":
    IS_RUNNING_AS_SCRIPT = True
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行原地索引赋值测试 ====={Colors.ENDC}")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试框架: Riemann vs PyTorch")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestInplaceSetitemFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出
    result = runner.run(test_suite)
    
    # 打印统计摘要
    stats.print_summary()
    
    # 为独立脚本运行设置退出码
    if not result.wasSuccessful():
        sys.exit(1)