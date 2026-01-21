import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
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
    print("警告: 无法导入PyTorch，将只测试riemann的backward函数")
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


def has_compute_graph(tsobj:rm.TN):
    """检查张量是否包含计算图的辅助函数"""
    if tsobj.requires_grad :
        if tsobj.gradfuncs and tsobj.gradfuncs != ():
            return True
    return False

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

class TestBackwardFunction(unittest.TestCase):
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
    
    def test_basic_backward_scalar(self):
        """测试场景1: 基本的标量反向传播"""
        test_cases = [
            {"name": "基本的标量反向传播", "x_np": 2.0}
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建输入
                    x_np = case["x_np"]
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_x = rm.tensor(x_np, requires_grad=True)
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True)
                        else:
                            torch_x = None
                    else:  # cuda
                        rm_x = rm.tensor(x_np, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True, device=device)
                        else:
                            torch_x = None
                    
                    # 计算输出
                    rm_y = rm_x ** 2.
                    if TORCH_AVAILABLE:
                        torch_y = torch_x ** 2.
                    
                    # 执行反向传播
                    rm_y.backward()
                    if TORCH_AVAILABLE:
                        torch_y.backward()
                    
                    # 获取梯度
                    grad_x_riemann = rm_x.grad
                    if TORCH_AVAILABLE:
                        grad_x_torch = torch_x.grad
                    else:
                        grad_x_torch = None
                    
                    # 比较结果
                    passed = compare_values(grad_x_riemann, grad_x_torch)
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: 失败")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"反向传播结果不匹配: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_basic_backward_tensor(self):
        """测试场景2: 基本的张量反向传播"""
        test_cases = [
            {
                "name": "基本的张量反向传播", 
                "x_np": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), 
                "grad_output_np": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建输入
                    x_np = case["x_np"]
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_x = rm.tensor(x_np, requires_grad=True)
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True)
                        else:
                            torch_x = None
                    else:  # cuda
                        rm_x = rm.tensor(x_np, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True, device=device)
                        else:
                            torch_x = None
                    
                    # 创建梯度输出
                    grad_output_np = case["grad_output_np"]
                    if device == "cpu":
                        rm_grad_output = rm.tensor(grad_output_np)
                        if TORCH_AVAILABLE:
                            torch_grad_output = torch.tensor(grad_output_np)
                        else:
                            torch_grad_output = None
                    else:  # cuda
                        rm_grad_output = rm.tensor(grad_output_np, device=device)
                        if TORCH_AVAILABLE:
                            torch_grad_output = torch.tensor(grad_output_np, device=device)
                        else:
                            torch_grad_output = None
                    
                    # 计算输出
                    rm_y = rm_x ** 2.
                    if TORCH_AVAILABLE:
                        torch_y = torch_x ** 2.
                    
                    # 执行反向传播
                    rm_y.backward(gradient=rm_grad_output)
                    if TORCH_AVAILABLE:
                        torch_y.backward(gradient=torch_grad_output)
                    
                    # 获取梯度
                    grad_x_riemann = rm_x.grad
                    if TORCH_AVAILABLE:
                        grad_x_torch = torch_x.grad
                    else:
                        grad_x_torch = None
                    
                    # 比较结果
                    passed = compare_values(grad_x_riemann, grad_x_torch)
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: 失败")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"反向传播结果不匹配: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_backward_create_graph_effect(self):
        """测试场景3.1: 验证create_graph=True对梯度计算图的影响"""
        test_cases = [
            {"name": "验证create_graph=True对梯度计算图的影响", "x_np": 2.0}
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建输入 - 只需要一个用于create_graph=True测试的张量
                    x_np = case["x_np"]
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_x = rm.tensor(x_np, requires_grad=True)  # 用于create_graph=True测试
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True)
                        else:
                            torch_x = None
                    else:  # cuda
                        rm_x = rm.tensor(x_np, requires_grad=True, device=device)  # 用于create_graph=True测试
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True, device=device)
                        else:
                            torch_x = None
                    
                    # 测试create_graph=True时，梯度中包含计算图
                    # Riemann测试
                    rm_y = rm_x ** 3.
                    rm_y.backward(create_graph=True)
                    rm_grad = rm_x.grad
                    
                    # 检查使用create_graph=True的梯度是否能再次求导
                    rm_can_backward = False
                    try:
                        # 尝试对梯度再次求导
                        rm_grad.backward()
                        rm_can_backward = True  # 如果没有异常，则说明可以再次求导
                    except Exception:
                        rm_can_backward = False  # 捕获异常，说明不能再次求导
                    
                    # PyTorch部分测试
                    if TORCH_AVAILABLE:
                        # 使用torch.autograd.grad代替backward()以避免警告
                        torch_y = torch_x ** 3.
                        torch_grad = torch.autograd.grad(torch_y, torch_x, create_graph=True)[0]
                        
                        torch_can_backward = False
                        try:
                            # 尝试对梯度再次求导
                            torch_grad.backward()
                            torch_can_backward = True
                        except Exception:
                            torch_can_backward = False
                        
                        # 验证PyTorch和Riemann行为一致
                        torch_passed = torch_can_backward
                        rm_passed = rm_can_backward
                        passed = rm_passed and (not TORCH_AVAILABLE or torch_passed)
                        
                        # 清理PyTorch梯度
                        torch_x.grad = None
                    else:
                        # 只有Riemann测试
                        passed = rm_can_backward
                    
                    # 清理Riemann梯度
                    rm_x.grad = None
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  create_graph=True梯度可求导: {rm_can_backward} (预期: True)")
                            if TORCH_AVAILABLE:
                                print(f"  PyTorch create_graph=True梯度可求导: {torch_can_backward} (预期: True)")
                    
                    # 断言确保测试通过
                    self.assertTrue(rm_can_backward, "create_graph=True时，梯度应该可以再次求导")
                    if TORCH_AVAILABLE:
                        self.assertTrue(torch_can_backward, "PyTorch create_graph=True时，梯度应该可以再次求导")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
            if hasattr(rm_x, 'grad'):
                rm_x.grad = None

    def test_backward_check_grad_graph_properties(self):
        """测试场景3: 检查grad_graph属性"""
        start_time = time.time()
        passed = True
        failed_reasons = []
        
        try:
            # 创建输入
            x_np = 2.0
            rm_x1 = rm.tensor(x_np, requires_grad=True)  # 用于create_graph=False测试
            rm_x2 = rm.tensor(x_np, requires_grad=True)  # 用于create_graph=True测试
            
            if TORCH_AVAILABLE:
                torch_x1 = torch.tensor(x_np, requires_grad=True)
                torch_x2 = torch.tensor(x_np, requires_grad=True)
            else:
                torch_x1 = None
                torch_x2 = None
            
            # 1. 测试create_graph=False时的梯度属性
            # Riemann测试
            rm_y1 = rm_x1 ** 3.0
            rm_y1.backward(create_graph=False)  # 默认值
            rm_grad1 = rm_x1.grad
            
            # 检查属性 - create_graph=False时，梯度不应包含计算图
            rm_has_compute_graph_false = has_compute_graph(rm_grad1)
            rm_requires_grad_false = rm_grad1.requires_grad
            # 检查叶子节点属性
            rm_is_leaf_false = getattr(rm_grad1, 'is_leaf', True)  # 默认假设是叶子节点
            
            # 2. 测试create_graph=True时的梯度属性
            # Riemann测试
            rm_y2 = rm_x2 ** 3.0
            rm_y2.backward(create_graph=True)
            rm_grad2 = rm_x2.grad
            
            # 检查属性 - create_graph=True时，梯度应包含计算图
            rm_has_compute_graph_true = has_compute_graph(rm_grad2)
            rm_requires_grad_true = rm_grad2.requires_grad
            # 检查叶子节点属性
            rm_is_leaf_true = getattr(rm_grad2, 'is_leaf', True)  # 默认假设是叶子节点
            
            # PyTorch部分测试
            if TORCH_AVAILABLE:
                # 1. 测试create_graph=False时的梯度属性
                torch_y1 = torch_x1 ** 3.0
                torch_y1.backward(create_graph=False)
                torch_grad1 = torch_x1.grad
                
                torch_has_grad_fn_false = hasattr(torch_grad1, 'grad_fn') and torch_grad1.grad_fn is not None
                torch_requires_grad_false = torch_grad1.requires_grad
                # 检查叶子节点属性
                torch_is_leaf_false = torch_grad1.is_leaf
                
                # 测试create_graph=True时的梯度属性
                torch_y2 = torch_x2 ** 3.0
                torch_grad2 = torch.autograd.grad(torch_y2, torch_x2, create_graph=True)[0]
                
                torch_has_grad_fn_true = hasattr(torch_grad2, 'grad_fn') and torch_grad2.grad_fn is not None
                torch_requires_grad_true = torch_grad2.requires_grad
                # 检查叶子节点属性
                torch_is_leaf_true = torch_grad2.is_leaf
                
                # 验证属性检查结果
                print(f"PyTorch create_graph=False - requires_grad: {torch_requires_grad_false}, has_grad_fn: {torch_has_grad_fn_false}, is_leaf: {torch_is_leaf_false}")
                print(f"PyTorch create_graph=True - requires_grad: {torch_requires_grad_true}, has_grad_fn: {torch_has_grad_fn_true}, is_leaf: {torch_is_leaf_true}")
                
                # 断言PyTorch行为符合预期
                if torch_requires_grad_false or torch_has_grad_fn_false:
                    failed_reasons.append("PyTorch create_graph=False时，梯度不应包含计算图")
                    passed = False
                
                if not torch_requires_grad_true or not torch_has_grad_fn_true:
                    failed_reasons.append("PyTorch create_graph=True时，梯度应包含计算图")
                    passed = False
                
                # 检查叶子节点属性一致性
                # 在PyTorch中，create_graph=False时梯度应该是叶子节点
                if not torch_is_leaf_false:
                    failed_reasons.append("PyTorch create_graph=False时，梯度应是叶子节点")
                    passed = False
                
                # 在PyTorch中，create_graph=True时梯度不应是叶子节点
                if torch_is_leaf_true:
                    failed_reasons.append("PyTorch create_graph=True时，梯度不应是叶子节点")
                    passed = False
                    
                # 清理PyTorch梯度
                torch_x1.grad = None
                torch_x2.grad = None
            
            # 验证Riemann行为
            print(f"Riemann create_graph=False - requires_grad: {rm_requires_grad_false}, has_compute_graph: {rm_has_compute_graph_false}, is_leaf: {rm_is_leaf_false}")
            print(f"Riemann create_graph=True - requires_grad: {rm_requires_grad_true}, has_compute_graph: {rm_has_compute_graph_true}, is_leaf: {rm_is_leaf_true}")
            
            # 断言Riemann行为符合预期
            if rm_requires_grad_false or rm_has_compute_graph_false:
                failed_reasons.append("Riemann create_graph=False时，梯度不应包含计算图")
                passed = False
            
            if not rm_requires_grad_true or not rm_has_compute_graph_true:
                failed_reasons.append("Riemann create_graph=True时，梯度应包含计算图")
                passed = False
            
            # 检查叶子节点属性
            # 对于Riemann，如果支持is_leaf属性，检查其与PyTorch的一致性
            if TORCH_AVAILABLE:
                if hasattr(rm_grad1, 'is_leaf') and hasattr(rm_grad2, 'is_leaf'):
                    # 检查与PyTorch叶子节点属性的一致性
                    if rm_is_leaf_false != torch_is_leaf_false:
                        failed_reasons.append(f"Riemann和PyTorch在create_graph=False时的叶子节点属性不一致: Riemann={rm_is_leaf_false}, PyTorch={torch_is_leaf_false}")
                        passed = False
                    
                    if rm_is_leaf_true != torch_is_leaf_true:
                        failed_reasons.append(f"Riemann和PyTorch在create_graph=True时的叶子节点属性不一致: Riemann={rm_is_leaf_true}, PyTorch={torch_is_leaf_true}")
                        passed = False
                    
                    # 检查逻辑一致性
                    # create_graph=False时梯度应该是叶子节点
                    if not rm_is_leaf_false:
                        failed_reasons.append("Riemann create_graph=False时，梯度应是叶子节点")
                        passed = False
                    
                    # create_graph=True时梯度不应是叶子节点
                    if rm_is_leaf_true:
                        failed_reasons.append("Riemann create_graph=True时，梯度不应是叶子节点")
                        passed = False
                
                # 如果Riemann不支持is_leaf属性，记录信息
                if not hasattr(rm_grad1, 'is_leaf') and not hasattr(rm_grad2, 'is_leaf'):
                    print("注意: Riemann张量当前不支持is_leaf属性")
                
        except Exception as e:
            passed = False
            failed_reasons.append(f"测试过程中出现异常: {str(e)}")
        finally:
            # 清理Riemann梯度
            rm_x1.grad = None
            rm_x2.grad = None
        
        time_taken = time.time() - start_time
        print(f"测试耗时: {time_taken:.4f}秒")
        
        # 输出测试结果
        if passed:
            print(f"{Colors.OKGREEN}测试通过!{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}测试失败!{Colors.ENDC} 失败原因:")
            for reason in failed_reasons:
                print(f"  - {reason}")
        
        # 记录测试结果
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result("backward_check_grad_graph_properties", passed)
        
        # 断言测试通过
        self.assertTrue(passed, "梯度计算图属性检查测试失败")

    def test_backward_gradient_accumulation(self):
        """测试场景4: 梯度累加功能"""
        test_cases = [
            {
                "name": "梯度累加功能", 
                "x_np": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建输入
                    x_np = case["x_np"]
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_x = rm.tensor(x_np, requires_grad=True)
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True)
                        else:
                            torch_x = None
                    else:  # cuda
                        rm_x = rm.tensor(x_np, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True, device=device)
                        else:
                            torch_x = None
                    
                    # 第一次计算和反向传播
                    rm_y1 = rm_x ** 2
                    rm_y1_sum = rm_y1.sum()
                    rm_y1_sum.backward()
                    
                    if TORCH_AVAILABLE:
                        torch_y1 = torch_x ** 2
                        torch_y1_sum = torch_y1.sum()
                        torch_y1_sum.backward()
                    
                    # 第二次计算和反向传播
                    rm_y2 = rm_x * 3
                    rm_y2_sum = rm_y2.sum()
                    rm_y2_sum.backward()
                    
                    if TORCH_AVAILABLE:
                        torch_y2 = torch_x * 3
                        torch_y2_sum = torch_y2.sum()
                        torch_y2_sum.backward()
                    
                    # 获取累积梯度
                    grad_x_riemann = rm_x.grad
                    if TORCH_AVAILABLE:
                        grad_x_torch = torch_x.grad
                    else:
                        grad_x_torch = None
                    
                    # 比较结果
                    passed = compare_values(grad_x_riemann, grad_x_torch)
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: 失败")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"梯度累加结果不匹配: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_backward_multi_path_accumulation(self):
        """测试场景5: 从两个张量沿着不同计算路径backward，同一个依赖变量的梯度会累加"""
        test_cases = [
            {
                "name": "多路径梯度累加", 
                "x_np": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建共享输入
                    x_np = case["x_np"]
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_x = rm.tensor(x_np, requires_grad=True)
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True)
                        else:
                            torch_x = None
                    else:  # cuda
                        rm_x = rm.tensor(x_np, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True, device=device)
                        else:
                            torch_x = None
                    
                    # 计算路径1
                    rm_y1 = rm_x ** 2
                    rm_y1_sum = rm_y1.sum()
                    
                    # 计算路径2
                    rm_y2 = rm_x * 3
                    rm_y2_sum = rm_y2.sum()
                    
                    if TORCH_AVAILABLE:
                        torch_y1 = torch_x ** 2
                        torch_y1_sum = torch_y1.sum()
                        torch_y2 = torch_x * 3
                        torch_y2_sum = torch_y2.sum()
                    
                    # 分别执行反向传播
                    rm_y1_sum.backward()
                    rm_y2_sum.backward()
                    
                    if TORCH_AVAILABLE:
                        torch_y1_sum.backward()
                        torch_y2_sum.backward()
                    
                    # 获取累积梯度
                    grad_x_riemann = rm_x.grad
                    if TORCH_AVAILABLE:
                        grad_x_torch = torch_x.grad
                    else:
                        grad_x_torch = None
                    
                    # 比较结果
                    passed = compare_values(grad_x_riemann, grad_x_torch)
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: 失败")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"多路径梯度累加结果不匹配: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_backward_complicated_graph(self):
        """测试场景6: 复杂计算图的反向传播"""
        test_cases = [
            {
                "name": "复杂计算图的反向传播", 
                "x_np": np.array([1.0, 2.0, 3.0], dtype=np.float64),
                "y_np": np.array([0.5, 1.5, 2.5], dtype=np.float64)
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建输入
                    x_np = case["x_np"]
                    y_np = case["y_np"]
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_x = rm.tensor(x_np, requires_grad=True)
                        rm_y = rm.tensor(y_np, requires_grad=True)
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True)
                            torch_y = torch.tensor(y_np, requires_grad=True)
                        else:
                            torch_x, torch_y = None, None
                    else:  # cuda
                        rm_x = rm.tensor(x_np, requires_grad=True, device=device)
                        rm_y = rm.tensor(y_np, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            torch_x = torch.tensor(x_np, requires_grad=True, device=device)
                            torch_y = torch.tensor(y_np, requires_grad=True, device=device)
                        else:
                            torch_x, torch_y = None, None
                    
                    # 构建复杂计算图
                    rm_z1 = rm_x * rm_y
                    rm_z2 = rm_x ** 2
                    rm_z3 = rm_z1 + rm_z2
                    rm_z4 = rm_z3.mean()
                    
                    if TORCH_AVAILABLE:
                        torch_z1 = torch_x * torch_y
                        torch_z2 = torch_x ** 2
                        torch_z3 = torch_z1 + torch_z2
                        torch_z4 = torch_z3.mean()
                    
                    # 执行反向传播
                    rm_z4.backward()
                    if TORCH_AVAILABLE:
                        torch_z4.backward()
                    
                    # 获取梯度
                    grad_x_riemann = rm_x.grad
                    grad_y_riemann = rm_y.grad
                    
                    if TORCH_AVAILABLE:
                        grad_x_torch = torch_x.grad
                        grad_y_torch = torch_y.grad
                    else:
                        grad_x_torch, grad_y_torch = None, None
                    
                    # 比较结果
                    rm_result = (grad_x_riemann, grad_y_riemann)
                    torch_result = (grad_x_torch, grad_y_torch)
                    passed = compare_values(rm_result, torch_result)
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: 失败")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"复杂计算图反向传播结果不匹配: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_backward_with_retain_grad(self):
        """测试场景7: 测试retain_grad参数"""
        test_cases = [
            {"name": "测试retain_grad参数", "x_np": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建输入
                    x_np = case["x_np"]
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_x = rm.tensor(x_np, requires_grad=True)
                    else:  # cuda
                        rm_x = rm.tensor(x_np, requires_grad=True, device=device)
                    
                    # 创建中间变量并设置retain_grad
                    rm_y = rm_x ** 2
                    rm_y.retain_grad()
                    rm_z = rm_y.sum()
                    
                    # 执行反向传播
                    rm_z.backward()
                    
                    # 验证中间变量梯度存在
                    has_middle_grad = rm_y.grad is not None
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, has_middle_grad)
                        status = "通过" if has_middle_grad else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if has_middle_grad else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not has_middle_grad:
                            print(f"  中间变量梯度不存在")
                    
                    # 断言确保测试通过
                    self.assertTrue(has_middle_grad, f"retain_grad功能不正常: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_backward_invalid_cases(self):
        """测试场景8: 测试无效的反向传播情况"""
        test_cases = [
            {"name": "测试无效的反向传播情况", "x_np": 2.0, "y_np": np.array([1.0, 2.0], dtype=np.float64), "a_np": 3.0}
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 测试叶子节点调用backward
                    x_np = case["x_np"]
                    if device == "cpu":
                        rm_x = rm.tensor(x_np, requires_grad=False)
                    else:  # cuda
                        rm_x = rm.tensor(x_np, requires_grad=False, device=device)
                    
                    # 应该抛出RuntimeError
                    with self.assertRaises(RuntimeError):
                        rm_x.backward()
                    
                    # 测试非标量且无gradient参数调用backward
                    y_np = case["y_np"]
                    if device == "cpu":
                        rm_y = rm.tensor(y_np, requires_grad=False)
                    else:  # cuda
                        rm_y = rm.tensor(y_np, requires_grad=False, device=device)
                    rm_z = rm_y * 2  # 这仍然是叶子节点
                    
                    with self.assertRaises(RuntimeError):
                        rm_z.backward()
                    
                    # 测试不需要梯度的张量调用backward
                    a_np = case["a_np"]
                    if device == "cpu":
                        rm_a = rm.tensor(a_np, requires_grad=False)
                    else:  # cuda
                        rm_a = rm.tensor(a_np, requires_grad=False, device=device)
                    rm_b = rm_a * 2  # 不需要梯度
                    
                    with self.assertRaises(RuntimeError):
                        rm_b.backward()
                    
                    passed = True
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

# 当作为独立脚本运行时的处理逻辑
def run_tests():
    global IS_RUNNING_AS_SCRIPT
    IS_RUNNING_AS_SCRIPT = True
    
    clear_screen()
    print(f"{Colors.HEADER}========== 开始测试 backward 函数 =========={Colors.ENDC}")
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBackwardFunction)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 使用自定义输出
    result = runner.run(suite)
    
    # 打印统计摘要
    stats.print_summary()
    
    # 返回测试结果状态
    return result.wasSuccessful()

# 当作为独立脚本运行时执行测试
if __name__ == "__main__":
    run_tests()