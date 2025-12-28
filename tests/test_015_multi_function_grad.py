import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.autograd import grad as rm_grad
    RM_AVAILABLE = True
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    RM_AVAILABLE = False
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    from torch import autograd as torch_autograd
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的梯度计算")
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

# 定义测试函数
def cubic_function(x):
    return x**3 + 2*x**2 + 3*x + 4

def multi_var_function(x):
    # 假设x是形状为(2,)的向量
    return x[0]**2 + x[1]**3 + x[0]*x[1]

def trigonometric_function(x):
    # 检测是否在PyTorch环境中运行
    if 'torch' in str(type(x)):
        return torch.sin(x) + torch.cos(x)
    else:
        return rm.sin(x) + rm.cos(x)

def exponential_function(x):
    # 检测是否在PyTorch环境中运行
    if 'torch' in str(type(x)):
        return torch.exp(x) + x**2
    else:
        return rm.exp(x) + x**2

def logsoftmax_function(x):
    # 检测是否在PyTorch环境中运行
    if 'torch' in str(type(x)):
        return torch.nn.functional.log_softmax(x, dim=-1)
    else:
        return rm.nn.functional.log_softmax(x, dim=-1)

def setitem_function(x):
    # 检测是否在PyTorch环境中运行
    if 'torch' in str(type(x)):
        # 创建与x形状相同的张量，并使用索引原地赋值
        result = torch.zeros_like(x).clone()
        result[0] = torch.sin(x[0]) + torch.cos(x[1])
        result[1] = torch.exp(x[0] * x[1])
        return result.sum()
    else:
        result = rm.zeros_like(x).clone()
        result[0] = rm.sin(x[0]) + rm.cos(x[1])
        result[1] = rm.exp(x[0] * x[1])
        return result.sum()

class TestMultiFunctionGrad(unittest.TestCase):
    def setUp(self):
        # 设置随机种子以确保结果可重复
        np.random.seed(82)
        if TORCH_AVAILABLE:
            torch.manual_seed(82)
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
    
    def compute_multi_order_grads(self, func, x, max_order):
        """
        计算从1阶到max_order阶的导数
        Args:
            func: 要计算导数的函数
            x: 输入张量
            max_order: 最大导数阶数
        Returns:
            导数结果列表，包含1阶到max_order阶的导数
        """
        # 检查输入有效性
        if not RM_AVAILABLE:
            return None
            
        # 计算函数值
        y = func(x)
        
        # 存储各阶导数结果
        grads = []
        current_grad = y
        current_x = x
        
        # 逐阶计算导数
        for order in range(1, max_order + 1):
            # 计算当前阶导数
            # 注意：对于高阶导数，create_graph参数只在不是最后一阶时设置为True
            create_graph = (order < max_order)
            
            # 无论是否为多输出，都先确保current_grad是标量
            if hasattr(current_grad, 'sum'):
                current_grad = current_grad.sum()
            elif isinstance(current_grad, (list, tuple)):
                # 对于多输出情况，我们也需要将每个输出转换为标量
                sum_grads = []
                for g in current_grad:
                    if hasattr(g, 'sum'):
                        sum_grads.append(g.sum())
                    else:
                        sum_grads.append(g)
                current_grad = sum_grads[0] if len(sum_grads) == 1 else sum_grads
            
            # 现在current_grad应该是标量，可以安全地调用grad()
            if isinstance(current_grad, (list, tuple)):
                # 处理多输出情况
                grad_result = []
                for g in current_grad:
                    gradient = rm.autograd.grad(g, current_x, create_graph=create_graph)
                    grad_result.append(gradient[0] if gradient else None)
                current_grad = grad_result
            else:
                gradient = rm.autograd.grad(current_grad, current_x, create_graph=create_graph)
                current_grad = gradient[0] if gradient else None
            
            # 保存当前阶导数结果
            grads.append(current_grad)
            
            # 更新用于下一阶导数计算的输入
            current_x = x  # 始终对原始输入求导
            
        return grads
    
    def compute_torch_multi_order_grads(self, func, x, max_order):
        """
        使用PyTorch计算从1阶到max_order阶的导数
        Args:
            func: 要计算导数的函数
            x: 输入张量
            max_order: 最大导数阶数
        Returns:
            导数结果列表，包含1阶到max_order阶的导数
        """
        if not TORCH_AVAILABLE:
            return None
            
        # 计算函数值
        y = func(x)
        
        # 存储各阶导数结果
        grads = []
        current_grad = y
        current_x = x
        
        # 逐阶计算导数
        for order in range(1, max_order + 1):
            # 计算当前阶导数
            # 注意：对于高阶导数，create_graph参数只在不是最后一阶时设置为True
            create_graph = (order < max_order)
            
            # 无论是否为多输出，都先确保current_grad是标量
            if hasattr(current_grad, 'sum'):
                current_grad = current_grad.sum()
            elif isinstance(current_grad, (list, tuple)):
                # 对于多输出情况，我们也需要将每个输出转换为标量
                sum_grads = []
                for g in current_grad:
                    if hasattr(g, 'sum'):
                        sum_grads.append(g.sum())
                    else:
                        sum_grads.append(g)
                current_grad = sum_grads[0] if len(sum_grads) == 1 else sum_grads
            
            # 现在current_grad应该是标量，可以安全地调用grad()
            if isinstance(current_grad, (list, tuple)):
                # 处理多输出情况
                grad_result = []
                for g in current_grad:
                    gradient = torch.autograd.grad(g, current_x, create_graph=create_graph)
                    grad_result.append(gradient[0] if gradient else None)
                current_grad = grad_result
            else:
                gradient = torch.autograd.grad(current_grad, current_x, create_graph=create_graph)
                current_grad = gradient[0] if gradient else None
            
            # 保存当前阶导数结果
            grads.append(current_grad)
            
            # 更新用于下一阶导数计算的输入
            current_x = x  # 始终对原始输入求导
            
        return grads
    
    def compare_results(self, case_name, rm_result, torch_result, order, atol=1e-6, rtol=1e-6):
        """比较riemann和torch的结果是否一致"""
        if not TORCH_AVAILABLE:
            # 如果没有PyTorch，只检查riemann结果是否存在
            return rm_result is not None
        
        try:
            # 处理嵌套元组/列表的情况
            if isinstance(rm_result, (list, tuple)) and isinstance(torch_result, (list, tuple)):
                if len(rm_result) != len(torch_result):
                    raise AssertionError(f"结果长度不匹配: {len(rm_result)} vs {len(torch_result)}")
                
                all_passed = True
                for i, (r, t) in enumerate(zip(rm_result, torch_result)):
                    if not self.compare_results(f"{case_name}[{i}]", r, t, order, atol, rtol):
                        all_passed = False
                
                return all_passed
            
            # 比较数值 - 使用detach和numpy方法获取数据
            # 对于riemann张量
            if hasattr(rm_result, 'detach') and hasattr(rm_result.detach(), 'numpy'):
                rm_data = rm_result.detach().numpy()
            elif hasattr(rm_result, 'data'):
                rm_data = rm_result.data
            else:
                rm_data = rm_result
            
            # 对于torch张量
            if hasattr(torch_result, 'detach') and hasattr(torch_result.detach(), 'numpy'):
                torch_data = torch_result.detach().numpy()
            else:
                torch_data = torch_result
            
            # 处理标量情况
            if np.isscalar(rm_data) and np.isscalar(torch_data):
                passed = np.isclose(rm_data, torch_data, atol=atol, rtol=rtol)
                self.assertTrue(passed, f"{case_name} (第{order}阶导数) 数值不接近: {rm_data} vs {torch_data}")
                return passed
            
            # 检查形状是否匹配
            if hasattr(rm_data, 'shape') and hasattr(torch_data, 'shape'):
                self.assertEqual(rm_data.shape, torch_data.shape, 
                                f"{case_name} (第{order}阶导数) 形状不匹配: {rm_data.shape} vs {torch_data.shape}")
            
            # 检查数值是否接近
            np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
            return True
            
        except Exception as e:
            if isinstance(e, AssertionError):
                raise
            raise AssertionError(f"{case_name} (第{order}阶导数) 比较过程出错: {type(e).__name__}: {str(e)}")
    
    def _run_test_function(self, name, func, input_shape, max_order):
        """
        测试一个函数的多阶梯度 - 去掉test_前缀避免被unittest框架自动调用
        Args:
            name: 函数名称
            func: 要测试的函数
            input_shape: 输入张量的形状
            max_order: 最大导数阶数
        """
        if IS_RUNNING_AS_SCRIPT:
            print(f"\n{Colors.OKBLUE}开始测试函数: {name}{Colors.ENDC}")
            print(f"输入形状: {input_shape}")
            print(f"测试阶数: 1 到 {max_order}")
        
        # 生成测试数据
        input_data = np.random.randn(*input_shape).astype(np.float64)
        
        # 创建Riemann输入张量
        rm_x = rm.tensor(input_data, requires_grad=True)
        
        # 定义适用于PyTorch的函数版本
        def torch_func(x):
            # 如果函数在PyTorch中使用不同的API，这里可以进行适配
            return func(x)
        
        # 创建PyTorch输入张量
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(input_data, requires_grad=True, dtype=torch.float64)
        else:
            torch_x = None
        
        # 计算Riemann多阶梯度
        rm_grads = self.compute_multi_order_grads(func, rm_x, max_order)
        
        # 计算PyTorch多阶梯度
        if TORCH_AVAILABLE:
            torch_grads = self.compute_torch_multi_order_grads(torch_func, torch_x, max_order)
        else:
            torch_grads = [None] * max_order
        
        # 比较各阶导数结果
        for order in range(max_order):
            case_name = f"{name} (第{order + 1}阶导数)"
            start_time = time.time()
            try:
                passed = self.compare_results(name, rm_grads[order], torch_grads[order], order + 1)
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{'通过' if passed else '失败'}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"{case_name} 测试失败")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_cubic_function(self):
        """测试三次函数的多阶梯度"""
        self._run_test_function("三次函数", cubic_function, (5,), max_order=3)
    
    def test_multi_var_function(self):
        """测试多元函数的多阶梯度"""
        self._run_test_function("多元函数", multi_var_function, (2,), max_order=3)
    
    def test_trigonometric_function(self):
        """测试三角函数的多阶梯度"""
        self._run_test_function("三角函数", trigonometric_function, (3,), max_order=3)
    
    def test_exponential_function(self):
        """测试指数函数的多阶梯度"""
        self._run_test_function("指数函数", exponential_function, (4,), max_order=2)
    
    def test_logsoftmax_function(self):
        """测试logsoftmax函数的多阶梯度"""
        self._run_test_function("logsoftmax函数", logsoftmax_function, (5,), max_order=3)
    
    def test_setitem_function(self):
        """测试原地索引赋值函数的多阶梯度"""
        self._run_test_function("原地索引赋值", setitem_function, (2,), max_order=5)

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行多函数多阶梯度测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 支持: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    # 排除_run_test_function方法
    test_suite = unittest.TestSuite()
    test_methods = [method for method in dir(TestMultiFunctionGrad) if method.startswith('test_') and method != 'test_function']
    
    for method_name in test_methods:
        test_suite.addTest(TestMultiFunctionGrad(method_name))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)