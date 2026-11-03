import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann import TN
    from riemann.autograd import gradcheck
    from riemann.tensordef import track_grad
    from riemann.gradmode import set_grad_enabled
    from riemann.autograd.grad import Function  # 正确导入Function类
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 可选依赖：PyTorch 用于梯度对比；CUDA 可用性用于扩展设备覆盖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
CUDA_AVAILABLE = getattr(getattr(rm, "cuda", None), "CUPY_AVAILABLE", False)

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
# 判断是否作为独立脚本运行（仅控制过程打印）
IS_SCRIPT_MODE = __name__ == '__main__'

# 测试自定义梯度功能类
class TestDefineGradFunc(unittest.TestCase):
    """测试自定义梯度功能，包括track_grad修饰器和Function类"""
    
    def setUp(self):
        """测试前的准备工作"""
        np.random.seed(42)
        self.current_test_name = self._testMethodName
        # 统计在两种运行模式下均完整工作；过程打印仅在脚本模式下输出
        stats.start_function(self.current_test_name)
        if IS_SCRIPT_MODE:
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")
            print(f"{Colors.OKBLUE}测试目标: {self.current_test_name}{Colors.ENDC}")
    
    def tearDown(self):
        """测试结束后的清理工作"""
        stats.end_function()
        if IS_SCRIPT_MODE:
            print(f"\n{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_track_grad_decorator(self):
        """测试 track_grad 修饰器：参数组合、返回值类型、矩阵乘与重复反向"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        # ==================== 被测函数（grad_func 签名为 (result, ...)） ====================
        # 1. 单输入：sin
        def _sin_derivative(result, x):
            return (result.grad_value * x.cos(),)

        @track_grad(_sin_derivative)
        def mysin(x):
            return rm.tensor(np.sin(x.detach().array()), device=x.device)

        # 2. 多输入：加法
        def _add_derivative(result, x, y):
            return (result.grad_value, result.grad_value)

        @track_grad(_add_derivative)
        def myadd(x, y):
            return rm.tensor(x.detach().array() + y.detach().array(), device=x.device)

        # 3. 除法：两个输入的梯度值不同，可校验参数与梯度的对应关系
        def _divide_derivative(result, x, y):
            return (result.grad_value / y, result.grad_value * (-x / (y * y)))

        @track_grad(_divide_derivative)
        def mydivide(x, y):
            return x / y

        # 4. 非 TN 参数在前
        def _scale_derivative(result, scale, x):
            return (result.grad_value * scale,)

        @track_grad(_scale_derivative)
        def myscale(scale, x):
            return rm.tensor(scale * x.detach().array(), device=x.device)

        # 5. TN 与非 TN 参数交替
        def _weighted_sum_derivative(result, a, weight1, b, weight2):
            return (result.grad_value * weight1, result.grad_value * weight2)

        @track_grad(_weighted_sum_derivative)
        def weighted_sum(a, weight1, b, weight2):
            return rm.tensor(weight1 * a.detach().array() + weight2 * b.detach().array(), device=a.device)

        # 6. 带默认参数
        def _pow_derivative(result, x, exponent=2.0):
            local = exponent * (x.detach().array() ** (exponent - 1))
            return (result.grad_value * rm.tensor(local, device=x.device),)

        @track_grad(_pow_derivative)
        def mypow(x, exponent=2.0):
            return rm.tensor(x.detach().array() ** exponent, device=x.device)

        # 7. grad_func 使用 *args
        def _sum_derivative(result, *args):
            return tuple(result.grad_value for _ in args)

        @track_grad(_sum_derivative)
        def mysum(*xs):
            return rm.tensor(sum(x.detach().array() for x in xs), device=xs[0].device)

        # 8. grad_func 使用 **kwargs
        def _sum_kwargs_derivative(result, **kwargs):
            return tuple(result.grad_value for _ in kwargs.values())

        @track_grad(_sum_kwargs_derivative)
        def mysum_kwargs(**kwargs):
            vals = list(kwargs.values())
            return rm.tensor(sum(v.detach().array() for v in vals), device=vals[0].device)

        # 9. 矩阵乘法：反向需要矩阵乘 @ 与转置
        def _matmul_derivative(result, x, y):
            g = result.grad_value
            return (g @ y.detach().T, x.detach().T @ g)

        @track_grad(_matmul_derivative)
        def mymatmul(x, y):
            return rm.matmul(x, y)

        # 10. 前向返回非 TN（numpy 数组），由修饰器自动转换为 TN
        def _double_derivative(result, x):
            return (result.grad_value * 2.0,)

        @track_grad(_double_derivative)
        def mydouble(x):
            return 2.0 * x.detach().array()

        # 11. grad_func 首参为裸 *args（args[0] 即 result）
        def _sum_bare_args_derivative(*args):
            result = args[0]
            return tuple(result.grad_value for _ in args[1:])

        @track_grad(_sum_bare_args_derivative)
        def mysum_bare(*xs):
            return rm.tensor(sum(x.detach().array() for x in xs), device=xs[0].device)

        # ==================== 辅助函数 ====================
        def to_np(t):
            return t.detach().cpu().numpy()

        def make(data, device, requires_grad=True):
            return rm.tensor(np.array(data, dtype=np.float32), requires_grad=requires_grad, device=device)

        # ==================== 各场景用例 ====================
        def case_single_input(device):
            np_x = np.random.randn(3, 4).astype(np.float32)
            x = make(np_x, device)
            y = mysin(x)
            forward_ok = np.allclose(to_np(y), np.sin(np_x), atol=1e-5)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(x.grad), np.cos(np_x), atol=1e-5)

        def case_multi_positional(device):
            np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            np_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
            a, b = make(np_a, device), make(np_b, device)
            y = myadd(a, b)
            forward_ok = np.allclose(to_np(y), np_a + np_b)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(a.grad), np.ones_like(np_a)) \
                and np.allclose(to_np(b.grad), np.ones_like(np_b))

        def case_multi_keyword(device):
            np_a = np.array([1.0, 2.0], dtype=np.float32)
            np_b = np.array([3.0, 4.0], dtype=np.float32)
            a, b = make(np_a, device), make(np_b, device)
            y = myadd(x=a, y=b)
            forward_ok = np.allclose(to_np(y), np_a + np_b)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(a.grad), np.ones_like(np_a)) \
                and np.allclose(to_np(b.grad), np.ones_like(np_b))

        def case_keyword_order(device):
            # 关键字顺序与签名 (x, y) 相反，梯度仍应按参数名对应
            np_a = np.array([1.0, 2.0], dtype=np.float32)
            np_b = np.array([3.0, 4.0], dtype=np.float32)
            a, b = make(np_a, device), make(np_b, device)
            y = myadd(y=b, x=a)
            forward_ok = np.allclose(to_np(y), np_a + np_b)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(a.grad), np.ones_like(np_a)) \
                and np.allclose(to_np(b.grad), np.ones_like(np_b))

        def case_divide(device):
            x = rm.tensor(5.0, requires_grad=True, device=device)
            y = rm.tensor(2.0, requires_grad=True, device=device)
            z = mydivide(x, y)
            forward_ok = np.allclose(to_np(z), 2.5)
            z.backward()
            return forward_ok and np.allclose(to_np(x.grad), 0.5) \
                and np.allclose(to_np(y.grad), -1.25)

        def case_non_tn_before(device):
            np_x = np.array([2.0, 3.0], dtype=np.float32)
            x = make(np_x, device)
            y = myscale(3.0, x)
            forward_ok = np.allclose(to_np(y), 3.0 * np_x)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(x.grad), np.full_like(np_x, 3.0))

        def case_mixed_tn_non_tn(device):
            np_a = np.array([1.0, 2.0], dtype=np.float32)
            np_b = np.array([3.0, 4.0], dtype=np.float32)
            a, b = make(np_a, device), make(np_b, device)
            w1, w2 = 2.0, 3.0
            y = weighted_sum(a, w1, b, w2)
            forward_ok = np.allclose(to_np(y), w1 * np_a + w2 * np_b)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(a.grad), np.full_like(np_a, w1)) \
                and np.allclose(to_np(b.grad), np.full_like(np_b, w2))

        def case_partial_requires_grad(device):
            np_a = np.array([1.0, 2.0], dtype=np.float32)
            np_b = np.array([3.0, 4.0], dtype=np.float32)
            a = make(np_a, device, requires_grad=True)
            b = make(np_b, device, requires_grad=False)
            y = myadd(a, b)
            y.sum().backward()
            return np.allclose(to_np(a.grad), np.ones_like(np_a)) and b.grad is None

        def case_default_param(device):
            np_x = np.array([2.0, 3.0], dtype=np.float32)
            x = make(np_x, device)
            y = mypow(x)   # 使用默认 exponent=2.0
            forward_ok = np.allclose(to_np(y), np_x ** 2)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(x.grad), 2.0 * np_x)

        def case_override_default(device):
            np_x = np.array([2.0, 3.0], dtype=np.float32)
            x = make(np_x, device)
            y = mypow(x, exponent=3.0)   # 覆盖默认值
            forward_ok = np.allclose(to_np(y), np_x ** 3)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(x.grad), 3.0 * np_x ** 2)

        def case_varargs(device):
            np_a = np.array([1.0, 2.0], dtype=np.float32)
            np_b = np.array([3.0, 4.0], dtype=np.float32)
            np_c = np.array([5.0, 6.0], dtype=np.float32)
            a, b, c = make(np_a, device), make(np_b, device), make(np_c, device)
            y = mysum(a, b, c)
            forward_ok = np.allclose(to_np(y), np_a + np_b + np_c)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(a.grad), np.ones_like(np_a)) \
                and np.allclose(to_np(b.grad), np.ones_like(np_b)) \
                and np.allclose(to_np(c.grad), np.ones_like(np_c))

        def case_kwargs(device):
            np_a = np.array([1.0, 2.0], dtype=np.float32)
            np_b = np.array([3.0, 4.0], dtype=np.float32)
            a, b = make(np_a, device), make(np_b, device)
            y = mysum_kwargs(a=a, b=b)
            forward_ok = np.allclose(to_np(y), np_a + np_b)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(a.grad), np.ones_like(np_a)) \
                and np.allclose(to_np(b.grad), np.ones_like(np_b))

        def case_non_tn_keyword(device):
            np_x = np.array([2.0, 3.0], dtype=np.float32)
            x = make(np_x, device)
            y = myscale(scale=4.0, x=x)   # 非 TN 参数通过关键字传入
            forward_ok = np.allclose(to_np(y), 4.0 * np_x)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(x.grad), np.full_like(np_x, 4.0))

        def case_matmul(device):
            np_x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np_y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            x, y = make(np_x, device), make(np_y, device)
            z = mymatmul(x, y)
            forward_ok = np.allclose(to_np(z), np_x @ np_y)
            z.sum().backward()
            # 上游梯度全 1：dx = 1 @ y^T，dy = x^T @ 1
            ones = np.ones((2, 2), dtype=np.float32)
            dx_expected = ones @ np_y.T
            dy_expected = np_x.T @ ones
            return forward_ok and np.allclose(to_np(x.grad), dx_expected) \
                and np.allclose(to_np(y.grad), dy_expected)

        def case_non_tn_return(device):
            # 前向返回 numpy 而不是 TN，修饰器应自动转换并建立计算图
            x = rm.tensor(2.0, requires_grad=True, device=device)
            y = mydouble(x)
            forward_ok = np.allclose(to_np(y), 4.0)
            y.backward()
            return forward_ok and np.allclose(to_np(x.grad), 2.0)

        def case_repeat_backward(device):
            # grad_func 依赖 result.grad_value，第二次反向必须重新计算而非复用缓存
            x = rm.tensor(2.0, requires_grad=True, device=device)
            y = mypow(x)               # y = x^2
            y.backward()               # 上游梯度 1 → dx = 2x = 4
            first_ok = np.allclose(to_np(x.grad), 4.0)
            x.grad = None
            y.backward(gradient=rm.tensor(3.0, device=device), retain_graph=True)
            second_ok = np.allclose(to_np(x.grad), 12.0)   # 3 * 2x = 12
            return first_ok and second_ok

        def case_pytorch_compare(device):
            np_a = np.random.randn(3, 4).astype(np.float32)
            np_b = np.random.randn(3, 4).astype(np.float32)

            a, b = make(np_a, device), make(np_b, device)
            myadd(a, b).sum().backward()

            torch_a = torch.tensor(np_a.copy(), requires_grad=True)
            torch_b = torch.tensor(np_b.copy(), requires_grad=True)
            (torch_a + torch_b).sum().backward()

            return np.allclose(to_np(a.grad), torch_a.grad.detach().numpy(), atol=1e-5, rtol=1e-5) \
                and np.allclose(to_np(b.grad), torch_b.grad.detach().numpy(), atol=1e-5, rtol=1e-5)

        def case_bare_varargs(device):
            np_a = np.array([1.0, 2.0], dtype=np.float32)
            np_b = np.array([3.0, 4.0], dtype=np.float32)
            a, b = make(np_a, device), make(np_b, device)
            y = mysum_bare(a, b)
            forward_ok = np.allclose(to_np(y), np_a + np_b)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(a.grad), np.ones_like(np_a)) \
                and np.allclose(to_np(b.grad), np.ones_like(np_b))

        def case_no_grad(device):
            np_x = np.array([1.0, 2.0], dtype=np.float32)
            x = make(np_x, device)
            with set_grad_enabled(False):
                y = mysin(x)
            return (not y.requires_grad) and np.allclose(to_np(y), np.sin(np_x), atol=1e-5)

        def case_grad_none_error(device):
            def _none_derivative(result, x):
                return (None,)

            @track_grad(_none_derivative)
            def f(x):
                return rm.tensor(x.detach().array() * 2.0, device=x.device)

            x = make(np.array([1.0, 2.0], dtype=np.float32), device)
            try:
                f(x).sum().backward()
            except ValueError:
                return True
            return False

        def case_short_tuple_error(device):
            def _short_derivative(result, x, y):
                return (result.grad_value,)

            @track_grad(_short_derivative)
            def f(x, y):
                return rm.tensor(x.detach().array() + y.detach().array(), device=x.device)

            x = make(np.array([1.0, 2.0], dtype=np.float32), device)
            y = make(np.array([3.0, 4.0], dtype=np.float32), device)
            try:
                f(x, y).sum().backward()
            except ValueError:
                return True
            return False

        def case_invalid_gradfunc(device):
            try:
                track_grad(lambda: ())
            except TypeError:
                no_param_ok = True
            else:
                no_param_ok = False

            try:
                track_grad(lambda **kwargs: ())
            except TypeError:
                kw_first_ok = True
            else:
                kw_first_ok = False

            return no_param_ok and kw_first_ok

        test_cases = [
            ("单输入基础函数", case_single_input),
            ("多输入位置参数", case_multi_positional),
            ("多输入关键字参数", case_multi_keyword),
            ("关键字参数顺序打乱", case_keyword_order),
            ("多输入不同梯度值(除法)", case_divide),
            ("非 TN 参数在前", case_non_tn_before),
            ("TN 与非 TN 参数交替", case_mixed_tn_non_tn),
            ("部分输入不需要梯度", case_partial_requires_grad),
            ("默认参数", case_default_param),
            ("覆盖默认参数", case_override_default),
            ("*args 变长参数", case_varargs),
            ("**kwargs 变长参数", case_kwargs),
            ("非 TN 参数关键字调用", case_non_tn_keyword),
            ("矩阵乘法", case_matmul),
            ("前向返回非TN", case_non_tn_return),
            ("重复反向(retain_graph)", case_repeat_backward),
            ("裸 *args 梯度函数", case_bare_varargs),
            ("梯度模式关闭", case_no_grad),
            ("梯度返回None报错", case_grad_none_error),
            ("梯度数量不足报错", case_short_tuple_error),
            ("非法grad_func签名", case_invalid_gradfunc),
        ]
        if TORCH_AVAILABLE:
            test_cases.append(("PyTorch 对比验证", case_pytorch_compare))

        for device in devices:
            for case_name, case_fn in test_cases:
                full_name = f"{case_name} - {device}"
                start_time = time.time()
                error_msg = None
                try:
                    passed = bool(case_fn(device))
                except Exception as e:
                    passed = False
                    error_msg = str(e)
                # 统计在两种运行模式下均完整工作
                stats.add_result(full_name, passed)
                if IS_SCRIPT_MODE:
                    color = Colors.OKGREEN if passed else Colors.FAIL
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {full_name} - {color}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                # 断言传导：确保 unittest/pytest 将失败判为测试失败
                self.assertTrue(passed, full_name + (f": {error_msg}" if error_msg else ""))
    
    def test_gradcheck_function(self):
        """测试 gradcheck 函数的功能"""
        
        # 测试场景列表
        test_cases = [
            {
                "name": "简单乘法测试",
                "func": lambda x, y: x * y,
                "inputs": lambda: [rm.tensor(2.0, dtype=rm.float64, requires_grad=True), rm.tensor(3.0, dtype=rm.float64, requires_grad=True)],
                "description": "测试两个标量的乘法运算"
            },
            {
                "name": "线性函数测试",
                "func": lambda x, w, b: (x * w).sum() + b,
                "inputs": lambda: [
                    rm.tensor([1.0, 2.0, 3.0], dtype=rm.float64, requires_grad=True),
                    rm.tensor([0.5, 0.5, 0.5], dtype=rm.float64, requires_grad=True),
                    rm.tensor(1.0, dtype=rm.float64, requires_grad=True)
                ],
                "description": "测试带权重和偏置的线性函数"
            },
            {
                "name": "Sigmoid激活函数测试",
                "func": lambda x: 1.0 / (1.0 + (-x).exp()),
                "inputs": lambda: [rm.tensor([-1.0, 0.0, 1.0], dtype=rm.float64, requires_grad=True)],
                "description": "测试sigmoid非线性激活函数"
            },
            {
                "name": "快速模式测试",
                "func": lambda x, y: x * y,
                "inputs": lambda: [rm.tensor(2.0, dtype=rm.float64, requires_grad=True), rm.tensor(3.0, dtype=rm.float64, requires_grad=True)],
                "description": "测试快速模式下的梯度检查",
                "fast_mode": True
            },
            {
                "name": "高维输入测试",
                "func": lambda x: x.sum(),
                "inputs": lambda: [rm.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=rm.float64, requires_grad=True)],
                "description": "测试高维张量的梯度检查"
            },
            {
                "name": "多个输出测试",
                "func": lambda x, y: (x * y, x + y),
                "inputs": lambda: [rm.tensor(2.0, dtype=rm.float64, requires_grad=True), rm.tensor(3.0, dtype=rm.float64, requires_grad=True)],
                "description": "测试函数返回多个张量的情况"
            },
            {
                "name": "复杂函数测试",
                "func": lambda x: (x * x + x).sin(),
                "inputs": lambda: [rm.tensor(1.0, dtype=rm.float64, requires_grad=True)],
                "description": "测试包含多个操作的复杂函数"
            },
            {
                "name": "不同误差容限测试",
                "func": lambda x: x * x,
                "inputs": lambda: [rm.tensor(2.0, dtype=rm.float64, requires_grad=True)],
                "description": "测试不同的误差容限参数",
                "eps": 1e-5,
                "atol": 1e-4,
                "rtol": 1e-2
            },
            {
                "name": "矩阵乘法测试",
                "func": lambda x, y: x @ y,
                "inputs": lambda: [
                    rm.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=rm.float64, requires_grad=True),
                    rm.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=rm.float64, requires_grad=True)
                ],
                "description": "测试矩阵乘法的梯度计算"
            },
            {
                "name": "自定义Function类测试",
                "func": None,  # 特殊处理，使用自定义Function类
                "inputs": lambda: [rm.tensor(2.0, dtype=rm.float64, requires_grad=True), rm.tensor(3.0, dtype=rm.float64, requires_grad=True)],
                "description": "测试使用自定义Function类的函数"
            }
        ]
        
        for case in test_cases:
            error_msg = None
            try:
                # 特殊处理自定义Function类测试
                if case["name"] == "自定义Function类测试":
                    # 定义自定义乘法函数
                    class MulFunction(Function):
                        """自定义乘法函数类"""
                        
                        @staticmethod
                        def forward(ctx, x, y):
                            """前向传播"""
                            ctx.save_for_backward(x, y)
                            return x * y
                        
                        @staticmethod
                        def backward(ctx, grad_output):
                            """反向传播"""
                            x, y = ctx.saved_tensors
                            return grad_output * y, grad_output * x
                    
                    # 创建使用自定义Function的函数
                    def custom_mul(x, y):
                        return MulFunction.apply(x, y)
                    
                    func = custom_mul
                else:
                    func = case["func"]
                
                inputs = case["inputs"]()  # 调用工厂函数获取输入
                
                # 调用gradcheck函数
                fast_mode = case.get("fast_mode", False)
                eps = case.get("eps", 1e-6)
                atol = case.get("atol", 1e-5)
                rtol = case.get("rtol", 1e-3)
                result = gradcheck(func, inputs, eps=eps, atol=atol, rtol=rtol, fast_mode=fast_mode)
                
                passed = result == True
                if passed:
                    print(f"{Colors.OKGREEN}[PASS]{Colors.ENDC} {case['name']}: {case['description']}")
                else:
                    print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {case['name']}: 梯度检查失败")
            except Exception as e:
                passed = False
                error_msg = str(e)
                print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {case['name']}: 错误 - {str(e)}")
            # 统计在两种运行模式下均完整工作
            stats.add_result(case['name'], passed)
            # 断言传导：确保 unittest/pytest 将梯度检查失败判为测试失败
            self.assertTrue(passed, case['name'] + (f": {error_msg}" if error_msg else ""))

    def test_function_class(self):
        """测试 Function 类的自定义梯度功能"""
        
        # 定义一个自定义的乘法函数
        class MulFunction(Function):
            """自定义乘法函数类"""
            
            @staticmethod
            def forward(ctx, x, y):
                """前向传播"""
                ctx.save_for_backward(x, y)
                return x * y
            
            @staticmethod
            def backward(ctx, grad_output):
                """反向传播"""
                x, y = ctx.saved_tensors
                return grad_output * y, grad_output * x
        
        # 定义一个自定义的ReLU函数
        class ReLUFunction(Function):
            """自定义ReLU函数类"""
            
            @staticmethod
            def forward(ctx, x):
                """前向传播"""
                ctx.save_for_backward(x)
                return rm.maximum(x, rm.tensor(0.0))
            
            @staticmethod
            def backward(ctx, grad_output):
                """反向传播"""
                x, = ctx.saved_tensors
                return grad_output * rm.tensor(x.detach().numpy() > 0)
        
        # 测试用例列表
        test_cases = [
            {
                "name": "MulFunction 梯度计算",
                "func": MulFunction.apply,
                "inputs": [rm.tensor(2.0, requires_grad=True), rm.tensor(3.0, requires_grad=True)],
                "expected_grads": [3.0, 2.0]
            },
            {
                "name": "ReLUFunction 梯度计算",
                "func": ReLUFunction.apply,
                "inputs": [rm.tensor(-1.0, requires_grad=True)],
                "expected_grad": 0.0
            }
        ]
        
        for case in test_cases:
            error_msg = None
            try:
                inputs = case["inputs"]
                result = case["func"](*inputs)
                # 反向传播
                result.backward()
                
                if len(inputs) == 1:
                    # 单输入情况
                    x = inputs[0]
                    passed = abs(x.grad.detach().numpy() - case["expected_grad"]) < 1e-8
                    if passed:
                        print(f"{Colors.OKGREEN}[PASS]{Colors.ENDC} ReLUFunction 梯度计算: x.grad = {x.grad.detach().numpy()}, 预期值 = {case['expected_grad']}")
                    else:
                        print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} ReLUFunction 梯度计算: x.grad = {x.grad.detach().numpy()}, 预期值 = {case['expected_grad']}")
                else:
                    # 多输入情况
                    x, y = inputs
                    expected_grads = case["expected_grads"]
                    passed = abs(x.grad.detach().numpy() - expected_grads[0]) < 1e-8 and abs(y.grad.detach().numpy() - expected_grads[1]) < 1e-8
                    if passed:
                        print(f"{Colors.OKGREEN}[PASS]{Colors.ENDC} MulFunction 梯度计算: x.grad = {x.grad.detach().numpy()}, y.grad = {y.grad.detach().numpy()}")
                    else:
                        print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} MulFunction 梯度计算: x.grad = {x.grad.detach().numpy()}, y.grad = {y.grad.detach().numpy()}, 预期值 = {expected_grads}")
            except Exception as e:
                passed = False
                error_msg = str(e)
                print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {case['name']}: 错误 - {str(e)}")
            # 统计在两种运行模式下均完整工作
            stats.add_result(case['name'], passed)
            # 断言传导：确保 unittest/pytest 将比较失败判为测试失败
            self.assertTrue(passed, case['name'] + (f": {error_msg}" if error_msg else ""))

if __name__ == "__main__":
    print("\n开始测试自定义梯度功能...")
    clear_screen()
    
    print("\n" + "="*60)
    print(f"{Colors.BOLD}Riemann 自定义梯度功能测试{Colors.ENDC}")
    print("="*60)
    print("测试目标: track_grad修饰器、Function类")
    print("\n开始执行所有测试...\n")
    
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestDefineGradFunc)
    test_result = unittest.TextTestRunner().run(test_suite)
    
    print("\n\n" + "="*60)
    print(f"{Colors.BOLD}测试完成{Colors.ENDC}")
    print("="*60)
    
    # 打印测试结果摘要
    stats.print_summary()
    
    if test_result.wasSuccessful():
        print(f"\n{Colors.OKGREEN}所有测试通过！{Colors.ENDC}")
    else:
        print(f"\n{Colors.FAIL}测试失败！请检查测试结果。{Colors.ENDC}")
    
    print(f"\n所有测试完成，共执行 {stats.total_cases} 个测试用例")
    
    # 退出码综合两个来源：unittest 方法级异常 + 统计用例通过率
    all_cases_passed = stats.total_cases > 0 and stats.passed_cases == stats.total_cases
    sys.exit(0 if (test_result.wasSuccessful() and all_cases_passed) else 1)