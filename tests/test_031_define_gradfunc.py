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
    from riemann.core import track_grad
    from riemann.core import set_grad_enabled
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


def _devices():
    """本次运行覆盖的设备列表：CUDA 可用时同时覆盖 CPU 与 CUDA"""
    return ["cpu", "cuda"] if CUDA_AVAILABLE else ["cpu"]


def _arr(t):
    """把 TN 张量/梯度转成 numpy，兼容 CPU 与 CUDA"""
    return t.detach().cpu().numpy()


def _mk(value, device=None, requires_grad=True):
    """构造 gradcheck 用的双精度输入张量"""
    return rm.tensor(value, dtype=rm.float64, device=device, requires_grad=requires_grad)


def _raises_runtime_error(call):
    """调用 call，若抛出 RuntimeError 则返回 True"""
    try:
        call()
    except RuntimeError:
        return True
    return False

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
    
    def test_gradcheck(self):
        """gradcheck 用例组：基础功能、错误检出、fast_mode、边界与参数校验"""
        for device in _devices():
            self._run_gradcheck_group(device)

    def _run_gradcheck_group(self, device):
        """执行 gradcheck 用例组内的全部子用例，并输出每个子用例的执行结果"""
        for case_name, case_fn in self._gradcheck_cases(device):
            full_name = f"{case_name} - {device}"
            start_time = time.time()
            error_msg = None
            try:
                passed = bool(case_fn())
            except Exception as e:
                passed = False
                error_msg = str(e)
            stats.add_result(full_name, passed)
            if IS_SCRIPT_MODE:
                color = Colors.OKGREEN if passed else Colors.FAIL
                status = "通过" if passed else "失败"
                print(f"测试用例: {full_name} - {color}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
            # 断言传导：确保 unittest/pytest 将子用例失败判为测试失败
            self.assertTrue(passed, full_name + (f": {error_msg}" if error_msg else ""))

    def _gradcheck_cases(self, device):
        """构造 gradcheck 用例组的全部子用例：[(子用例名, 返回 bool 的检查函数), ...]"""
        d = device
        cases = []

        # ==================== 基础功能 ====================
        cases.append(("基础-简单乘法", lambda: gradcheck(lambda x, y: x * y, (_mk(2.0, d), _mk(3.0, d)))))
        cases.append(("基础-线性函数", lambda: gradcheck(
            lambda x, w, b: (x * w).sum() + b,
            (_mk([1.0, 2.0, 3.0], d), _mk([0.5, 0.5, 0.5], d), _mk(1.0, d)))))
        cases.append(("基础-Sigmoid激活", lambda: gradcheck(
            lambda x: 1.0 / (1.0 + (-x).exp()), (_mk([-1.0, 0.0, 1.0], d),))))
        cases.append(("基础-高维输入", lambda: gradcheck(
            lambda x: x.sum(), (_mk([[1.0, 2.0], [3.0, 4.0]], d),))))
        cases.append(("基础-复杂函数", lambda: gradcheck(
            lambda x: (x * x + x).sin(), (_mk(1.0, d),))))
        cases.append(("基础-矩阵乘法", lambda: gradcheck(
            lambda x, y: x @ y,
            (_mk([[1.0, 2.0], [3.0, 4.0]], d), _mk([[5.0, 6.0], [7.0, 8.0]], d)))))
        cases.append(("基础-多输出", lambda: gradcheck(
            lambda x, y: (x * y, x + y), (_mk(2.0, d), _mk(3.0, d)))))
        cases.append(("基础-自定义容差", lambda: gradcheck(
            lambda x: x * x, (_mk(2.0, d),), eps=1e-5, atol=1e-4, rtol=1e-2)))

        class MulFunction(Function):
            """自定义乘法函数类"""

            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x * y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * y, grad_output * x

        cases.append(("基础-自定义Function", lambda: gradcheck(
            lambda x, y: MulFunction.apply(x, y), (_mk(2.0, d), _mk(3.0, d)))))

        # ==================== 错误梯度检出 ====================
        class BadMul(Function):
            """乘法前向正确，但 dx 被错误放大 1.5 倍的 Function"""

            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x * y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * y * 1.5, grad_output * x

        cases.append(("错误检出-返回False", lambda: gradcheck(
            lambda x, y: BadMul.apply(x, y), (_mk(2.0, d), _mk(3.0, d)),
            raise_exception=False) is False))

        def _bad_grad_raises():
            try:
                gradcheck(lambda x, y: BadMul.apply(x, y), (_mk(2.0, d), _mk(3.0, d)),
                          raise_exception=True)
            except RuntimeError:
                return True
            return False

        cases.append(("错误检出-抛出RuntimeError", _bad_grad_raises))

        # ==================== fast_mode ====================
        class FirstOnlyBadSquare(Function):
            """首元素梯度正确、次元素梯度错误的平方函数"""

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                factor = rm.tensor([1.0, 2.0], dtype=rm.float64, device=x.device)
                return (grad_output * 2.0 * x * factor,)

        cases.append(("fast_mode-正确函数通过", lambda: gradcheck(
            lambda a: (a.exp() * a.sin()).sum(), (_mk([1.0, 2.0], d),), fast_mode=True)))
        # 与 torch 一致：fast_mode 用随机投影，但仍覆盖完整 Jacobian，
        # 因此非首元素的梯度错误同样能被检出
        cases.append(("fast_mode-检出非首元素错误", lambda: gradcheck(
            lambda a: FirstOnlyBadSquare.apply(a), (_mk([1.0, 2.0], d),),
            fast_mode=True, raise_exception=False) is False))
        cases.append(("fast_mode-close模式同样检出", lambda: gradcheck(
            lambda a: FirstOnlyBadSquare.apply(a), (_mk([1.0, 2.0], d),),
            fast_mode=False, raise_exception=False) is False))
        # 多元素多输出场景下 fast_mode 也应通过
        cases.append(("fast_mode-多元素多输出通过", lambda: gradcheck(
            lambda x, y: (x.exp() * y, x.sin() + y * y),
            (_mk([[1.0, 2.0], [3.0, 4.0]], d), _mk([[0.5, 1.5], [2.5, 3.5]], d)),
            fast_mode=True)))
        # fast_mode 失败时应抛出 RuntimeError（内部会回退到全量模式定位元素）
        cases.append(("fast_mode-失败抛RuntimeError", lambda: _raises_runtime_error(
            lambda: gradcheck(lambda a: FirstOnlyBadSquare.apply(a),
                              (_mk([1.0, 2.0], d),), fast_mode=True))))

        # ==================== 多元素 + 多输出 ====================
        cases.append(("多元素多输出", lambda: gradcheck(
            lambda x, y: (x.exp() * y, x.sin() + y * y),
            (_mk([[1.0, 2.0], [3.0, 4.0]], d), _mk([[0.5, 1.5], [2.5, 3.5]], d)))))

        # ==================== 边界与副作用 ====================
        def _input_unchanged():
            x = _mk([[1.0, 2.0], [3.0, 4.0]], d)
            backup = np.array(_arr(x), copy=True)
            ok = gradcheck(lambda a: (a.exp() * a.sin()).sum(), (x,))
            return ok and np.array_equal(_arr(x), backup)

        cases.append(("边界-输入不被修改", _input_unchanged))
        cases.append(("边界-空张量输入", lambda: gradcheck(
            lambda a: a.sum(), (rm.tensor([], dtype=rm.float64, device=d, requires_grad=True),))))
        cases.append(("边界-未使用输入零梯度", lambda: gradcheck(
            lambda p, q: (p * p).sum(), (_mk([1.0, 2.0], d), _mk([3.0, 4.0], d)))))

        # ==================== Jacobian 逐元素校验 ====================
        class BadJacobian(Function):
            """前向 x @ ones(2,2)；反向乘一个行和正确但元素错误的矩阵。

            对 x 的每一行，错误雅可比的行和与真实雅可比的行和相同，
            因此"把输出求和后比较"的弱校验会漏检，只有逐输出元素比较才能发现。
            """

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                arrlib = x._get_array_lib()
                w = arrlib.ones((2, 2))
                return rm.tensor(x.detach().array() @ w, device=x.device, dtype=x.dtype)

            @staticmethod
            def backward(ctx, g):
                x, = ctx.saved_tensors
                # 真实雅可比全为 1，这里用 [[1.5,0.5],[0.5,1.5]]：
                # 每行行和 1.5+0.5 == 1+1，求和校验无法察觉
                arrlib = x._get_array_lib()
                w_wrong = arrlib.array([[1.5, 0.5], [0.5, 1.5]])
                return rm.tensor(g.detach().array() @ w_wrong.T,
                                 device=x.device, dtype=x.dtype)

        cases.append(("Jacobian-检出输出元素间抵消的错误", lambda: gradcheck(
            lambda a: BadJacobian.apply(a), (_mk([1.0, 2.0], d),),
            raise_exception=False) is False))
        cases.append(("Jacobian-同一错误抛RuntimeError", lambda: _raises_runtime_error(
            lambda: gradcheck(lambda a: BadJacobian.apply(a), (_mk([1.0, 2.0], d),)))))
        cases.append(("Jacobian-nondet_tol容忍小误差", lambda: gradcheck(
            lambda a: BadJacobian.apply(a), (_mk([1.0, 2.0], d),),
            nondet_tol=1.0, raise_exception=False) is True))
        # 非标量单输出：逐元素检查（旧实现会先 sum 成标量）
        cases.append(("Jacobian-非标量单输出", lambda: gradcheck(
            lambda a: a * a, (_mk([[1.0, 2.0], [3.0, 4.0]], d),))))
        # 多输出且形状不同
        cases.append(("Jacobian-多输出不同形状", lambda: gradcheck(
            lambda a, b: (a * b, (a + b).sum()), (_mk([1.0, 2.0], d), _mk([3.0, 4.0], d)))))
        # check_grad_dtypes 打开时，梯度 dtype 与输入一致的正常用例应通过
        cases.append(("Jacobian-check_grad_dtypes通过", lambda: gradcheck(
            lambda a: (a * a).sum(), (_mk([1.0, 2.0], d),), check_grad_dtypes=True)))

        # ==================== 参数/输入校验（与设备无关，各设备均执行）====================
        def _expect_value_error(call):
            try:
                call()
            except ValueError:
                return True
            return False

        def _expect_not_implemented(call):
            try:
                call()
            except NotImplementedError:
                return True
            return False

        x = _mk([1.0, 2.0], d)
        cases.append(("校验-单个TN作为inputs可用", lambda: gradcheck(
            lambda a: (a * a).sum(), _mk([1.0, 2.0], d))))
        cases.append(("校验-list形式inputs可用", lambda: gradcheck(
            lambda a: (a * a).sum(), [_mk([1.0, 2.0], d)])))
        cases.append(("校验-inputs非TN/序列", lambda: _expect_value_error(
            lambda: gradcheck(lambda a: a.sum(), 123))))
        cases.append(("校验-空inputs", lambda: _expect_value_error(
            lambda: gradcheck(lambda a: a.sum(), ()))))
        cases.append(("校验-输入非TN", lambda: _expect_value_error(
            lambda: gradcheck(lambda a: a.sum(), ([1.0, 2.0],)))))
        cases.append(("校验-requires_grad为False", lambda: _expect_value_error(
            lambda: gradcheck(lambda a: a.sum(), (_mk([1.0, 2.0], d, requires_grad=False),)))))
        cases.append(("校验-输入float32", lambda: _expect_value_error(
            lambda: gradcheck(lambda a: a.sum(),
                              (rm.tensor([1.0, 2.0], dtype=rm.float32, device=d, requires_grad=True),)))))
        cases.append(("校验-func不可调用", lambda: _expect_value_error(
            lambda: gradcheck(1.0, (x,)))))
        cases.append(("校验-func返回空tuple", lambda: _expect_value_error(
            lambda: gradcheck(lambda a: (), (x,)))))
        cases.append(("校验-func返回非TN", lambda: _expect_value_error(
            lambda: gradcheck(lambda a: 1.0, (x,)))))
        cases.append(("校验-tuple内含非TN", lambda: _expect_value_error(
            lambda: gradcheck(lambda a: (a.sum(), 1.0), (x,)))))

        # 与 torch 签名对齐：未实现的选项应显式报错而不是被静默忽略
        cases.append(("校验-check_forward_ad=True", lambda: _expect_not_implemented(
            lambda: gradcheck(lambda a: a.sum(), (x,), check_forward_ad=True))))
        cases.append(("校验-check_batched_grad=True", lambda: _expect_not_implemented(
            lambda: gradcheck(lambda a: a.sum(), (x,), check_batched_grad=True))))
        cases.append(("校验-check_batched_forward_grad=True", lambda: _expect_not_implemented(
            lambda: gradcheck(lambda a: a.sum(), (x,), check_batched_forward_grad=True))))
        cases.append(("校验-check_backward_ad=False", lambda: _expect_not_implemented(
            lambda: gradcheck(lambda a: a.sum(), (x,), check_backward_ad=False))))
        cases.append(("校验-masked非None", lambda: _expect_not_implemented(
            lambda: gradcheck(lambda a: a.sum(), (x,), masked=True))))

        # 关键参数只能按关键字传入（torch 中 inputs 之后有 *）
        def _positional_rejected():
            try:
                gradcheck(lambda a: a.sum(), (x,), 1e-6)
            except TypeError:
                return True
            return False

        cases.append(("校验-eps必须关键字传入", _positional_rejected))

        try:
            xc = rm.tensor([1.0 + 0j, 2.0 + 0j], dtype=rm.complex128, device=d, requires_grad=True)
            cases.append(("校验-输入complex128", lambda: _expect_value_error(
                lambda: gradcheck(lambda a: a.sum(), (xc,)))))
        except Exception:
            pass

        # ==================== 设备相关场景 ====================
        if d != "cpu":
            # 非连续（转置）输入：扰动写回需要正确处理非 C 连续内存
            cases.append(("设备-非连续(转置)输入", lambda: gradcheck(
                lambda a: (a.exp() * a.sin()).sum(),
                (rm.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]).T, dtype=rm.float64,
                           device=d, requires_grad=True),))))

            # 较大张量走完全部元素的数值梯度循环
            def _large_tensor():
                big = rm.tensor(np.arange(36.0).reshape(6, 6) / 10 + 0.5,
                                dtype=rm.float64, device=d, requires_grad=True)
                return gradcheck(lambda a: (a.sin() * a).sum(), (big,))

            cases.append(("设备-大张量全元素检查", _large_tensor))

            # 同一输入连续调用两次，验证扰动缓冲区复用不会污染输入
            def _repeat_calls():
                same = _mk([1.0, 2.0], d)
                first = gradcheck(lambda a: (a * a).sum(), (same,))
                second = gradcheck(lambda a: (a * a).sum(), (same,))
                return first and second and np.array_equal(_arr(same), np.array([1.0, 2.0]))

            cases.append(("设备-重复调用同一输入", _repeat_calls))

            # 解析梯度必须与输入同设备，不能隐式回落到 CPU
            def _grad_stays_on_device():
                t = _mk([1.0, 2.0], d)
                g, = rm.autograd.grad((t * t).sum(), [t])
                return d in str(g.device) and np.allclose(_arr(g), np.array([2.0, 4.0]))

            cases.append(("设备-梯度保持cuda设备", _grad_stays_on_device))

            # CPU 与 CUDA 混合输入应在计算时报错，而不是静默产生错误结果
            def _mixed_device_rejected():
                a_cuda = _mk([1.0, 2.0], "cuda")
                a_cpu = _mk([1.0, 2.0], "cpu")
                try:
                    gradcheck(lambda p, q: (p * q).sum(), (a_cuda, a_cpu))
                except RuntimeError:
                    return True
                return False

            cases.append(("设备-混合设备输入报错", _mixed_device_rejected))

        return cases

    def test_function_class(self):
        """测试 Function 类的自定义梯度功能（CPU + CUDA）"""

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

        # 定义一个自定义的ReLU函数（设备无关写法：零常量与掩码都建在输入所在设备上）
        class ReLUFunction(Function):
            """自定义ReLU函数类"""

            @staticmethod
            def forward(ctx, x):
                """前向传播"""
                ctx.save_for_backward(x)
                return rm.maximum(x, rm.tensor(0.0, device=x.device))

            @staticmethod
            def backward(ctx, grad_output):
                """反向传播"""
                x, = ctx.saved_tensors
                arr = x.detach().array()
                return grad_output * rm.tensor((arr > 0).astype(arr.dtype), device=x.device)

        # 测试用例列表（make_inputs 接收设备参数，便于在 CPU/CUDA 上分别构造）
        test_cases = [
            {
                "name": "MulFunction 梯度计算",
                "func": MulFunction.apply,
                "make_inputs": lambda d: [rm.tensor(2.0, device=d, requires_grad=True),
                                          rm.tensor(3.0, device=d, requires_grad=True)],
                "expected": [[3.0], [2.0]],
            },
            {
                "name": "ReLUFunction 梯度计算",
                "func": ReLUFunction.apply,
                "make_inputs": lambda d: [rm.tensor([-1.0, 2.0], device=d, requires_grad=True)],
                "expected": [[0.0, 1.0]],
            },
        ]

        for device in _devices():
            for case in test_cases:
                full_name = f"{case['name']} - {device}"
                error_msg = None
                try:
                    inputs = case["make_inputs"](device)
                    result = case["func"](*inputs)
                    # 反向传播
                    result.sum().backward()

                    actuals = [_arr(t.grad) for t in inputs]
                    expecteds = [np.asarray(e, dtype=a.dtype)
                                 for a, e in zip(actuals, case["expected"])]
                    passed = len(actuals) == len(expecteds) and all(
                        np.allclose(a, e, atol=1e-8, rtol=0) for a, e in zip(actuals, expecteds))
                    if IS_SCRIPT_MODE:
                        color = Colors.OKGREEN if passed else Colors.FAIL
                        status = "PASS" if passed else "FAIL"
                        detail = ", ".join(
                            f"in{i}.grad={_arr(t.grad).tolist()}" for i, t in enumerate(inputs))
                        print(f"{color}[{status}]{Colors.ENDC} [{device}] {case['name']}: "
                              f"{detail}, 预期={[e.tolist() for e in expecteds]}")
                except Exception as e:
                    passed = False
                    error_msg = str(e)
                    if IS_SCRIPT_MODE:
                        print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} [{device}] {case['name']}: 错误 - {str(e)}")
                # 统计在两种运行模式下均完整工作
                stats.add_result(full_name, passed)
                # 断言传导：确保 unittest/pytest 将比较失败判为测试失败
                self.assertTrue(passed, full_name + (f": {error_msg}" if error_msg else ""))

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