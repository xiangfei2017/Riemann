import unittest
import numpy as np
import time
import sys, os
import warnings

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann import TN
    from riemann.autograd import gradcheck
    from riemann.core import register_grad_rule, register_jvp_rule, TangentPair
    from riemann.core import set_grad_enabled
    from riemann.func import jvp
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
    """测试自定义梯度功能，包括 register_grad_rule / register_jvp_rule 修饰器和 Function 类"""
    
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

    def _run_cases(self, cases, device):
        """执行 (用例名, 返回 bool 的函数) 组，统一统计与断言"""
        for case_name, case_fn in cases:
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
            self.assertTrue(passed, full_name + (f": {error_msg}" if error_msg else ""))
    
    def test_register_grad_rule_decorator(self):
        """测试 register_grad_rule 修饰器：参数组合、返回值类型、矩阵乘与重复反向"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        # ==================== 被测函数（grad_func 签名为 (result, ...)） ====================
        # 1. 单输入：sin
        def _sin_derivative(result, x):
            return (result._grad_value * x.cos(),)

        @register_grad_rule(_sin_derivative)
        def mysin(x):
            return rm.tensor(np.sin(x.detach().array()), device=x.device)

        # 2. 多输入：加法
        def _add_derivative(result, x, y):
            return (result._grad_value, result._grad_value)

        @register_grad_rule(_add_derivative)
        def myadd(x, y):
            return rm.tensor(x.detach().array() + y.detach().array(), device=x.device)

        # 3. 除法：两个输入的梯度值不同，可校验参数与梯度的对应关系
        def _divide_derivative(result, x, y):
            return (result._grad_value / y, result._grad_value * (-x / (y * y)))

        @register_grad_rule(_divide_derivative)
        def mydivide(x, y):
            return x / y

        # 4. 非 TN 参数在前
        def _scale_derivative(result, scale, x):
            return (result._grad_value * scale,)

        @register_grad_rule(_scale_derivative)
        def myscale(scale, x):
            return rm.tensor(scale * x.detach().array(), device=x.device)

        # 5. TN 与非 TN 参数交替
        def _weighted_sum_derivative(result, a, weight1, b, weight2):
            return (result._grad_value * weight1, result._grad_value * weight2)

        @register_grad_rule(_weighted_sum_derivative)
        def weighted_sum(a, weight1, b, weight2):
            return rm.tensor(weight1 * a.detach().array() + weight2 * b.detach().array(), device=a.device)

        # 6. 带默认参数
        def _pow_derivative(result, x, exponent=2.0):
            local = exponent * (x.detach().array() ** (exponent - 1))
            return (result._grad_value * rm.tensor(local, device=x.device),)

        @register_grad_rule(_pow_derivative)
        def mypow(x, exponent=2.0):
            return rm.tensor(x.detach().array() ** exponent, device=x.device)

        # 7. grad_func 使用 *args
        def _sum_derivative(result, *args):
            return tuple(result._grad_value for _ in args)

        @register_grad_rule(_sum_derivative)
        def mysum(*xs):
            return rm.tensor(sum(x.detach().array() for x in xs), device=xs[0].device)

        # 8. grad_func 使用 **kwargs
        def _sum_kwargs_derivative(result, **kwargs):
            return tuple(result._grad_value for _ in kwargs.values())

        @register_grad_rule(_sum_kwargs_derivative)
        def mysum_kwargs(**kwargs):
            vals = list(kwargs.values())
            return rm.tensor(sum(v.detach().array() for v in vals), device=vals[0].device)

        # 9. 矩阵乘法：反向需要矩阵乘 @ 与转置
        def _matmul_derivative(result, x, y):
            g = result._grad_value
            return (g @ y.detach().T, x.detach().T @ g)

        @register_grad_rule(_matmul_derivative)
        def mymatmul(x, y):
            return rm.matmul(x, y)

        # 10. 前向返回非 TN（numpy 数组），由修饰器自动转换为 TN
        def _double_derivative(result, x):
            return (result._grad_value * 2.0,)

        @register_grad_rule(_double_derivative)
        def mydouble(x):
            return 2.0 * x.detach().array()

        # 11. grad_func 首参为裸 *args（args[0] 即 result）
        def _sum_bare_args_derivative(*args):
            result = args[0]
            return tuple(result._grad_value for _ in args[1:])

        @register_grad_rule(_sum_bare_args_derivative)
        def mysum_bare(*xs):
            return rm.tensor(sum(x.detach().array() for x in xs), device=xs[0].device)

        # 12. 仅关键字形参（KEYWORD_ONLY）：非 TN 的 scale 只能按关键字传入
        def _scale_kw_derivative(result, x, *, scale=1.0):
            return (result._grad_value * scale,)

        @register_grad_rule(_scale_kw_derivative)
        def myscale_kw(x, *, scale=1.0):
            return rm.tensor(scale * x.detach().array(), device=x.device)

        # 13. 位置限定形参（POSITIONAL_ONLY）
        def _divpos_derivative(result, x, y, /):
            return (result._grad_value / y, result._grad_value * (-x / (y * y)))

        @register_grad_rule(_divpos_derivative)
        def mydivpos(x, y, /):
            return x / y

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

        def case_fused_single_call(device):
            # 标准backward只调用一次用户梯度函数，并能正确处理冻结左输入
            calls = []

            def _mul_derivative(result, x, y):
                calls.append(None)
                return (result._grad_value * y, result._grad_value * x)

            @register_grad_rule(_mul_derivative)
            def mymul(x, y):
                return x * y

            x = make(2.0, device)
            y = make(3.0, device)
            result = mymul(x, y)
            result.backward()
            both_ok = (len(calls) == 1 and np.allclose(to_np(x.grad), 3.0)
                       and np.allclose(to_np(y.grad), 2.0))

            calls.clear()
            x = make(2.0, device, requires_grad=False)
            y = make(3.0, device, requires_grad=True)
            mymul(x, y).backward()
            partial_ok = (len(calls) == 1 and x.grad is None
                          and np.allclose(to_np(y.grad), 2.0))
            return both_ok and partial_ok

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

        def case_keyword_only(device):
            # 仅关键字形参：scale 只能按关键字传入，梯度仍按签名顺序对齐
            np_x = np.array([2.0, 3.0], dtype=np.float32)
            x = make(np_x, device)
            y = myscale_kw(x, scale=3.0)
            forward_ok = np.allclose(to_np(y), 3.0 * np_x)
            y.sum().backward()
            explicit_ok = forward_ok and np.allclose(to_np(x.grad), np.full_like(np_x, 3.0))

            x2 = make(np_x, device)
            y2 = myscale_kw(x2)          # 使用默认值 scale=1.0
            y2.sum().backward()
            return explicit_ok and np.allclose(to_np(y2), np_x) \
                and np.allclose(to_np(x2.grad), np.ones_like(np_x))

        def case_positional_only(device):
            x = rm.tensor(6.0, requires_grad=True, device=device)
            y = rm.tensor(3.0, requires_grad=True, device=device)
            z = mydivpos(x, y)
            forward_ok = np.allclose(to_np(z), 2.0)
            z.backward()
            return forward_ok \
                and np.allclose(to_np(x.grad), 1.0 / 3.0) \
                and np.allclose(to_np(y.grad), -2.0 / 3.0)

        def case_mixed_positional_keyword(device):
            # 同一个调用里位置与关键字混用，fromvars 仍按签名顺序（x, y）
            np_a = np.array([1.0, 2.0], dtype=np.float32)
            np_b = np.array([3.0, 4.0], dtype=np.float32)
            a, b = make(np_a, device), make(np_b, device)
            y = myadd(a, y=b)
            forward_ok = np.allclose(to_np(y), np_a + np_b)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(a.grad), np.ones_like(np_a)) \
                and np.allclose(to_np(b.grad), np.ones_like(np_b))

        def case_duplicate_input(device):
            # 同一个 TN 占据两个实参位置：fromvars 保留两份，梯度应累加两次
            np_x = np.array([1.0, 2.0], dtype=np.float32)
            x = make(np_x, device)
            y = myadd(x, x)
            forward_ok = np.allclose(to_np(y), 2.0 * np_x)
            y.sum().backward()
            return forward_ok and np.allclose(to_np(x.grad), np.full_like(np_x, 2.0))

        def case_no_grad(device):
            np_x = np.array([1.0, 2.0], dtype=np.float32)
            x = make(np_x, device)
            with set_grad_enabled(False):
                y = mysin(x)
            return (not y.requires_grad) and np.allclose(to_np(y), np.sin(np_x), atol=1e-5)

        def case_grad_none_error(device):
            def _none_derivative(result, x):
                return (None,)

            @register_grad_rule(_none_derivative)
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
                return (result._grad_value,)

            @register_grad_rule(_short_derivative)
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
                register_grad_rule(lambda: ())
            except TypeError:
                no_param_ok = True
            else:
                no_param_ok = False

            try:
                register_grad_rule(lambda **kwargs: ())
            except TypeError:
                kw_first_ok = True
            else:
                kw_first_ok = False

            return no_param_ok and kw_first_ok

        test_cases = [
            ("单输入基础函数", case_single_input),
            ("多输入位置参数", case_multi_positional),
            ("融合梯度单次调用及冻结输入", case_fused_single_call),
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
            ("仅关键字形参", case_keyword_only),
            ("位置限定形参", case_positional_only),
            ("位置与关键字混用", case_mixed_positional_keyword),
            ("重复实参梯度累加", case_duplicate_input),
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
    
    def test_register_jvp_rule_decorator(self):
        """测试 register_jvp_rule 修饰器：单独使用、与 register_grad_rule 组合、variadic 与高阶"""
        for device in _devices():
            self._run_jvp_rule_cases(device)

    def _run_jvp_rule_cases(self, device):
        """register_jvp_rule 用例组"""
        def to_np(t):
            return _arr(t)

        # ==================== 规则函数（签名与原函数一致；TN 实参为 (primal, tangent)） ====================
        def _square_jvp(result, x):
            x, x_t = x
            if x_t is not None:
                return (2.0 * x) * x_t

        def _square_grad(result, x):
            return (2.0 * x * result._grad_value,)

        def _mul_grad(result, x, y):
            return (result._grad_value * y, result._grad_value * x)

        def _mul_jvp(result, x, y):
            x, x_t = x
            y, y_t = y
            total = None
            if x_t is not None:
                total = x_t * y
            if y_t is not None:
                term = x * y_t
                total = term if total is None else total + term
            return total

        def _sum_jvp(result, *args, **kwargs):
            total = None
            for _primal, tangent in list(args) + list(kwargs.values()):
                if tangent is not None:
                    total = tangent if total is None else total + tangent
            return total

        def _double_jvp(result, x):
            _x, x_t = x
            if x_t is not None:
                return 2.0 * x_t

        # ==================== 被测函数 ====================
        # 1. 只登记 jvp 规则：反向由函数体内的可微算子提供
        @register_jvp_rule(_square_jvp)
        def sq_jvp_only(x):
            return x * x

        # 2/3. 两个装饰器同时使用（两种顺序）
        @register_grad_rule(_square_grad)
        @register_jvp_rule(_square_jvp)
        def sq_both_grad_outer(x):
            return rm.tensor(x.detach().array() ** 2, device=x.device)

        @register_jvp_rule(_square_jvp)
        @register_grad_rule(_square_grad)
        def sq_both_jvp_outer(x):
            return rm.tensor(x.detach().array() ** 2, device=x.device)

        # 4. 多输入：反向过滤 fromvars，jvp 规则需看到完整实参
        @register_grad_rule(_mul_grad)
        @register_jvp_rule(_mul_jvp)
        def myprod(x, y):
            return rm.tensor(x.detach().array() * y.detach().array(), device=x.device)

        # 5. *args / **kwargs
        @register_jvp_rule(_sum_jvp)
        def mysum_jvp(*xs):
            return rm.tensor(sum(v.detach().array() for v in xs), device=xs[0].device)

        @register_jvp_rule(_sum_jvp)
        def mysum_kw_jvp(**xs):
            vals = list(xs.values())
            return rm.tensor(sum(v.detach().array() for v in vals), device=vals[0].device)

        # 6. 前向返回非 TN，由修饰器转换
        @register_jvp_rule(_double_jvp)
        def mydouble_jvp(x):
            return 2.0 * x.detach().array()

        # 7. 元组实参：非 TN 元组原样传入，不能被误当作 (primal, tangent) 配对
        def _mean_dim_jvp(result, x, dim):
            x, x_t = x
            if x_t is None:
                return None
            return x_t.sum(dim=dim) / (x.shape[0] * x.shape[1])

        @register_jvp_rule(_mean_dim_jvp)
        def mymean_dim(x, dim=None):
            return rm.tensor(x.detach().array().mean(axis=dim), device=x.device)

        def _sum_mixed_jvp(result, *args):
            total = None
            for item in args:
                # 只有配对实参才是 TangentPair；普通元组/标量视为常量
                if not isinstance(item, TangentPair):
                    continue
                _primal, tangent = item
                if tangent is not None:
                    total = tangent if total is None else total + tangent
            return total

        @register_jvp_rule(_sum_mixed_jvp)
        def mysum_mixed(*xs):
            total = None
            for v in xs:
                if isinstance(v, TN):
                    total = v if total is None else total + v
            # 加 0.0 生成新节点：若直接返回输入张量本身，修饰器会把规则挂到输入节点上
            # 并写入自引用的 fromvars，图排序时沿自环死循环
            return total + 0.0

        # 8. 恒等映射：前向直接返回输入节点本身
        def _id_jvp(result, x):
            _primal, tangent = x
            return tangent

        @register_jvp_rule(_id_jvp)
        def myid_jvp(x):
            return x

        # 9. 仅关键字形参（KEYWORD_ONLY）与位置限定形参（POSITIONAL_ONLY）
        def _scale_kw_jvp(result, x, *, scale=1.0):
            _x, x_t = x
            if x_t is not None:
                return x_t * scale

        @register_jvp_rule(_scale_kw_jvp)
        def myscale_kw_jvp(x, *, scale=1.0):
            return rm.tensor(scale * x.detach().array(), device=x.device)

        def _divpos_jvp(result, x, y, /):
            x, x_t = x
            y, y_t = y
            total = None
            if x_t is not None:
                total = x_t / y
            if y_t is not None:
                term = -(x / (y * y)) * y_t
                total = term if total is None else total + term
            return total

        @register_jvp_rule(_divpos_jvp)
        def mydivpos_jvp(x, y, /):
            return rm.tensor(x.detach().array() / y.detach().array(), device=x.device)

        # 10. 两个装饰器同时使用（jvp 在外层）+ 多输入关键字反序
        @register_jvp_rule(_mul_jvp)
        @register_grad_rule(_mul_grad)
        def myprod_jvp_outer(x, y):
            return rm.tensor(x.detach().array() * y.detach().array(), device=x.device)

        # ==================== 各场景用例 ====================
        def _check_square(fn):
            a = np.array([2.0, 3.0])
            va = np.array([0.1, -0.4])
            x = _mk(a, device)
            v = _mk(va, device, requires_grad=False)
            y, y_t = jvp(fn, (x,), (v,))
            forward_ok = np.allclose(to_np(y), a * a)
            tangent_ok = np.allclose(to_np(y_t), 2.0 * a * va)
            y.sum().backward()
            return forward_ok and tangent_ok and np.allclose(to_np(x.grad), 2.0 * a)

        def case_jvp_only():
            return _check_square(sq_jvp_only)

        def case_combined_grad_outer():
            return _check_square(sq_both_grad_outer)

        def case_combined_jvp_outer():
            return _check_square(sq_both_jvp_outer)

        def case_multi_input():
            a = np.array([1.0, 2.0])
            b = np.array([3.0, 4.0])
            va = np.array([0.5, -0.5])
            vb = np.array([0.2, 0.3])
            x, y = _mk(a, device), _mk(b, device)
            o, t = jvp(myprod, (x, y),
                       (_mk(va, device, requires_grad=False),
                        _mk(vb, device, requires_grad=False)))
            forward_ok = np.allclose(to_np(o), a * b)
            tangent_ok = np.allclose(to_np(t), va * b + a * vb)
            o.sum().backward()
            return (forward_ok and tangent_ok
                    and np.allclose(to_np(x.grad), b)
                    and np.allclose(to_np(y.grad), a))

        def case_filtered_fromvars():
            # fromvars 记录全部 TN 实参（含 requires_grad=False 的 const），
            # 这是前向图（JVP）与反向图共用的边定义；反向传播再按各来源的
            # requires_grad 过滤，因此冻结实参不会收到梯度。jvp 规则通过
            # (primal, tangent) 实参拿到完整输入，不依赖 fromvars 的过滤结果。
            a = np.array([1.0, 2.0])
            b = np.array([3.0, 4.0])
            va = np.array([0.5, -0.5])
            x = _mk(a, device)
            const = _mk(b, device, requires_grad=False)
            v = _mk(va, device, requires_grad=False)
            o, t = jvp(lambda p: myprod(p, const), (x,), (v,))
            o.sum().backward()
            return (np.allclose(to_np(o), a * b)
                    and np.allclose(to_np(t), va * b)
                    and len(o.fromvars) == 2
                    and const.grad is None)

        def case_varargs():
            arrays = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
            vecs = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([-0.1, 0.1])]
            xs = [_mk(a, device) for a in arrays]
            vs = [_mk(v, device, requires_grad=False) for v in vecs]
            o, t = jvp(mysum_jvp, tuple(xs), tuple(vs))
            return (np.allclose(to_np(o), sum(arrays))
                    and np.allclose(to_np(t), sum(vecs)))

        def case_kwargs():
            a = np.array([1.0, 2.0])
            b = np.array([3.0, 4.0])
            va = np.array([0.1, 0.2])
            vb = np.array([0.3, 0.4])
            x, y = _mk(a, device), _mk(b, device)
            o, t = jvp(lambda p, q: mysum_kw_jvp(x=p, y=q), (x, y),
                       (_mk(va, device, requires_grad=False),
                        _mk(vb, device, requires_grad=False)))
            return (np.allclose(to_np(o), a + b)
                    and np.allclose(to_np(t), va + vb))

        def case_non_tn_return():
            a = np.array([2.0, 3.0])
            va = np.array([0.1, -0.4])
            x = _mk(a, device)
            v = _mk(va, device, requires_grad=False)
            y, y_t = jvp(mydouble_jvp, (x,), (v,))
            # 前向返回值被修饰器转换为 TN；无 grad 规则，仅校验前向与 tangent
            return (isinstance(y, TN)
                    and np.allclose(to_np(y), 2.0 * a)
                    and np.allclose(to_np(y_t), 2.0 * va))

        def case_second_order():
            a = np.array([2.0, 3.0])
            va = np.array([0.1, -0.2])
            x = _mk(a, device)
            v = _mk(va, device, requires_grad=False)
            _, t = jvp(sq_both_grad_outer, (x,), (v,))
            total = t.sum()
            total.backward()
            return (np.allclose(to_np(total), np.sum(2.0 * a * va))
                    and np.allclose(to_np(x.grad), 2.0 * va))

        def case_invalid_rule():
            try:
                register_jvp_rule(123)
            except TypeError:
                return True
            return False

        def case_pytorch_compare():
            a = np.random.randn(3, 4)
            va = np.random.randn(3, 4)
            x = _mk(a, device)
            v = _mk(va, device, requires_grad=False)
            y, y_t = jvp(sq_both_grad_outer, (x,), (v,))
            tx = torch.tensor(a, requires_grad=True)
            with warnings.catch_warnings():
                # Python 3.14 下 torch.func.jvp 内部会触发 jit.script 的 FutureWarning
                warnings.filterwarnings(
                    "ignore", message=".*torch\\.jit\\.script.*", category=FutureWarning)
                ty, ty_t = torch.func.jvp(lambda t: t * t, (tx,), (torch.tensor(va),))
            return (np.allclose(to_np(y), ty.detach().numpy(), atol=1e-5, rtol=1e-5)
                    and np.allclose(to_np(y_t), ty_t.detach().numpy(), atol=1e-5, rtol=1e-5))

        def case_tuple_args():
            # 非 TN 的元组实参（如 dim=(0, 1)）必须原样传给规则；
            # 即使元组里含 TN（(p, 1.0)），也不是 TangentPair，按常量处理
            a = np.array([[1.0, 2.0], [3.0, 4.0]])
            va = np.array([[0.5, -0.5], [0.5, -0.5]])
            x = _mk(a, device)
            v = _mk(va, device, requires_grad=False)
            o1, t1 = jvp(lambda p: mymean_dim(p, dim=(0, 1)), (x,), (v,))
            o2, t2 = jvp(lambda p: mysum_mixed(p, (p, 1.0)), (x,), (v,))
            return (np.allclose(to_np(o1), a.mean(axis=(0, 1)))
                    and np.allclose(to_np(t1), va.mean(axis=(0, 1)))
                    and np.allclose(to_np(o2), a)
                    and np.allclose(to_np(t2), va))

        def case_identity_return():
            # 前向直接返回输入节点：装饰器不得写入自引用的 fromvars（图排序会
            # 沿自环死循环），也不得覆盖输入节点；输出 tangent 就是输入 tangent
            a = np.array([1.0, 2.0])
            va = np.array([0.3, -0.2])
            x = _mk(a, device)
            v = _mk(va, device, requires_grad=False)
            o, t = jvp(myid_jvp, (x,), (v,))
            return (o is x and not x.fromvars
                    and np.allclose(to_np(o), a)
                    and np.allclose(to_np(t), va))

        def case_keyword_only():
            # 仅关键字形参：scale 走关键字实参，规则按签名收到它；默认值同样生效
            a = np.array([2.0, 3.0])
            va = np.array([0.1, 0.2])
            x = _mk(a, device)
            v = _mk(va, device, requires_grad=False)
            y1, t1 = jvp(lambda p: myscale_kw_jvp(p, scale=3.0), (x,), (v,))
            y2, t2 = jvp(lambda p: myscale_kw_jvp(p), (x,), (v,))   # 默认 scale=1.0
            return (np.allclose(to_np(y1), 3.0 * a) and np.allclose(to_np(t1), 3.0 * va)
                    and np.allclose(to_np(y2), a) and np.allclose(to_np(t2), va))

        def case_positional_only():
            # 位置限定形参（POSITIONAL_ONLY）：规则同样按位置拿到两个配对实参
            x = _mk(6.0, device)
            y = _mk(3.0, device)
            vx = _mk(0.5, device, requires_grad=False)
            vy = _mk(-0.25, device, requires_grad=False)
            o, t = jvp(lambda p, q: mydivpos_jvp(p, q), (x, y), (vx, vy))
            expected = 0.5 / 3.0 - (6.0 / 9.0) * (-0.25)
            return np.allclose(to_np(o), 2.0) and np.allclose(to_np(t), expected)

        def case_mixed_kw_call():
            # 位置与关键字混用、关键字顺序与签名相反；同时使用两个装饰器（jvp 外层）
            a = np.array([1.0, 2.0])
            b = np.array([3.0, 4.0])
            va = np.array([0.5, -0.5])
            vb = np.array([0.2, 0.3])
            x, y = _mk(a, device), _mk(b, device)
            o, t = jvp(lambda p, q: myprod_jvp_outer(y=q, x=p), (x, y),
                       (_mk(va, device, requires_grad=False),
                        _mk(vb, device, requires_grad=False)))
            forward_ok = np.allclose(to_np(o), a * b)
            tangent_ok = np.allclose(to_np(t), va * b + a * vb)
            o.sum().backward()
            return (forward_ok and tangent_ok
                    and np.allclose(to_np(x.grad), b) and np.allclose(to_np(y.grad), a))

        def case_duplicate_input():
            # 同一张量占据两个输入位置：切线按各位置贡献叠加，梯度累加两次
            a = np.array([1.0, 2.0])
            va = np.array([0.3, -0.2])
            x = _mk(a, device)
            v = _mk(va, device, requires_grad=False)
            o, t = jvp(lambda p: myprod(p, p), (x,), (v,))
            o.sum().backward()
            return (np.allclose(to_np(o), a * a) and np.allclose(to_np(t), 2.0 * a * va)
                    and np.allclose(to_np(x.grad), 2.0 * a))

        def case_frozen_tangent():
            # 冻结实参（requires_grad=False）也可以有切线：它仍在 fromvars 中，
            # 图遍历会访问并播种它的方向；反向仍不为它累计梯度
            a = np.array([1.0, 2.0])
            b = np.array([3.0, 4.0])
            va = np.array([0.5, -0.5])
            vb = np.array([0.2, 0.3])
            x = _mk(a, device)
            const = _mk(b, device, requires_grad=False)
            o, t = jvp(lambda p, q: myprod(p, q), (x, const),
                       (_mk(va, device, requires_grad=False),
                        _mk(vb, device, requires_grad=False)))
            ok = np.allclose(to_np(o), a * b) and np.allclose(to_np(t), va * b + a * vb)
            o.sum().backward()
            return ok and np.allclose(to_np(x.grad), b) and const.grad is None

        def case_zero_direction():
            # tangents 中某个位置为零向量：该输入方向为零
            a = np.array([1.0, 2.0])
            b = np.array([3.0, 4.0])
            va = np.array([0.5, -0.5])
            vb = np.array([0.2, 0.3])
            zeros_a = np.zeros_like(a)
            zeros_b = np.zeros_like(b)
            x = _mk(a, device)
            const = _mk(b, device, requires_grad=False)
            o1, t1 = jvp(lambda p, q: myprod(p, q), (x, const),
                         (_mk(zeros_a, device, requires_grad=False),
                          _mk(vb, device, requires_grad=False)))
            o2, t2 = jvp(lambda p, q: myprod(p, q), (x, const),
                         (_mk(va, device, requires_grad=False),
                          _mk(zeros_b, device, requires_grad=False)))
            return (np.allclose(to_np(o1), a * b) and np.allclose(to_np(t1), a * vb)
                    and np.allclose(to_np(o2), a * b) and np.allclose(to_np(t2), va * b))

        cases = [
            ("单独登记 jvp 规则", case_jvp_only),
            ("grad 外层组合", case_combined_grad_outer),
            ("jvp 外层组合", case_combined_jvp_outer),
            ("多输入 tangent", case_multi_input),
            ("冻结实参 fromvars 过滤与恢复", case_filtered_fromvars),
            ("*args 变长参数", case_varargs),
            ("**kwargs 变长参数", case_kwargs),
            ("元组实参不误配对", case_tuple_args),
            ("恒等映射不建自环", case_identity_return),
            ("仅关键字形参", case_keyword_only),
            ("位置限定形参", case_positional_only),
            ("位置与关键字混用", case_mixed_kw_call),
            ("重复实参切线叠加", case_duplicate_input),
            ("冻结实参带切线", case_frozen_tangent),
            ("零方向 tangent", case_zero_direction),
            ("前向返回非TN", case_non_tn_return),
            ("create_graph 二阶", case_second_order),
            ("非法 jvp_func 报错", case_invalid_rule),
        ]
        if TORCH_AVAILABLE and device == "cpu":
            cases.append(("PyTorch 对比验证", case_pytorch_compare))

        self._run_cases(cases, device)

    def test_function_class_jvp(self):
        """测试 Function 类通过静态 jvp 方法参与前向自动微分"""
        for device in _devices():
            self._run_function_jvp_cases(device)

    def _run_function_jvp_cases(self, device):
        """Function.jvp 用例组"""
        def to_np(t):
            return _arr(t)

        class MulFn(Function):
            """forward/backward/jvp 全实现，jvp 读取 save_for_forward 的张量"""

            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                ctx.save_for_forward(x, y)
                return x * y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * y, grad_output * x

            @staticmethod
            def jvp(ctx, gx, gy):
                x, y = ctx.saved_tensors
                total = None
                if gx is not None:
                    total = gx * y
                if gy is not None:
                    term = x * gy
                    total = term if total is None else total + term
                return total

        class MulNoJvp(Function):
            """未实现 jvp：前向模式 AD 应直接报错"""

            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x * y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * y, grad_output * x

        class SplitSaveFn(Function):
            """backward 与 jvp 分别保存不同张量，并验证 ctx 自定义属性可见"""

            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x)       # backward 只解包 x
                ctx.save_for_forward(x, y)     # jvp 解包 x, y
                ctx.z = 3.0
                return x * y

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                assert x is not None
                return grad_output * 2.0, grad_output * 5.0

            @staticmethod
            def jvp(ctx, gx, gy):
                x, y = ctx.saved_tensors
                total = None
                if gx is not None:
                    total = gx * y
                if gy is not None:
                    term = x * gy
                    total = term if total is None else total + term
                return None if total is None else ctx.z * total

        class EmptyFwdSaveFn(Function):
            """只调用 save_for_backward：jvp 中 saved_tensors 必须为空"""

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return (grad_output * 2.0 * x,)

            @staticmethod
            def jvp(ctx, gx):
                assert ctx.saved_tensors == (), ctx.saved_tensors
                return None if gx is None else gx * 0.0

        class ScaleFn(Function):
            """包含非张量参数；jvp 对应位置收到 None"""

            @staticmethod
            def forward(ctx, x, k):
                return x * k

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 3.0, None

            @staticmethod
            def jvp(ctx, gx, gk):
                assert gk is None
                return None if gx is None else gx * 3.0

        class BadFwdSaveFn(Function):
            """save_for_forward 传入非张量应报错"""

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_forward(x, 3)
                return x

            @staticmethod
            def jvp(ctx, gx):
                return gx

        class NewStyleFn(Function):
            """新式写法：forward 不接收 ctx，由 setup_context 建上下文"""

            @staticmethod
            def forward(x, y):
                return x * y

            @staticmethod
            def setup_context(ctx, inputs, output):
                x, y = inputs
                assert isinstance(output, TN)
                assert ctx.needs_input_grad == (True, True), ctx.needs_input_grad
                ctx.save_for_backward(x, y)
                ctx.save_for_forward(x, y)

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * y, grad_output * x

            @staticmethod
            def jvp(ctx, gx, gy):
                x, y = ctx.saved_tensors
                total = None
                if gx is not None:
                    total = gx * y
                if gy is not None:
                    term = x * gy
                    total = term if total is None else total + term
                return total

        class NewStyleScaleFn(Function):
            """新式写法 + 非张量参数：needs_input_grad 逐位置对应"""

            @staticmethod
            def forward(x, k):
                return x * k

            @staticmethod
            def setup_context(ctx, inputs, output):
                x, k = inputs
                assert ctx.needs_input_grad == (True, False), ctx.needs_input_grad
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return grad_output * 3.0, None

        class ProbeNeedsGradFn(Function):
            """记录 needs_input_grad，验证无梯度输入时为 False"""

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                assert ctx.needs_input_grad == (False,), ctx.needs_input_grad
                return x * 2.0

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output * 2.0,)

        class VarargsFn(Function):
            """变长参数：inputs 为 *xs，backward/jvp 也按位置逐项对应"""

            @staticmethod
            def forward(ctx, *xs):
                ctx.save_for_backward(*xs)
                ctx.save_for_forward(*xs)
                total = xs[0]
                for v in xs[1:]:
                    total = total + v
                return total

            @staticmethod
            def backward(ctx, grad_output):
                return tuple(grad_output for _ in ctx.saved_tensors)

            @staticmethod
            def jvp(ctx, *gxs):
                total = None
                for gx in gxs:
                    if gx is not None:
                        total = gx if total is None else total + gx
                return total

        class ZeroJvpFn(Function):
            """jvp 返回 None 表示零方向：传播后得到与输出同形的零 tangent"""

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return (grad_output * 2.0 * x,)

            @staticmethod
            def jvp(ctx, gx):
                return None

        class NonTensorJvpFn(Function):
            """jvp 返回非 TN（numpy 数组）：由 apply 的规则包装转换为 TN"""

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * 3.0

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output * 3.0,)

            @staticmethod
            def jvp(ctx, gx):
                return None if gx is None else gx.detach().array() * 3.0

        # 与 Function 节点混用同一张图的装饰器算子（同时登记反向与前向规则）
        def _affine_jvp(result, x):
            _x, x_t = x
            return None if x_t is None else 2.0 * x_t

        def _affine_grad(result, x):
            return (2.0 * result._grad_value,)

        @register_jvp_rule(_affine_jvp)
        @register_grad_rule(_affine_grad)
        def mydouble(x):
            return rm.tensor(2.0 * x.detach().array(), device=x.device)

        a = np.array([[1.0, -2.0], [3.0, 0.5]])
        b = np.array([[0.5, 2.0], [-1.0, 4.0]])
        va = np.array([[0.1, 0.2], [-0.3, 0.4]])
        vb = np.array([[0.7, -0.1], [0.2, 0.5]])

        def case_mul_jvp_and_backward():
            x, y = _mk(a, device), _mk(b, device)
            vx = _mk(va, device, requires_grad=False)
            vy = _mk(vb, device, requires_grad=False)
            o, t = jvp(MulFn.apply, (x, y), (vx, vy))
            forward_ok = np.allclose(to_np(o), a * b)
            tangent_ok = np.allclose(to_np(t), va * b + a * vb)
            o.sum().backward()
            return (forward_ok and tangent_ok
                    and np.allclose(to_np(x.grad), b)
                    and np.allclose(to_np(y.grad), a))

        def case_missing_jvp_raises():
            x, y = _mk(a, device), _mk(b, device)
            vx = _mk(va, device, requires_grad=False)
            vy = _mk(vb, device, requires_grad=False)
            try:
                jvp(MulNoJvp.apply, (x, y), (vx, vy))
            except RuntimeError as exc:
                return "no forward (jvp) rule" in str(exc)
            return False

        def case_split_saved_tensors():
            x, y = _mk(a, device), _mk(b, device)
            vx = _mk(va, device, requires_grad=False)
            vy = _mk(vb, device, requires_grad=False)
            o, t = jvp(SplitSaveFn.apply, (x, y), (vx, vy))
            tangent_ok = np.allclose(to_np(t), 3.0 * (va * b + a * vb))
            o.sum().backward()
            return (tangent_ok
                    and np.allclose(to_np(x.grad), np.full_like(a, 2.0))
                    and np.allclose(to_np(y.grad), np.full_like(b, 5.0)))

        def case_empty_forward_saved():
            x = _mk(a, device)
            vx = _mk(va, device, requires_grad=False)
            o, t = jvp(EmptyFwdSaveFn.apply, (x,), (vx,))
            return (np.allclose(to_np(o), a * a)
                    and np.allclose(to_np(t), np.zeros_like(a)))

        def case_non_tensor_arg():
            x = _mk(a, device)
            vx = _mk(va, device, requires_grad=False)
            o, t = jvp(lambda p: ScaleFn.apply(p, 3.0), (x,), (vx,))
            o.sum().backward()
            return (np.allclose(to_np(o), a * 3.0)
                    and np.allclose(to_np(t), va * 3.0)
                    and np.allclose(to_np(x.grad), np.full_like(a, 3.0)))

        def case_second_order():
            x = _mk(a, device)
            vx = _mk(va, device, requires_grad=False)
            _, t = jvp(lambda p: MulFn.apply(p, p), (x,), (vx,))
            total = t.sum()
            total.backward()
            return (np.allclose(to_np(total), np.sum(2.0 * a * va))
                    and np.allclose(to_np(x.grad), 2.0 * va))

        def case_reject_non_tensor_forward_save():
            x = _mk(a, device)
            try:
                BadFwdSaveFn.apply(x)
            except AssertionError as e:
                return "save_for_forward expects all arguments to be tensors" in str(e)
            return False

        def case_setup_context_forward_backward_jvp():
            x, y = _mk(a, device), _mk(b, device)
            vx = _mk(va, device, requires_grad=False)
            vy = _mk(vb, device, requires_grad=False)
            o, t = jvp(NewStyleFn.apply, (x, y), (vx, vy))
            tangent_ok = np.allclose(to_np(t), va * b + a * vb)
            o.sum().backward()
            return (np.allclose(to_np(o), a * b)
                    and tangent_ok
                    and np.allclose(to_np(x.grad), b)
                    and np.allclose(to_np(y.grad), a))

        def case_setup_context_non_tensor_arg():
            x = _mk(a, device)
            o = NewStyleScaleFn.apply(x, 3.0)
            o.sum().backward()
            return (np.allclose(to_np(o), a * 3.0)
                    and np.allclose(to_np(x.grad), np.full_like(a, 3.0)))

        def case_needs_input_grad_false():
            x = _mk(a, device, requires_grad=False)
            o = ProbeNeedsGradFn.apply(x)
            return np.allclose(to_np(o), a * 2.0)

        def case_default_setup_context():
            try:
                Function.setup_context(None, (), None)
            except NotImplementedError:
                return True
            return False

        def case_no_grad_mode():
            x, y = _mk(a, device), _mk(b, device)
            with set_grad_enabled(False):
                o = MulFn.apply(x, y)
            return (not o.requires_grad) and np.allclose(to_np(o), a * b)

        def case_pytorch_compare():
            x, y = _mk(a, device), _mk(b, device)
            vx = _mk(va, device, requires_grad=False)
            vy = _mk(vb, device, requires_grad=False)
            o, t = jvp(MulFn.apply, (x, y), (vx, vy))
            tx = torch.tensor(a, requires_grad=True)
            ty = torch.tensor(b, requires_grad=True)
            with warnings.catch_warnings():
                # Python 3.14 下 torch.func.jvp 内部会触发 jit.script 的 FutureWarning
                warnings.filterwarnings(
                    "ignore", message=".*torch\\.jit\\.script.*", category=FutureWarning)
                to, tt = torch.func.jvp(lambda p, q: p * q,
                                        (tx, ty), (torch.tensor(va), torch.tensor(vb)))
            return (np.allclose(to_np(o), to.detach().numpy(), atol=1e-5, rtol=1e-5)
                    and np.allclose(to_np(t), tt.detach().numpy(), atol=1e-5, rtol=1e-5))

        def case_backward_only_function():
            # 只实现 backward 的 Function：不请求前向 AD 时正常参与反向
            x, y = _mk(a, device), _mk(b, device)
            o = MulNoJvp.apply(x, y)
            forward_ok = np.allclose(to_np(o), a * b)
            rules_ok = o._grad_rule is not None
            o.sum().backward()
            return (forward_ok and rules_ok
                    and np.allclose(to_np(x.grad), b)
                    and np.allclose(to_np(y.grad), a))

        def case_varargs_function():
            # 变长参数 Function：fromvars 与 jvp 的 *gxs 都按 *xs 位置对齐
            arrays = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
            vecs = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([-0.1, 0.1])]
            xs = [_mk(v, device) for v in arrays]
            vs = [_mk(v, device, requires_grad=False) for v in vecs]
            o, t = jvp(VarargsFn.apply, tuple(xs), tuple(vs))
            tangent_ok = np.allclose(to_np(t), sum(vecs))
            o.sum().backward()
            grads_ok = all(np.allclose(to_np(x.grad), np.ones_like(arr))
                           for x, arr in zip(xs, arrays))
            return np.allclose(to_np(o), sum(arrays)) and tangent_ok and grads_ok

        def case_duplicate_input_function():
            # 同一个 TN 占据两个输入位置：切线按两位置叠加，梯度累加两次
            x = _mk(a, device)
            v = _mk(va, device, requires_grad=False)
            o, t = jvp(lambda p: MulFn.apply(p, p), (x,), (v,))
            o.sum().backward()
            return (np.allclose(to_np(o), a * a)
                    and np.allclose(to_np(t), 2.0 * a * va)
                    and np.allclose(to_np(x.grad), 2.0 * a))

        def case_frozen_tangent_function():
            # 冻结输入（requires_grad=False）带非零切线：前向跟踪时它仍在
            # fromvars 中并参与 tangent 传播，但反向不为其累计梯度
            x = _mk(a, device)
            const = _mk(b, device, requires_grad=False)
            vx = _mk(va, device, requires_grad=False)
            vy = _mk(vb, device, requires_grad=False)
            o, t = jvp(lambda p, q: MulFn.apply(p, q), (x, const), (vx, vy))
            ok = (np.allclose(to_np(o), a * b)
                  and np.allclose(to_np(t), va * b + a * vb))
            o.sum().backward()
            return ok and np.allclose(to_np(x.grad), b) and const.grad is None

        def case_zero_direction_function():
            # 某位置切线为零向量：只保留另一位置的方向
            x, y = _mk(a, device), _mk(b, device)
            vx = _mk(va, device, requires_grad=False)
            vy = _mk(vb, device, requires_grad=False)
            zx = _mk(np.zeros_like(a), device, requires_grad=False)
            zy = _mk(np.zeros_like(b), device, requires_grad=False)
            o1, t1 = jvp(lambda p, q: MulFn.apply(p, q), (x, y), (zx, vy))
            o2, t2 = jvp(lambda p, q: MulFn.apply(p, q), (x, y), (vx, zy))
            return (np.allclose(to_np(o1), a * b) and np.allclose(to_np(t1), a * vb)
                    and np.allclose(to_np(o2), a * b) and np.allclose(to_np(t2), va * b))

        def case_jvp_returns_none():
            # jvp 返回 None 表示零方向：对外仍返回同形零 tangent，反向不受影响
            x = _mk(a, device)
            vx = _mk(va, device, requires_grad=False)
            o, t = jvp(ZeroJvpFn.apply, (x,), (vx,))
            rules_ok = (o._grad_rule is not None) and (o._jvp_rule is not None)
            o.sum().backward()
            return (rules_ok
                    and np.allclose(to_np(o), a * a)
                    and np.allclose(to_np(t), np.zeros_like(a))
                    and np.allclose(to_np(x.grad), 2.0 * a))

        def case_jvp_returns_non_tensor():
            # jvp 返回 numpy 数组：apply 的规则包装会把它转换成 TN 再传播
            x = _mk(a, device)
            vx = _mk(va, device, requires_grad=False)
            o, t = jvp(NonTensorJvpFn.apply, (x,), (vx,))
            o.sum().backward()
            return (np.allclose(to_np(o), 3.0 * a)
                    and np.allclose(to_np(t), 3.0 * va)
                    and np.allclose(to_np(x.grad), np.full_like(a, 3.0)))

        def case_both_rules_same_node():
            # 同一个 Function 输出节点上 _grad_rule 与 _jvp_rule 同时存在：
            # 先针对已有节点求 jvp，再反向，两个规则互不干扰
            x, y = _mk(a, device), _mk(b, device)
            vx = _mk(va, device, requires_grad=False)
            vy = _mk(vb, device, requires_grad=False)
            o = MulFn.apply(x, y)
            rules_ok = (o._grad_rule is not None) and (o._jvp_rule is not None)
            _, t = jvp(lambda p, q: o, (x, y), (vx, vy))
            tangent_ok = np.allclose(to_np(t), va * b + a * vb)
            o.sum().backward()
            return (rules_ok and tangent_ok
                    and np.allclose(to_np(x.grad), b)
                    and np.allclose(to_np(y.grad), a))

        def case_mixed_with_decorator():
            # Function 节点与 register_*_rule 装饰器节点在同一张图中混用：
            # jvp 与反向都要跨两类节点正确传播
            x, y = _mk(a, device), _mk(b, device)
            vx = _mk(va, device, requires_grad=False)
            vy = _mk(vb, device, requires_grad=False)
            o, t = jvp(lambda p, q: MulFn.apply(p, q) + mydouble(p),
                       (x, y), (vx, vy))
            tangent_ok = np.allclose(to_np(t), va * b + a * vb + 2.0 * va)
            o.sum().backward()
            return (np.allclose(to_np(o), a * b + 2.0 * a) and tangent_ok
                    and np.allclose(to_np(x.grad), b + 2.0)
                    and np.allclose(to_np(y.grad), a))

        cases = [
            ("MulFn jvp 与反向", case_mul_jvp_and_backward),
            ("未实现 jvp 时前向模式报错", case_missing_jvp_raises),
            ("backward/jvp 保存集合隔离", case_split_saved_tensors),
            ("未调用 save_for_forward 时为空", case_empty_forward_saved),
            ("非张量参数传入 None", case_non_tensor_arg),
            ("create_graph 二阶", case_second_order),
            ("save_for_forward 拒绝非张量", case_reject_non_tensor_forward_save),
            ("setup_context 前向/反向/jvp", case_setup_context_forward_backward_jvp),
            ("setup_context 非张量参数", case_setup_context_non_tensor_arg),
            ("needs_input_grad 无梯度输入", case_needs_input_grad_false),
            ("setup_context 默认未实现", case_default_setup_context),
            ("no_grad 下不建图", case_no_grad_mode),
            ("只实现 backward 的 Function", case_backward_only_function),
            ("变长参数 Function", case_varargs_function),
            ("重复实参梯度/切线叠加", case_duplicate_input_function),
            ("冻结输入带切线", case_frozen_tangent_function),
            ("零方向 tangent", case_zero_direction_function),
            ("jvp 返回 None 为零切线", case_jvp_returns_none),
            ("jvp 返回非 TN 自动转换", case_jvp_returns_non_tensor),
            ("grad/jvp 规则同一节点共存", case_both_rules_same_node),
            ("Function 与装饰器节点混合图", case_mixed_with_decorator),
        ]
        if TORCH_AVAILABLE and device == "cpu":
            cases.append(("PyTorch 对比验证", case_pytorch_compare))

        self._run_cases(cases, device)

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

    def test_function_class_grad(self):
        """测试 Function 类的自定义反向梯度：参数组合、冻结输入、重复使用与重复反向"""
        for device in _devices():
            self._run_function_grad_cases(device)

    def _run_function_grad_cases(self, device):
        """Function.backward 用例组"""
        def to_np(t):
            return _arr(t)

        # ==================== 被测 Function 类 ====================
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

        class ReLUFunction(Function):
            """自定义ReLU函数类；backward 返回单个 TN（非元组），由 apply 包装"""

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return rm.maximum(x, rm.tensor(0.0, device=x.device))

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                arr = x.detach().array()
                return grad_output * rm.tensor((arr > 0).astype(arr.dtype), device=x.device)

        class ScaleFn(Function):
            """非张量参数：needs_input_grad 逐位置对应，backward 该位置返回 None"""

            @staticmethod
            def forward(ctx, x, k):
                ctx.save_for_backward(x)
                assert ctx.needs_input_grad == (True, False), ctx.needs_input_grad
                return x * k

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return grad_output * 3.0, None

        class MixedGradFn(Function):
            """两个 TN 输入、只对第一个求梯度：fromvars 只保留可微输入"""

            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                assert ctx.needs_input_grad == (True, False), ctx.needs_input_grad
                return x * y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * y, grad_output * x

        class VarargsFn(Function):
            """变长参数：backward 的返回值与 *xs 逐位置对应"""

            @staticmethod
            def forward(ctx, *xs):
                ctx.save_for_backward(*xs)
                total = xs[0]
                for v in xs[1:]:
                    total = total + v
                return total

            @staticmethod
            def backward(ctx, grad_output):
                return tuple(grad_output for _ in ctx.saved_tensors)

        class NumpyOutFn(Function):
            """forward 返回 numpy 数组：apply 转换为 TN 后仍可反向"""

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return 2.0 * x.detach().array()

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output * 2.0,)

        # ==================== 各场景用例 ====================
        def case_mul():
            x, y = _mk(2.0, device), _mk(3.0, device)
            o = MulFunction.apply(x, y)
            forward_ok = np.allclose(to_np(o), 6.0)
            o.backward()
            return (forward_ok
                    and np.allclose(to_np(x.grad), 3.0)
                    and np.allclose(to_np(y.grad), 2.0))

        def case_relu():
            x = _mk([-1.0, 2.0], device)
            o = ReLUFunction.apply(x)
            forward_ok = np.allclose(to_np(o), [0.0, 2.0])
            o.sum().backward()
            return forward_ok and np.allclose(to_np(x.grad), [0.0, 1.0])

        def case_non_tensor_arg():
            # 非张量参数占据输入位置：backward 对应位置返回 None，不被当作梯度
            x = _mk([1.0, 2.0], device)
            o = ScaleFn.apply(x, 3.0)
            forward_ok = np.allclose(to_np(o), [3.0, 6.0])
            o.sum().backward()
            return forward_ok and np.allclose(to_np(x.grad), [3.0, 3.0])

        def case_frozen_input():
            # 冻结 TN 输入：不进 fromvars、不累计梯度，但 backward 仍返回它的位置值
            x = _mk([1.0, 2.0], device)
            y = _mk([3.0, 4.0], device, requires_grad=False)
            o = MixedGradFn.apply(x, y)
            fromvars_ok = len(o.fromvars) == 1 and o.fromvars[0] is x
            o.sum().backward()
            return (fromvars_ok
                    and np.allclose(to_np(x.grad), [3.0, 4.0])
                    and y.grad is None)

        def case_varargs():
            xs = [_mk([1.0, 2.0], device),
                  _mk([3.0, 4.0], device),
                  _mk([5.0, 6.0], device)]
            o = VarargsFn.apply(*xs)
            forward_ok = np.allclose(to_np(o), [9.0, 12.0])
            o.sum().backward()
            return forward_ok and all(
                np.allclose(to_np(x.grad), np.ones(2)) for x in xs)

        def case_duplicate_input():
            # 同一 TN 占据两个输入位置：fromvars 保留两份，梯度累加两次
            x = _mk([1.0, 2.0], device)
            o = MulFunction.apply(x, x)
            forward_ok = np.allclose(to_np(o), [1.0, 4.0])
            o.sum().backward()
            return forward_ok and np.allclose(to_np(x.grad), [2.0, 4.0])

        def case_numpy_forward():
            x = _mk([1.0, 2.0], device)
            o = NumpyOutFn.apply(x)
            o.sum().backward()
            return (isinstance(o, TN)
                    and np.allclose(to_np(o), [2.0, 4.0])
                    and np.allclose(to_np(x.grad), [2.0, 2.0]))

        def case_shared_output():
            # 同一输出被两条路径引用：每条边的梯度都要累加
            x, y = _mk([1.0, 2.0], device), _mk([3.0, 4.0], device)
            (MulFunction.apply(x, y) * 1.0 + 0.0).sum().backward()
            single_ok = (np.allclose(to_np(x.grad), [3.0, 4.0])
                         and np.allclose(to_np(y.grad), [1.0, 2.0]))
            x.grad = None
            y.grad = None
            o = MulFunction.apply(x, y)
            (o + o).sum().backward()
            return (single_ok
                    and np.allclose(to_np(x.grad), [6.0, 8.0])
                    and np.allclose(to_np(y.grad), [2.0, 4.0]))

        def case_repeat_backward():
            x, y = _mk(2.0, device), _mk(3.0, device)
            o = MulFunction.apply(x, y)
            o.backward()
            first_ok = (np.allclose(to_np(x.grad), 3.0)
                        and np.allclose(to_np(y.grad), 2.0))
            x.grad = None
            y.grad = None
            o.backward(gradient=rm.tensor(2.0, device=device), retain_graph=True)
            return (first_ok
                    and np.allclose(to_np(x.grad), 6.0)
                    and np.allclose(to_np(y.grad), 4.0))

        def case_no_grad_mode():
            x, y = _mk(2.0, device), _mk(3.0, device)
            with set_grad_enabled(False):
                o = MulFunction.apply(x, y)
            return (not o.requires_grad) and np.allclose(to_np(o), 6.0)

        cases = [
            ("MulFunction 梯度计算", case_mul),
            ("ReLUFunction 梯度计算", case_relu),
            ("非张量参数返回 None", case_non_tensor_arg),
            ("冻结输入不进 fromvars", case_frozen_input),
            ("变长参数 Function", case_varargs),
            ("重复实参梯度累加", case_duplicate_input),
            ("前向返回 numpy", case_numpy_forward),
            ("输出被多次引用", case_shared_output),
            ("重复反向(retain_graph)", case_repeat_backward),
            ("no_grad 下不建图", case_no_grad_mode),
        ]
        self._run_cases(cases, device)

if __name__ == "__main__":
    print("\n开始测试自定义梯度功能...")
    clear_screen()
    
    print("\n" + "="*60)
    print(f"{Colors.BOLD}Riemann 自定义梯度功能测试{Colors.ENDC}")
    print("="*60)
    print("测试目标: register_grad_rule / register_jvp_rule 修饰器、Function类（含 jvp）")
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