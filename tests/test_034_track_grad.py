"""测试 track_grad 装饰器在各种参数组合下的正确性"""
import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 导入 riemann 模块
try:
    import riemann as rm
    from riemann.tensordef import track_grad, TN
    CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
except ImportError as e:
    print(f"无法导入 riemann 模块: {e}")
    sys.exit(1)

# 尝试导入 PyTorch 进行梯度对比
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 PyTorch，跳过梯度对比测试")
    TORCH_AVAILABLE = False


# ==================== 颜色类 ====================
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ==================== 统计收集器 ====================
class StatisticsCollector:
    def __init__(self):
        self.total_cases = 0
        self.passed_cases = 0
        self.total_time = 0.0
        self.function_stats = {}
        self.current_function = None
        self.current_function_start_time = 0
        self.current_test_details = []

    def start_function(self, function_name):
        self.current_function = function_name
        self.current_function_start_time = time.time()
        self.current_test_details = []
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
        width = 0
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                width += 2
            else:
                width += 1
        return width

    def print_summary(self):
        headers = ['用例名', '通过/总数', '通过率', '耗时(秒)']
        header_widths = [self._get_display_width(h) for h in headers]
        max_func_name_width = header_widths[0]
        for func_name in self.function_stats.keys():
            max_func_name_width = max(max_func_name_width, self._get_display_width(func_name))
        col_widths = [
            max(max_func_name_width, header_widths[0]) + 2,
            header_widths[1] + 4,
            header_widths[2] + 4,
            header_widths[3] + 4
        ]
        total_width = sum(col_widths)
        print("\n" + "=" * total_width)
        print(f"{Colors.BOLD}测试统计摘要{Colors.ENDC}")
        print("=" * total_width)
        print(f"总测试用例数: {self.total_cases}")
        print(f"通过测试用例数: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases}{Colors.ENDC}")
        print(f"测试通过率: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases / self.total_cases * 100:.2f}%{Colors.ENDC}")
        print(f"总耗时: {self.total_time:.4f} 秒")
        print("\n各用例测试详情:")
        print("-" * total_width)
        header_line = ""
        for i, header in enumerate(headers):
            header_width = self._get_display_width(header)
            padding = col_widths[i] - header_width
            header_line += header + " " * padding
        print(header_line)
        print("-" * total_width)
        for func_name, stats in self.function_stats.items():
            pass_rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            status_color = Colors.OKGREEN if pass_rate == 100 else Colors.FAIL
            func_name_width = self._get_display_width(func_name)
            func_name_padding = col_widths[0] - func_name_width
            pass_total_display = f"{stats['passed']}/{stats['total']}"
            pass_total_width = self._get_display_width(pass_total_display)
            pass_total_padding = col_widths[1] - pass_total_width
            pass_rate_display = f"{pass_rate:.2f}"
            pass_rate_width = self._get_display_width(pass_rate_display)
            pass_rate_padding = col_widths[2] - pass_rate_width
            time_display = f"{stats['time']:.4f}"
            time_width = self._get_display_width(time_display)
            time_padding = col_widths[3] - time_width
            print(
                f"{func_name}{' ' * func_name_padding}" +
                f"{pass_total_display}{' ' * pass_total_padding}" +
                f"{status_color}{pass_rate_display}{' ' * pass_rate_padding}{Colors.ENDC}" +
                f"{time_display}{' ' * time_padding}"
            )
        print("=" * total_width)


# 全局统计实例
stats = StatisticsCollector()
IS_RUNNING_AS_SCRIPT = False


# ==================== 辅助函数 ====================
def compare_gradients(rm_tensor, torch_tensor, atol=1e-5, rtol=1e-5):
    """比较 Riemann 和 PyTorch 的梯度值"""
    if not TORCH_AVAILABLE:
        return True
    if rm_tensor.grad is None and torch_tensor.grad is None:
        return True
    if rm_tensor.grad is None or torch_tensor.grad is None:
        return False
    # 处理 CUDA 张量：先移动到 CPU 再转 numpy
    if rm_tensor.grad.is_cuda:
        rm_data = rm_tensor.grad.data.get()
    else:
        rm_data = rm_tensor.grad.numpy() if hasattr(rm_tensor.grad, 'numpy') else np.array(rm_tensor.grad.data)
    torch_data = torch_tensor.grad.detach().cpu().numpy()
    try:
        np.testing.assert_allclose(rm_data, torch_data, atol=atol, rtol=rtol)
        return True
    except AssertionError:
        return False


def rm_tensor(data, requires_grad=True, device="cpu"):
    """创建 Riemann 张量的便捷函数"""
    return rm.tensor(np.array(data, dtype=np.float32), requires_grad=requires_grad, device=device)


# ==================== 被装饰的测试函数 ====================
# 1. 单输入函数
def _sin_derivative(x):
    return (x.cos(),)

@track_grad(_sin_derivative)
def mysin(x):
    return rm.tensor(np.sin(x.data), device=x.device)


# 2. 多输入函数 — 位置参数
def _add_derivative(x, y):
    return (rm.tensor(1., device=x.device), rm.tensor(1., device=y.device))

@track_grad(_add_derivative)
def myadd(x, y):
    return rm.tensor(x.data + y.data, device=x.device)


# 3. 非 TN 参数在前
def _scale_derivative(scale, x):
    return (rm.tensor(scale, device=x.device),)

@track_grad(_scale_derivative)
def myscale(scale, x):
    return rm.tensor(scale * x.data, device=x.device)


# 4. TN 与非 TN 参数交替
def _weighted_sum_derivative(a, weight1, b, weight2):
    return (rm.tensor(weight1, device=a.device), rm.tensor(weight2, device=b.device))

@track_grad(_weighted_sum_derivative)
def weighted_sum(a, weight1, b, weight2):
    return rm.tensor(weight1 * a.data + weight2 * b.data, device=a.device)


# 5. 带默认参数
def _pow_derivative(x, exponent=2.0):
    return (rm.tensor(exponent * (x.data ** (exponent - 1)), device=x.device),)

@track_grad(_pow_derivative)
def mypow(x, exponent=2.0):
    return rm.tensor(x.data ** exponent, device=x.device)


# 6. 使用 *args 的 grad_func
def _sum_derivative(*args):
    return tuple(rm.tensor(1., device=args[0].device) for _ in args)

@track_grad(_sum_derivative)
def mysum(*xs):
    return rm.tensor(sum(x.data for x in xs), device=xs[0].device)


# 7. 使用 **kwargs 的 grad_func
def _sum_kwargs_derivative(**kwargs):
    vals = list(kwargs.values())
    return tuple(rm.tensor(1., device=vals[0].device) for _ in vals)

@track_grad(_sum_kwargs_derivative)
def mysum_kwargs(**kwargs):
    vals = list(kwargs.values())
    return rm.tensor(sum(x.data for x in vals), device=vals[0].device)


# ==================== 测试类 ====================
class TestTrackGrad(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.current_test_name = self._testMethodName
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")

    def tearDown(self):
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")

    # ---------- 1. 单输入基础函数 ----------
    def test_single_input_functions(self):
        """测试单输入基础函数的前向和反向传播"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"单输入基础函数 - {device}"
            start_time = time.time()
            try:
                np_x = np.random.randn(3, 4).astype(np.float32)
                rm_x = rm.tensor(np_x, requires_grad=True, device=device)

                # 前向传播
                rm_result = mysin(rm_x)
                expected = np.sin(np_x)
                forward_passed = np.allclose(rm_result.data, expected, atol=1e-5)

                # 反向传播
                rm_result.sum().backward()
                grad_expected = np.cos(np_x)
                backward_passed = np.allclose(rm_x.grad.data, grad_expected, atol=1e-5)

                passed = forward_passed and backward_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"单输入基础函数测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 2. 多输入位置参数 ----------
    def test_multi_input_positional(self):
        """测试多输入函数使用位置参数"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"多输入位置参数 - {device}"
            start_time = time.time()
            try:
                np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                np_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
                rm_a = rm.tensor(np_a, requires_grad=True, device=device)
                rm_b = rm.tensor(np_b, requires_grad=True, device=device)

                rm_result = myadd(rm_a, rm_b)
                forward_passed = np.allclose(rm_result.data, np_a + np_b)

                rm_result.sum().backward()
                grad_a_passed = np.allclose(rm_a.grad.data, np.ones_like(np_a))
                grad_b_passed = np.allclose(rm_b.grad.data, np.ones_like(np_b))

                passed = forward_passed and grad_a_passed and grad_b_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"多输入位置参数测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 3. 多输入关键字参数 ----------
    def test_multi_input_keyword(self):
        """测试多输入函数使用关键字参数"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"多输入关键字参数 - {device}"
            start_time = time.time()
            try:
                np_a = np.array([1.0, 2.0], dtype=np.float32)
                np_b = np.array([3.0, 4.0], dtype=np.float32)
                rm_a = rm.tensor(np_a, requires_grad=True, device=device)
                rm_b = rm.tensor(np_b, requires_grad=True, device=device)

                rm_result = myadd(x=rm_a, y=rm_b)
                forward_passed = np.allclose(rm_result.data, np_a + np_b)

                rm_result.sum().backward()
                grad_a_passed = np.allclose(rm_a.grad.data, np.ones_like(np_a))
                grad_b_passed = np.allclose(rm_b.grad.data, np.ones_like(np_b))

                passed = forward_passed and grad_a_passed and grad_b_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"多输入关键字参数测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 4. 关键字参数顺序打乱 ----------
    def test_keyword_order_shuffled(self):
        """测试关键字参数顺序与函数签名不一致时梯度仍正确匹配"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"关键字参数顺序打乱 - {device}"
            start_time = time.time()
            try:
                np_a = np.array([1.0, 2.0], dtype=np.float32)
                np_b = np.array([3.0, 4.0], dtype=np.float32)
                rm_a = rm.tensor(np_a, requires_grad=True, device=device)
                rm_b = rm.tensor(np_b, requires_grad=True, device=device)

                # 故意打乱关键字顺序: y 在前，x 在后
                rm_result = myadd(y=rm_b, x=rm_a)
                forward_passed = np.allclose(rm_result.data, np_a + np_b)

                rm_result.sum().backward()
                grad_a_passed = np.allclose(rm_a.grad.data, np.ones_like(np_a))
                grad_b_passed = np.allclose(rm_b.grad.data, np.ones_like(np_b))

                passed = forward_passed and grad_a_passed and grad_b_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"关键字参数顺序打乱测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 5. 非 TN 参数在前 ----------
    def test_non_tn_param_before(self):
        """测试非 TN 参数出现在 TN 参数之前"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"非 TN 参数在前 - {device}"
            start_time = time.time()
            try:
                np_x = np.array([2.0, 3.0], dtype=np.float32)
                rm_x = rm.tensor(np_x, requires_grad=True, device=device)
                scale = 3.0

                rm_result = myscale(scale, rm_x)
                forward_passed = np.allclose(rm_result.data, scale * np_x)

                rm_result.sum().backward()
                grad_expected = np.full_like(np_x, scale)
                backward_passed = np.allclose(rm_x.grad.data, grad_expected)

                passed = forward_passed and backward_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"非 TN 参数在前测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 6. TN 与非 TN 参数交替 ----------
    def test_mixed_tn_non_tn(self):
        """测试 TN 参数与非 TN 参数交替出现"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"TN 与非 TN 交替 - {device}"
            start_time = time.time()
            try:
                np_a = np.array([1.0, 2.0], dtype=np.float32)
                np_b = np.array([3.0, 4.0], dtype=np.float32)
                rm_a = rm.tensor(np_a, requires_grad=True, device=device)
                rm_b = rm.tensor(np_b, requires_grad=True, device=device)
                w1, w2 = 2.0, 3.0

                rm_result = weighted_sum(rm_a, w1, rm_b, w2)
                forward_passed = np.allclose(rm_result.data, w1 * np_a + w2 * np_b)

                rm_result.sum().backward()
                grad_a_passed = np.allclose(rm_a.grad.data, np.full_like(np_a, w1))
                grad_b_passed = np.allclose(rm_b.grad.data, np.full_like(np_b, w2))

                passed = forward_passed and grad_a_passed and grad_b_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"TN 与非 TN 交替测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 7. 部分输入不需要梯度 ----------
    def test_partial_requires_grad(self):
        """测试部分输入 requires_grad=False"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"部分输入不需要梯度 - {device}"
            start_time = time.time()
            try:
                np_a = np.array([1.0, 2.0], dtype=np.float32)
                np_b = np.array([3.0, 4.0], dtype=np.float32)
                rm_a = rm.tensor(np_a, requires_grad=True, device=device)
                rm_b = rm.tensor(np_b, requires_grad=False, device=device)

                rm_result = myadd(rm_a, rm_b)
                rm_result.sum().backward()

                grad_a_passed = np.allclose(rm_a.grad.data, np.ones_like(np_a))
                grad_b_none = rm_b.grad is None

                passed = grad_a_passed and grad_b_none
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"部分输入不需要梯度测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 8. 带默认参数 ----------
    def test_default_param(self):
        """测试带默认参数的场景"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"默认参数 - {device}"
            start_time = time.time()
            try:
                np_x = np.array([2.0, 3.0], dtype=np.float32)
                rm_x = rm.tensor(np_x, requires_grad=True, device=device)

                # 使用默认值 exponent=2.0
                rm_result = mypow(rm_x)
                forward_passed = np.allclose(rm_result.data, np_x ** 2)

                rm_result.sum().backward()
                grad_expected = 2.0 * np_x
                backward_passed = np.allclose(rm_x.grad.data, grad_expected)

                passed = forward_passed and backward_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"默认参数测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 9. 覆盖默认参数 ----------
    def test_override_default_param(self):
        """测试覆盖默认参数"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"覆盖默认参数 - {device}"
            start_time = time.time()
            try:
                np_x = np.array([2.0, 3.0], dtype=np.float32)
                rm_x = rm.tensor(np_x, requires_grad=True, device=device)

                # 覆盖默认 exponent=2.0 为 3.0
                rm_result = mypow(rm_x, exponent=3.0)
                forward_passed = np.allclose(rm_result.data, np_x ** 3)

                rm_result.sum().backward()
                grad_expected = 3.0 * (np_x ** 2)
                backward_passed = np.allclose(rm_x.grad.data, grad_expected)

                passed = forward_passed and backward_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"覆盖默认参数测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 10. 使用 *args 的 grad_func ----------
    def test_varargs_grad_func(self):
        """测试 grad_func 使用 *args（变长参数）"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"变长参数 grad_func - {device}"
            start_time = time.time()
            try:
                np_a = np.array([1.0, 2.0], dtype=np.float32)
                np_b = np.array([3.0, 4.0], dtype=np.float32)
                np_c = np.array([5.0, 6.0], dtype=np.float32)
                rm_a = rm.tensor(np_a, requires_grad=True, device=device)
                rm_b = rm.tensor(np_b, requires_grad=True, device=device)
                rm_c = rm.tensor(np_c, requires_grad=True, device=device)

                rm_result = mysum(rm_a, rm_b, rm_c)
                forward_passed = np.allclose(rm_result.data, np_a + np_b + np_c)

                rm_result.sum().backward()
                grad_a_passed = np.allclose(rm_a.grad.data, np.ones_like(np_a))
                grad_b_passed = np.allclose(rm_b.grad.data, np.ones_like(np_b))
                grad_c_passed = np.allclose(rm_c.grad.data, np.ones_like(np_c))

                passed = forward_passed and grad_a_passed and grad_b_passed and grad_c_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"变长参数 grad_func 测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 11. 使用 **kwargs 的 grad_func ----------
    def test_kwargs_grad_func(self):
        """测试 grad_func 使用 **kwargs"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"kwargs grad_func - {device}"
            start_time = time.time()
            try:
                np_a = np.array([1.0, 2.0], dtype=np.float32)
                np_b = np.array([3.0, 4.0], dtype=np.float32)
                rm_a = rm.tensor(np_a, requires_grad=True, device=device)
                rm_b = rm.tensor(np_b, requires_grad=True, device=device)

                rm_result = mysum_kwargs(a=rm_a, b=rm_b)
                forward_passed = np.allclose(rm_result.data, np_a + np_b)

                rm_result.sum().backward()
                grad_a_passed = np.allclose(rm_a.grad.data, np.ones_like(np_a))
                grad_b_passed = np.allclose(rm_b.grad.data, np.ones_like(np_b))

                passed = forward_passed and grad_a_passed and grad_b_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"kwargs grad_func 测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 12. 与 PyTorch 对比验证 ----------
    def test_pytorch_comparison(self):
        """使用 PyTorch 验证梯度计算正确性"""
        if not TORCH_AVAILABLE:
            return

        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"PyTorch 对比验证 - {device}"
            start_time = time.time()
            try:
                np_a = np.random.randn(3, 4).astype(np.float32)
                np_b = np.random.randn(3, 4).astype(np.float32)

                # Riemann
                rm_a = rm.tensor(np_a.copy(), requires_grad=True, device=device)
                rm_b = rm.tensor(np_b.copy(), requires_grad=True, device=device)
                rm_result = myadd(rm_a, rm_b)
                rm_loss = rm_result.sum()
                rm_loss.backward()

                # PyTorch
                torch_a = torch.tensor(np_a.copy(), requires_grad=True)
                torch_b = torch.tensor(np_b.copy(), requires_grad=True)
                torch_result = torch_a + torch_b
                torch_loss = torch_result.sum()
                torch_loss.backward()

                grad_a_passed = compare_gradients(rm_a, torch_a)
                grad_b_passed = compare_gradients(rm_b, torch_b)

                passed = grad_a_passed and grad_b_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"PyTorch 对比验证失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise

    # ---------- 13. 非 TN 参数关键字调用 ----------
    def test_non_tn_keyword(self):
        """测试非 TN 参数使用关键字调用"""
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            case_name = f"非 TN 关键字调用 - {device}"
            start_time = time.time()
            try:
                np_x = np.array([2.0, 3.0], dtype=np.float32)
                rm_x = rm.tensor(np_x, requires_grad=True, device=device)

                # scale 通过关键字传入
                rm_result = myscale(scale=4.0, x=rm_x)
                forward_passed = np.allclose(rm_result.data, 4.0 * np_x)

                rm_result.sum().backward()
                backward_passed = np.allclose(rm_x.grad.data, np.full_like(np_x, 4.0))

                passed = forward_passed and backward_passed
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
                self.assertTrue(passed, f"非 TN 关键字调用测试失败: {case_name}")
            except Exception as e:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} - {str(e)}")
                raise


# ==================== 独立脚本入口 ====================
if __name__ == '__main__':
    IS_RUNNING_AS_SCRIPT = True

    # 清屏
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行 track_grad 测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()

    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTrackGrad)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(test_suite)

    # 打印测试统计摘要
    stats.print_summary()

    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)
