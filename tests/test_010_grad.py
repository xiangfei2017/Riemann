"""
tests/test_010_grad.py
======================
riemann.autograd.grad 函数 + TN.d() 方法测试。

被测对象:
    riemann.autograd.grad(outputs, inputs, grad_outputs=None,
                          create_graph=False, retain_graph=None,
                          allow_unused=False)
        —— 自动微分的核心反向传播函数。对 outputs 中的每个张量，
          沿计算图反向传播，返回 inputs 中每个张量的梯度。
          outputs 必须是标量（或非标量 + 提供 grad_outputs）。

    TN.d(*args, create_graph=False)
        —— TN 类的高阶混合偏导数方法。当前 self 必须是标量张量。
          args 支持两种形式可混用:
            - TN          对该变量求一阶
            - (TN, k)     对该变量求 k 阶 (k >= 1)
          同一变量多次出现时阶数自动累加。
          输出形状 = 各变量形状按 args 顺序扁平拼接。

用例组织:
    TestAutogradGrad   —— autograd.grad 函数，11 个数据驱动子用例
        覆盖: 标量梯度 / 多输入 / allow_unused / grad_outputs /
              create_graph 高阶 / 错误场景
    TestTensorD        —— TN.d() 方法，15 个数据驱动子用例
        覆盖: 旧版标量兼容 / 向量变量一阶 & Hessian /
              多向量混合偏导 / 三阶张量 / args 合并 / 异常场景

运行方式:
    python tests/test_010_grad.py           独立脚本 (逐 case 打印 + 统计表)
    python -m pytest tests/test_010_grad.py pytest 模式 (仅断言零噪声)

风格与 test_034_func_grad.py 一致:
    - 一个 class 一个被测对象，每个 class 一个 test_xxx 方法
    - 数据驱动: cases 列表 → 逐个子用例
    - StatisticsCollector 按方法名分组统计
"""

import unittest
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import riemann as rm
    from riemann.autograd import grad as rm_grad
except ImportError as e:
    print(f"无法导入 riemann 模块: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 终端颜色 / 清屏 / 全局开关
# ---------------------------------------------------------------------------

class Colors:
    """ANSI 终端颜色常量。仅在独立脚本运行时使用。"""

    HEADER = '\033[95m'  # 紫色，标题
    OKBLUE = '\033[94m'  # 蓝色，元信息
    OKGREEN = '\033[92m' # 绿色，通过
    FAIL   = '\033[91m'  # 红色，失败
    ENDC   = '\033[0m'   # 颜色重置
    BOLD   = '\033[1m'   # 粗体


def clear_screen():
    """跨平台清屏: Windows 用 cls, Unix 用 clear。"""
    os.system('cls' if os.name == 'nt' else 'clear')


# ---------------------------------------------------------------------------
# 数值比较工具
# ---------------------------------------------------------------------------

def _close(a, b, atol=1e-5):
    """
    比较两个 TN / numpy array / Python 数值是否在容差内接近。

    自动从 TN 提取 _array 属性，转换为 numpy array，
    若类型/形状不兼容则退化为 Python == 比较。

    参数:
        a, b: 任意可比较对象 (TN / np.ndarray / float / int)
        atol: 绝对容差，默认 1e-5

    返回:
        bool — a ≈ b 在容差内且形状相同
    """
    def _to_arr(x):
        try:
            return np.asarray(x._array if hasattr(x, '_array') else x)
        except Exception:
            return x
    a = _to_arr(a)
    b = _to_arr(b)
    try:
        if a.shape != b.shape:
            return False
        return np.allclose(a, b, atol=atol)
    except Exception:
        return a == b


# ---------------------------------------------------------------------------
# 测试统计收集器
# ---------------------------------------------------------------------------

class StatisticsCollector:
    """
    测试运行时的分组统计收集器。

    按方法名 (self._testMethodName) 分组，每组记录:
        - 子用例总数 / 通过数 / 耗时
    最后 print_summary() 打印与 test_034_func_grad.py 风格一致的分组统计表。
    """

    def __init__(self):
        self.total_cases = 0          # 全局子用例总数
        self.passed_cases = 0         # 全局通过数
        self.total_time = 0.0         # 全局累计耗时
        self.function_stats: dict[str, dict] = {}  # {method_name: {total, passed, time}}
        self.current = None           # 当前正在统计的方法名
        self._t0 = 0.0                # 当前方法开始时间戳

    def start_function(self, name):
        """开始一个方法级别的统计，记录起始时间戳。"""
        self.current = name
        self._t0 = time.time()
        self.function_stats.setdefault(name, {"total": 0, "passed": 0, "time": 0.0})

    def add_result(self, case_name, passed):
        """记录单个子用例的结果。"""
        self.total_cases += 1
        if passed:
            self.passed_cases += 1
        if self.current:
            d = self.function_stats[self.current]
            d["total"] += 1
            if passed:
                d["passed"] += 1

    def end_function(self):
        """结束当前方法的统计，累加耗时到该方法。"""
        if self.current:
            elapsed = time.time() - self._t0
            self.function_stats[self.current]["time"] += elapsed
            self.total_time += elapsed

    @staticmethod
    def _w(text):
        """计算字符串的终端显示宽度 (中文字符占 2 列，其他 1 列)。"""
        return sum(2 if '\u4e00' <= c <= '\u9fff' else 1 for c in str(text))

    def print_summary(self):
        """
        打印与 test_034_func_grad.py 完全一致的精美分组统计表。
        包含: 总用例 / 通过数 / 通过率 / 总耗时 / 各用例组详情表格 (边框 + 对齐)。
        """
        headers = ['用例名', '通过/总数', '通过率', '耗时(秒)']
        hw = [self._w(h) for h in headers]

        max_name_w = hw[0]
        for fn in self.function_stats.keys():
            max_name_w = max(max_name_w, self._w(fn))

        cw = [
            max(max_name_w, hw[0]) + 2,  # 用例名列宽
            hw[1] + 4,                    # 通过/总数 列宽
            hw[2] + 4,                    # 通过率 列宽
            hw[3] + 4,                    # 耗时 列宽
        ]
        tw = sum(cw)

        # —— 表头 ——
        print("\n" + "=" * tw)
        print(f"{Colors.BOLD}测试统计摘要{Colors.ENDC}")
        print("=" * tw)
        print(f"总测试用例数: {self.total_cases}")
        c_ok = Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL
        print(f"通过测试用例数: {c_ok}{self.passed_cases}{Colors.ENDC}")
        rate = self.passed_cases / self.total_cases * 100 if self.total_cases else 0
        print(f"测试通过率: {c_ok}{rate:.2f}%{Colors.ENDC}")
        print(f"总耗时: {self.total_time:.4f} 秒")

        # —— 分组详情表 ——
        print("\n各用例测试详情:")
        print("-" * tw)

        # 表头行
        hl = ""
        for i, h in enumerate(headers):
            hl += h + " " * (cw[i] - hw[i])
        print(hl)
        print("-" * tw)

        # 数据行
        for fn, st in self.function_stats.items():
            pr = st["passed"] / st["total"] * 100 if st["total"] else 0
            sc = Colors.OKGREEN if pr == 100 else Colors.FAIL

            fn_w = self._w(fn)
            fn_pad = cw[0] - fn_w
            pt = f"{st['passed']}/{st['total']}"
            pt_w = self._w(pt)
            pt_pad = cw[1] - pt_w
            pr_s = f"{pr:.2f}"
            pr_w = self._w(pr_s)
            pr_pad = cw[2] - pr_w
            ti = f"{st['time']:.4f}"
            ti_w = self._w(ti)
            ti_pad = cw[3] - ti_w

            print(
                f"{fn}{' ' * fn_pad}"
                f"{pt}{' ' * pt_pad}"
                f"{sc}{pr_s}{' ' * pr_pad}{Colors.ENDC}"
                f"{ti}{' ' * ti_pad}"
            )

        print("=" * tw)


# 全局实例 — 两个 Test 类共享同一个 collector
stats = StatisticsCollector()

# 运行模式开关:
#   False (pytest)  —— 不打印逐 case 输出，StatisticsCollector 静默运行
#   True  (独立脚本) —— 逐 case 打印通过/失败 + 末尾统计表
IS_RUNNING_AS_SCRIPT = False


def _run_cases(test_class, cases):
    """
    数据驱动执行器 (辅助函数，实际未被两个 Test 类直接调用，
    仅保留供未来扩展使用)。

    参数:
        test_class: unittest.TestCase 子类 (将被实例化)
        cases: 子用例列表，每项为 dict:
            - name: 子用例名 (str)
            - fn: 闭包 fn(self, case) —— 实际测试逻辑
            - expect_error: bool, 若 True 则 fn 抛出异常视为通过
    """
    instance = test_class()
    instance.setUp()
    method_name = instance.current_test_name
    stats.start_function(method_name)

    try:
        for case in cases:
            case_name = case["name"]
            fn = case["fn"]
            expected_error = case.get("expect_error", False)
            try:
                fn(instance, case)
                passed = not expected_error
            except Exception as e:
                passed = expected_error

            stats.add_result(case_name, passed)
            status = "通过" if passed else "失败"
            if IS_RUNNING_AS_SCRIPT:
                color = Colors.OKGREEN if passed else Colors.FAIL
                print(f"  [{color}{status}{Colors.ENDC}] {case_name}")

            if not passed:
                raise AssertionError(f"子用例失败: {case_name}")

    finally:
        stats.end_function()


# ===================================================================
# 第一组: TestAutogradGrad —— autograd.grad 函数数据驱动测试
# ===================================================================

def _grad_setup(x_np, y_np=None, device="cpu"):
    """
    构造 requires_grad=True 的标量张量。

    参数:
        x_np: 标量或 0-d array-like —— 第一个输入
        y_np: 可选标量 —— 第二个输入 (多输入测试用)
        device: "cpu" 或其他设备
    返回:
        (x_tensor,) 或 (x_tensor, y_tensor) —— 后者当 y_np 非 None 时
    """
    x = rm.tensor(np.asarray(x_np, dtype=np.float64), requires_grad=True, device=device)
    if y_np is None:
        return x, None
    y = rm.tensor(np.asarray(y_np, dtype=np.float64), requires_grad=True, device=device)
    return x, y


def _grad_setup_matrix(x_np, grad_out_np=None, device="cpu"):
    """
    构造 requires_grad=True 的矩阵张量 + 可选 grad_outputs。

    参数:
        x_np: 2-D array-like —— 要对其求导的张量
        grad_out_np: 与 x_np 同形状 —— 非标量输出场景的权重
    返回:
        (x_tensor, grad_out_tensor | None)
    """
    x = rm.tensor(np.asarray(x_np, dtype=np.float64), requires_grad=True, device=device)
    if grad_out_np is None:
        return x, None
    go = rm.tensor(np.asarray(grad_out_np, dtype=np.float64), device=device)
    return x, go


class TestAutogradGrad(unittest.TestCase):
    """
    riemann.autograd.grad 函数数据驱动测试。

    test_grad_function 方法内定义 11 个子用例，覆盖:
        1-2  标量对标的一阶导数
        3-4  多输入函数的联合梯度
        5    allow_unused=True 对未参与计算变量的容错
        6    张量输出 + grad_outputs 的非标量梯度
        7-8  create_graph=True 支持二阶 / 三阶高阶导数
        9    异常: 输入无 requires_grad
        10   异常: 非标量输出无 grad_outputs
        11   异常: grad_outputs 形状不匹配
    """

    def setUp(self):
        """
        每个用例组开始时调用 (unittest.TestCase 钩子)。
        - 重置随机种子 (可复现)
        - 记录方法名供统计
        - 独立脚本模式下打印 "开始测试: ..." 标题
        """
        np.random.seed(42)
        self.current_test_name = self._testMethodName
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")

    def tearDown(self):
        """
        每个用例组结束时调用 (unittest.TestCase 钩子)。
        独立脚本模式下关闭统计周期。
        """
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()

    def test_grad_function(self):
        """autograd.grad 的基本 / 多输入 / 未使用变量 / 张量 / 高阶 / 错误场景"""

        # ================================================================
        # 子用例定义: 每个 fn(self, case) 闭包内含精确断言
        # ================================================================
        cases = []

        # —— 1. f(x) = x² —— 在 x=2.0 处一阶导数 = 2x = 4.0 ——
        def scalar_fn(self, case):
            x, _ = _grad_setup(case["x"])
            y = x ** 2
            self.assertTrue(_close(rm_grad(y, x)[0], 2 * case["x"]), case["name"])
        cases.append({"name": "scalar_gradient_x2_at_2", "fn": scalar_fn, "x": 2.0})

        # —— 2. f(x) = x³ + 2x —— 在 x=3.0 处 = 3x²+2 = 29 ——
        def scalar_fn2(self, case):
            x, _ = _grad_setup(case["x"])
            y = x ** 3 + 2 * x
            self.assertTrue(_close(rm_grad(y, x)[0], 3 * case["x"] ** 2 + 2), case["name"])
        cases.append({"name": "scalar_gradient_x3p2x_at_3", "fn": scalar_fn2, "x": 3.0})

        # —— 3. 多输入 f(x,y) = 2x + y² —— 在 (3,4) 处 grad_x=2, grad_y=2y=8 ——
        def multi_fn(self, case):
            x, y = _grad_setup(case["x"], case["y"])
            z = 2 * x + y ** 2
            gx, gy = rm_grad(z, [x, y])
            self.assertTrue(_close(gx, 2.0) and _close(gy, 2 * case["y"]), case["name"])
        cases.append({"name": "multi_input_2x_py2_at_3_4", "fn": multi_fn, "x": 3.0, "y": 4.0})

        # —— 4. 多输入 f(x,y) = x·y + sin(x) —— 在 (π,2) 处 grad_x=y+cos(x), grad_y=x ——
        def multi_fn2(self, case):
            x, y = _grad_setup(np.pi, 2.0)
            z = x * y + rm.sin(x)
            gx, gy = rm_grad(z, [x, y])
            self.assertTrue(_close(gx, 2.0 + np.cos(np.pi)) and _close(gy, np.pi), case["name"])
        cases.append({"name": "multi_input_xypysinx_at_pi_2", "fn": multi_fn2})

        # —— 5. allow_unused=True —— 对未参与计算的变量 y1 不报错 ——
        def unused_fn(self, case):
            x, y = _grad_setup(case["x"], case["y"])
            y1 = rm.tensor(case["unused"], requires_grad=True)  # 闭包，未在 z 中使用
            z = 2 * x + y ** 2
            gx, gy, gy1 = rm_grad(z, [x, y, y1], allow_unused=True)
            self.assertTrue(_close(gx, 2.0) and _close(gy, 2 * case["y"]), case["name"])
        cases.append({"name": "unused_var_allow_unused", "fn": unused_fn, "x": 3.0, "y": 4.0, "unused": 5.0})

        # —— 6. 张量 x ∈ R²ˣ², y=x² —— grad_outputs 全 1 → 2x ——
        def tensor_fn(self, case):
            x, go = _grad_setup_matrix(case["x_np"], case["grad_out_np"])
            y = x ** 2
            g = rm_grad(y, x, grad_outputs=go)[0]
            # 链式规则: grad_outputs · d/dx(x²) = grad_outputs · 2x
            self.assertTrue(_close(g, 2 * x._array * case["grad_out_np"]), case["name"])
        cases.append({
            "name": "tensor_gradient_with_grad_outputs",
            "fn": tensor_fn,
            "x_np": np.array([[1., 2.], [3., 4.]]),
            "grad_out_np": np.ones((2, 2)),
        })

        # —— 7. create_graph=True → 二阶 —— f(x)=x³, d²f/dx² = 6x ——
        def higher_fn(self, case):
            x, _ = _grad_setup(case["x"])
            y = x ** 3
            dy = rm_grad(y, x, create_graph=True)[0]
            d2y = rm_grad(dy, x)[0]
            self.assertTrue(_close(d2y, 6 * case["x"]), case["name"])
        cases.append({"name": "higher_order_x3_at_2", "fn": higher_fn, "x": 2.0})

        # —— 8. create_graph=True → 三阶 —— f(x)=x⁴, d³f/dx³ = 24 ——
        def third_fn(self, case):
            x, _ = _grad_setup(case["x"])
            y = x ** 4
            dy = rm_grad(y, x, create_graph=True)[0]
            d2y = rm_grad(dy, x, create_graph=True)[0]
            d3y = rm_grad(d2y, x)[0]
            self.assertTrue(_close(d3y, 24.0), case["name"])
        cases.append({"name": "third_order_x4_at_any", "fn": third_fn, "x": 1.0})

        # —— 9. 异常 —— requires_grad=False 的输入直接报 RuntimeError ——
        def err_nograd(self, case):
            x = rm.tensor(1.0)  # 默认 requires_grad=False
            rm_grad(x, x)
        cases.append({"name": "err_no_requires_grad", "fn": err_nograd, "expect_error": True})

        # —— 10. 异常 —— 非标量输出且无 grad_outputs ——
        def err_non_scalar(self, case):
            x = rm.tensor([1.0, 2.0], requires_grad=True)
            y = x ** 2
            rm_grad(y, x)
        cases.append({"name": "err_non_scalar_output_no_grad_outputs", "fn": err_non_scalar, "expect_error": True})

        # —— 11. 异常 —— grad_outputs 形状不匹配 ——
        def err_wrong_shape(self, case):
            x = rm.tensor([1.0, 2.0], requires_grad=True)
            y = x ** 2
            go = rm.tensor([1.0])  # 形状 (1,) 不匹配 y 的 (2,)
            rm_grad(y, x, grad_outputs=go)
        cases.append({"name": "err_grad_outputs_shape_mismatch", "fn": err_wrong_shape, "expect_error": True})

        # ================================================================
        # 逐子用例执行 + 计时 + 打印 + 统计
        # ================================================================
        for case in cases:
            case_name = case["name"]
            t0 = time.time()
            passed = True
            err = None
            try:
                case["fn"](self, case)
            except Exception as e:
                passed = case.get("expect_error", False)
                if not passed:
                    err = str(e)
                    raise AssertionError(
                        f"子用例 '{case_name}' 抛出异常: {type(e).__name__}: {e}"
                    ) from e

            time_taken = time.time() - t0
            stats.add_result(case_name, passed)

            # —— 逐 case 打印 (与 test_034_func_grad.py 风格一致) ——
            if IS_RUNNING_AS_SCRIPT:
                if passed:
                    print(f"  {Colors.OKGREEN}通过{Colors.ENDC}  {case_name}  ({time_taken:.4f}s)")
                else:
                    print(f"  {Colors.FAIL}失败{Colors.ENDC}  {case_name}  ({time_taken:.4f}s)"
                          + (f"  — {err}" if err else ""))

            if not passed:
                self.fail(f"子用例失败: {case_name}")


# ===================================================================
# 第二组: TestTensorD —— TN.d() 高阶混合偏导数方法
# ===================================================================

def _T(arr, rg=True):
    """快速构造 requires_grad=rg 的 float64 TN。"""
    return rm.tensor(np.asarray(arr, dtype=np.float64), requires_grad=rg)


class TestTensorD(unittest.TestCase):
    """
    TN.d() 高阶混合偏导数方法 —— 15 个数据驱动子用例。

    test_d_method 内部定义 4 个分组共 15 个子用例:
        旧版兼容 (4) —— 标量变量一阶/混合偏导/显式阶数/重复传参
        向量变量 (3) —— 梯度向量 / Hessian 对角 / 向量阶数语义
        混合偏导 (3) —— 两向量交叉矩阵 / 标量+向量 / 三阶张量
        args 合并 (2) —— 同变量 TN+tuple 阶数累加 / 显式多阶规格
        异常 (3) —— self 非标量 / 无 args / k < 1
    """

    def setUp(self):
        """
        每个用例组开始时调用 (unittest.TestCase 钩子)。
        - 重置随机种子 (可复现)
        - 记录方法名供统计
        - 独立脚本模式下打印 "开始测试: ..." 标题
        """
        np.random.seed(42)
        self.current_test_name = self._testMethodName
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")

    def tearDown(self):
        """
        每个用例组结束时调用 (unittest.TestCase 钩子)。
        独立脚本模式下关闭统计周期。
        """
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()

    def test_d_method(self):
        """TN.d() — 兼容旧版标量变量 + 新增向量变量 / 混合偏导 / args 合并 / 异常"""

        cases = []

        # ============================================================
        # 分组 A: 旧版标量变量兼容 (TN.d 原本仅支持 0-d 变量)
        # ============================================================

        # A1: f=x³y², d(x) → 3x²y² = 3·4·9 = 108
        def d1(self, _):
            x, y = _T(2.0), _T(3.0)
            self.assertTrue(_close((x**3 * y**2).d(x), 108.0))
        cases.append({"name": "compat_scalar_first", "fn": d1})

        # A2: d²f/dxdy = 6x²y = 6·4·3 = 72
        def d2(self, _):
            x, y = _T(2.0), _T(3.0)
            self.assertTrue(_close((x**3 * y**2).d(x, y), 72.0))
        cases.append({"name": "compat_scalar_mixed", "fn": d2})

        # A3: 显式阶数 (x, 2) → 6xy² = 6·2·9 = 108
        def d3(self, _):
            x, y = _T(2.0), _T(3.0)
            self.assertTrue(_close((x**3 * y**2).d((x, 2)), 108.0))
        cases.append({"name": "compat_tuple_k", "fn": d3})

        # A4: 重复传参 d(x,x) 与 d((x,2)) 对 f=x³y² 等价
        def d4(self, _):
            x, y = _T(2.0), _T(3.0)
            f = x**3 * y**2
            self.assertTrue(_close(f.d(x, x), f.d((x, 2))))
        cases.append({"name": "compat_repeat_eq_tuple", "fn": d4})

        # ============================================================
        # 分组 B: 向量变量 (新核心能力)
        # ============================================================

        # B1: f(x)=Σx³, x∈R³ → 梯度向量 [3·1², 3·2², 3·3²] = [3,12,27]
        def d5(self, _):
            x = _T([1., 2., 3.])
            self.assertTrue(_close((x**3).sum().d(x), np.array([3., 12., 27.])))
        cases.append({"name": "vec_first_gradient", "fn": d5})

        # B2: f(x)=Σx³ → Hessian 矩阵 diag([6·1, 6·2, 6·3]) = diag([6,12,18])
        def d6(self, _):
            x = _T([1., 2., 3.])
            self.assertTrue(_close((x**3).sum().d(x, x), np.diag([6., 12., 18.])))
        cases.append({"name": "vec_second_hessian", "fn": d6})

        # B3: 向量场景下 repeat vs tuple 一致性
        def d7(self, _):
            x = _T([1., 2., 3.])
            f = (x**3).sum()
            self.assertTrue(_close(f.d(x, x), f.d((x, 2))))
        cases.append({"name": "vec_repeat_eq_tuple", "fn": d7})

        # ============================================================
        # 分组 C: 混合偏导矩阵
        # ============================================================

        # C1: f(x,y)=(x·y)², x∈R², y∈R² → ∂²f/∂x∂y 矩阵
        # 手算: s=x·y → f=s² → ∂f/∂x_i=2s·y_i → ∂²f/∂x_i∂y_j=2y_i·y_j + 2s·δ_ij
        # 在 (1,2), (3,4) 处 s=11 → [[28,12],[8,38]]
        def d8(self, _):
            x, y = _T([1., 2.]), _T([3., 4.])
            expected = np.array([[28., 12.], [8., 38.]])
            self.assertTrue(_close(((x * y).sum() ** 2).d(x, y), expected))
        cases.append({"name": "mixed_cross_matrix", "fn": d8})

        # C2: 标量 x + 向量 y — f(x,y)=x·Σy² → ∂²f/∂x∂y = 2y, shape=(3,)
        def d9(self, _):
            x, y = _T(2.0), _T([1., 2., 3.])
            r = (x * (y**2).sum()).d(x, y)
            self.assertTrue(_close(r, np.array([2., 4., 6.])) and r.shape == (3,))
        cases.append({"name": "scalar_vec_mixed", "fn": d9})

        # C3: 三阶张量 — f(x)=Σx⁴, x∈R³ → 三阶偏导张量 shape=(3,3,3)
        # 对角线 [i,i,i] = 24x_i = [24, 48, 72], 其余全零
        def d10(self, _):
            x = _T([1., 2., 3.])
            expected = np.zeros((3, 3, 3))
            for i in range(3):
                expected[i, i, i] = 24 * [1., 2., 3.][i]
            self.assertTrue(_close((x**4).sum().d(x, x, x), expected))
        cases.append({"name": "third_order_tensor", "fn": d10})

        # ============================================================
        # 分组 D: args 合并 (同一变量 TN + tuple 混用)
        # ============================================================

        # D1: f.d(x, (x,2)) —— 同一变量既直接传又用 tuple，阶数累加为 3
        # f=x³y² → d³f/dx³ = 6y² = 54
        def d11(self, _):
            x, y = _T(2.0), _T(3.0)
            self.assertTrue(_close((x**3 * y**2).d(x, (x, 2)), 54.0))
        cases.append({"name": "merge_triple_scalar", "fn": d11})

        # D2: f.d((x,2), y) —— 显式多阶规格: x 二阶 + y 一阶
        # f=x³y² → ∂³f/∂x²∂y: step1 grad x → 3x²y² → step2 grad x → 6xy² → step3 grad y → 12xy = 72
        def d12(self, _):
            x, y = _T(2.0), _T(3.0)
            self.assertTrue(_close((x**3 * y**2).d((x, 2), y), 72.0))
        cases.append({"name": "mixed_explicit_specs", "fn": d12})

        # ============================================================
        # 分组 E: 异常场景
        # ============================================================

        # E1: self 必须是标量张量
        def d13(self, _):
            x = _T([1., 2., 3.])
            x.d(x)
        cases.append({"name": "err_self_non_scalar", "fn": d13, "expect_error": True})

        # E2: 必须至少提供一个变量
        def d14(self, _):
            _T(2.0).d()
        cases.append({"name": "err_no_args", "fn": d14, "expect_error": True})

        # E3: k 必须 >= 1
        def d15(self, _):
            (_T(2.0)**2).d((_T(2.0), 0))
        cases.append({"name": "err_k_zero", "fn": d15, "expect_error": True})

        # ================================================================
        # 逐子用例执行 + 计时 + 打印 + 统计
        # ================================================================
        for case in cases:
            case_name = case["name"]
            t0 = time.time()
            passed = True
            err = None
            try:
                case["fn"](self, case)
            except Exception as e:
                passed = case.get("expect_error", False)
                if not passed:
                    err = str(e)
                    raise AssertionError(
                        f"子用例 '{case_name}' 抛出异常: {type(e).__name__}: {e}"
                    ) from e

            time_taken = time.time() - t0
            stats.add_result(case_name, passed)

            # —— 逐 case 打印 (与 test_034_func_grad.py 风格一致) ——
            if IS_RUNNING_AS_SCRIPT:
                if passed:
                    print(f"  {Colors.OKGREEN}通过{Colors.ENDC}  {case_name}  ({time_taken:.4f}s)")
                else:
                    print(f"  {Colors.FAIL}失败{Colors.ENDC}  {case_name}  ({time_taken:.4f}s)"
                          + (f"  — {err}" if err else ""))

            if not passed:
                self.fail(f"子用例失败: {case_name}")


# ===================================================================
# 脚本入口 —— 独立运行时启用逐 case 打印 + 统计表
# ===================================================================

if __name__ == '__main__':
    IS_RUNNING_AS_SCRIPT = True
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}===== test_010_grad.py ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAutogradGrad))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorD))

    result = unittest.TextTestRunner(verbosity=0).run(suite)
    stats.print_summary()
    sys.exit(0 if result.wasSuccessful() else 1)
