"""
tests/test_010_func_grad.py
============================
riemann.func 模块 —— torch.func 风格函数式自动微分接口测试。

被测函数:
    grad, grad_and_value, jvp, vjp,
    jacrev, jacfwd,
    hessian, vmap, functional_call

风格与 test_005_shape_operations.py 一致：
    - 一个 TestFuncs 类，每个被测函数对应一个 test_xxx 方法
    - StatisticsCollector 按方法名（= 函数名）分组统计
    - 独立脚本运行 → 详细打印 + 统计表
    - pytest 运行 → 仅断言，无噪声
"""

import unittest
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import riemann as rm
    from riemann import func, tensor
except ImportError as e:
    print(f"无法导入 riemann 模块: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Colors / StatisticsCollector / IS_RUNNING_AS_SCRIPT（与 test_005 一致）
# ---------------------------------------------------------------------------

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


class StatisticsCollector:
    def __init__(self):
        self.total_cases = 0
        self.passed_cases = 0
        self.total_time = 0.0
        self.function_stats = {}
        self.current_function = None
        self.current_function_start_time = 0

    def start_function(self, function_name):
        self.current_function = function_name
        self.current_function_start_time = time.time()
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

    def end_function(self):
        if self.current_function:
            elapsed = time.time() - self.current_function_start_time
            self.function_stats[self.current_function]["time"] += elapsed
            self.total_time += elapsed

    @staticmethod
    def _display_width(text):
        w = 0
        for c in text:
            if '\u4e00' <= c <= '\u9fff':
                w += 2
            else:
                w += 1
        return w

    def print_summary(self):
        headers = ['用例名', '通过/总数', '通过率', '耗时(秒)']
        hw = [self._display_width(h) for h in headers]

        max_name_w = hw[0]
        for fn in self.function_stats.keys():
            max_name_w = max(max_name_w, self._display_width(fn))

        cw = [
            max(max_name_w, hw[0]) + 2,
            hw[1] + 4,
            hw[2] + 4,
            hw[3] + 4,
        ]
        tw = sum(cw)

        print("\n" + "=" * tw)
        print(f"{Colors.BOLD}测试统计摘要{Colors.ENDC}")
        print("=" * tw)
        print(f"总测试用例数: {self.total_cases}")
        c_ok = Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL
        print(f"通过测试用例数: {c_ok}{self.passed_cases}{Colors.ENDC}")
        rate = self.passed_cases / self.total_cases * 100 if self.total_cases else 0
        print(f"测试通过率: {c_ok}{rate:.2f}%{Colors.ENDC}")
        print(f"总耗时: {self.total_time:.4f} 秒")
        print("\n各用例测试详情:")
        print("-" * tw)

        hl = ""
        for i, h in enumerate(headers):
            hl += h + " " * (cw[i] - hw[i])
        print(hl)
        print("-" * tw)

        for fn, st in self.function_stats.items():
            pr = st["passed"] / st["total"] * 100 if st["total"] else 0
            sc = Colors.OKGREEN if pr == 100 else Colors.FAIL

            fn_w = self._display_width(fn)
            fn_pad = cw[0] - fn_w
            pt = f"{st['passed']}/{st['total']}"
            pt_w = self._display_width(pt)
            pt_pad = cw[1] - pt_w
            pr_s = f"{pr:.2f}"
            pr_w = self._display_width(pr_s)
            pr_pad = cw[2] - pr_w
            ti = f"{st['time']:.4f}"
            ti_w = self._display_width(ti)
            ti_pad = cw[3] - ti_w

            print(
                f"{fn}{' ' * fn_pad}"
                f"{pt}{' ' * pt_pad}"
                f"{sc}{pr_s}{' ' * pr_pad}{Colors.ENDC}"
                f"{ti}{' ' * ti_pad}"
            )

        print("=" * tw)


stats = StatisticsCollector()
IS_RUNNING_AS_SCRIPT = False


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _rm_tensor(arr, requires_grad=True):
    return tensor(np.asarray(arr, dtype=np.float64), requires_grad=requires_grad)


def _close(a, b, atol=1e-6, rtol=1e-6):
    if hasattr(a, "_array"):
        a = a._array
    if hasattr(b, "_array"):
        b = b._array
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol)


def _rm_ctensor(arr, requires_grad=True):
    """构造复数张量（dtype=complex128）。"""
    return tensor(np.asarray(arr, dtype=np.complex128), requires_grad=requires_grad)


def _run_case(case_name, fn):
    """执行单个测试用例，打印结果并返回 passed bool"""
    t0 = time.time()
    passed = False
    err = None
    try:
        passed = bool(fn())
    except Exception as e:
        err = str(e)

    time_taken = time.time() - t0
    stats.add_result(case_name, passed)

    if IS_RUNNING_AS_SCRIPT:
        if passed:
            print(f"  {Colors.OKGREEN}通过{Colors.ENDC}  {case_name}  ({time_taken:.4f}s)")
        else:
            print(f"  {Colors.FAIL}失败{Colors.ENDC}  {case_name}  ({time_taken:.4f}s)"
                  + (f"  — {err}" if err else ""))

    if not passed:
        raise AssertionError(f"{case_name}: {err or '结果不匹配'}")
    return passed


# ---------------------------------------------------------------------------
# TestFuncs
# ---------------------------------------------------------------------------

class TestFuncs(unittest.TestCase):
    """riemann.func 模块测试 —— 每个被测函数对应一个 test_xxx 方法"""

    def setUp(self):
        self.current_test_name = self._testMethodName
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")

    def tearDown(self):
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()

    # ---------------------------------------------------------------
    # grad
    # ---------------------------------------------------------------
    def test_grad(self):
        cases = [
            ("square_sum",         lambda x: (x ** 2).sum(),          (_rm_tensor(np.array([1.0, 2.0, 3.0])),),           0,        np.array([2.0, 4.0, 6.0])),
            ("cube_sum",           lambda x: (x ** 3).sum(),          (_rm_tensor(np.array([1.0, 2.0, 3.0])),),           0,        np.array([3.0, 12.0, 27.0])),
            ("dot_self",           lambda x: x @ x,                   (_rm_tensor(np.array([1.0, 2.0, 3.0])),),           0,        np.array([2.0, 4.0, 6.0])),
            ("dot_product",        lambda x, w: (x * w).sum(),         (_rm_tensor(np.array([1.0, 2.0])),
                                                                              _rm_tensor(np.array([0.5, 1.0]))),                 (0, 1),  (np.array([0.5, 1.0]), np.array([1.0, 2.0]))),
            ("quadratic_bilinear", lambda x, w: (x * x * w).sum(),     (_rm_tensor(np.array([1.0, 2.0])),
                                                                              _rm_tensor(np.array([0.5, 1.0]))),                 (0, 1),  (np.array([1.0, 4.0]), np.array([1.0, 4.0]))),
            ("partial_wrt_0",      lambda x, w: (x * w).sum(),         (_rm_tensor(np.array([1.0, 2.0])),
                                                                              _rm_tensor(np.array([0.5, 1.0]))),                 0,        np.array([0.5, 1.0])),
            ("matrix_sum_square",  lambda x: (x ** 2).sum(),          (_rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]])),), 0,        np.array([[2.0, 4.0], [6.0, 8.0]])),
        ]

        for desc, func_def, inputs_rm, argnums, expected in cases:
            _run_case(f"grad - {desc}", lambda: self._check_grad(func_def, inputs_rm, argnums, expected))

        # has_aux
        def f_aux(x): return (x ** 2).sum(), "aux"
        _run_case("grad - has_aux", lambda: self._check_grad_has_aux(f_aux))

        # non_scalar 异常
        _run_case("grad - non_scalar_raises", lambda: self._check_grad_raises())

        # argnums=1（非零位置）
        _run_case("grad - argnums_second_arg", lambda: self._check_grad_argnums_1())

        # create_graph=True 高阶可微
        _run_case("grad - create_graph_high_order", lambda: self._check_grad_create_graph())

        # has_aux + 多 argnums
        def f_aux_multi(x, y): return ((x * y).sum(), "aux")
        _run_case("grad - has_aux_multi_argnums",
                  lambda: self._check_grad_has_aux_multi(f_aux_multi))

        # kwargs 透传
        _run_case("grad - kwargs_passthrough", lambda: self._check_grad_kwargs())

        # 返回顺序与 argnums 一致
        _run_case("grad - argnums_order", lambda: self._check_grad_argnums_order())

    def _check_grad(self, func_def, inputs_rm, argnums, expected):
        df = func.grad(func_def, argnums=argnums)
        grad_val = df(*inputs_rm)
        if isinstance(expected, tuple):
            return (isinstance(grad_val, tuple) and len(grad_val) == len(expected)
                    and all(_close(g, e) for g, e in zip(grad_val, expected)))
        return _close(grad_val, expected)

    def _check_grad_has_aux(self, f_aux):
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        g, aux = func.grad(f_aux, has_aux=True)(x)
        return _close(g, np.array([2.0, 4.0, 6.0])) and aux == "aux"

    def _check_grad_raises(self):
        x = _rm_tensor(np.array([1.0, 2.0]))
        try:
            func.grad(lambda x: x ** 2)(x)
            return False  # 没抛异常 = 失败
        except RuntimeError:
            return True

    def _check_grad_argnums_1(self):
        f = lambda x, w: (x * w ** 2).sum()
        x = _rm_tensor(np.array([1.0, 2.0]))
        w = _rm_tensor(np.array([0.5, 1.0]))
        g = func.grad(f, argnums=1)(x, w)
        return _close(g, np.array([1.0, 4.0]))  # d(x*w^2)/dw = 2*x*w

    def _check_grad_create_graph(self):
        f = lambda x: (x ** 4).sum()
        d1 = func.grad(f, create_graph=True)
        # d1 本身还能再被 grad —— create_graph=True 保留计算图
        d2y = func.grad(lambda a: d1(a).sum())(_rm_tensor(np.array([1.0, 2.0, 3.0])))
        return _close(d2y, 12.0 * np.array([1.0, 4.0, 9.0]))

    def _check_grad_has_aux_multi(self, f_aux):
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        g, aux = func.grad(f_aux, argnums=(0, 1), has_aux=True)(x, y)
        return (aux == "aux" and isinstance(g, tuple) and len(g) == 2
                and _close(g[0], np.array([3.0, 4.0]))    # d(x*y)/dx = y
                and _close(g[1], np.array([1.0, 2.0])))

    def _check_grad_kwargs(self):
        x = _rm_tensor(np.array([1.0, 2.0]))
        g = func.grad(lambda a, scale=1.0: scale * (a ** 2).sum())(x, scale=3.0)
        return _close(g, np.array([6.0, 12.0]))

    def _check_grad_argnums_order(self):
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        g = func.grad(lambda a, b: (a * b).sum(), argnums=(1, 0))(x, y)
        return _close(g[0], np.array([1.0, 2.0])) and _close(g[1], np.array([3.0, 4.0]))

    # ---------------------------------------------------------------
    # grad_and_value
    # ---------------------------------------------------------------
    def test_grad_and_value(self):
        _run_case("grad_and_value - single_input",
                  lambda: self._check_gav_single())
        _run_case("grad_and_value - multi_input",
                  lambda: self._check_gav_multi())
        _run_case("grad_and_value - has_aux",
                  lambda: self._check_gav_aux())
        _run_case("grad_and_value - matches_grad_plus_value",
                  lambda: self._check_gav_consistency())

        # create_graph=True 高阶可微
        _run_case("grad_and_value - create_graph",
                  lambda: self._check_gav_create_graph())

        # has_aux + create_graph 组合
        _run_case("grad_and_value - has_aux_and_create_graph",
                  lambda: self._check_gav_aux_create_graph())

    def _check_gav_single(self):
        f = lambda x: (x ** 2).sum()
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        g, v = func.grad_and_value(f)(x)
        return _close(g, np.array([2.0, 4.0, 6.0])) and _close(v, 14.0)

    def _check_gav_multi(self):
        f = lambda x, w: (x * w).sum()
        x = _rm_tensor(np.array([1.0, 2.0]))
        w = _rm_tensor(np.array([0.5, 1.0]))
        (gx, gw), v = func.grad_and_value(f, argnums=(0, 1))(x, w)
        return _close(v, 2.5) and _close(gx, np.array([0.5, 1.0])) and _close(gw, np.array([1.0, 2.0]))

    def _check_gav_aux(self):
        f = lambda x: ((x ** 2).sum(), "aux_value")
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        g, v, aux = func.grad_and_value(f, has_aux=True)(x)
        return _close(g, np.array([2.0, 4.0, 6.0])) and _close(v, 14.0) and aux == "aux_value"

    def _check_gav_consistency(self):
        f = lambda x, w: (x ** 2 * w).sum()
        x = _rm_tensor(np.array([1.0, 2.0]))
        w = _rm_tensor(np.array([0.5, 1.0]))
        g_av, v_av = func.grad_and_value(f, argnums=0)(x, w)
        g_direct = func.grad(f, argnums=0)(x, w)
        return _close(g_av, g_direct) and _close(v_av, f(x, w))

    def _check_gav_create_graph(self):
        f = lambda x: (x ** 3).sum()
        gav = func.grad_and_value(f, create_graph=True)
        g, v = gav(_rm_tensor(np.array([1.0, 2.0, 3.0])))
        # g = 3*x^2, d(g.sum())/dx = 6*x
        g2 = func.grad(lambda a: gav(a)[0].sum())(_rm_tensor(np.array([1.0, 2.0, 3.0])))
        return _close(g2, 6.0 * np.array([1.0, 2.0, 3.0]))

    def _check_gav_aux_create_graph(self):
        def f_aux(x): return ((x ** 3).sum(), "meta_info")
        gav = func.grad_and_value(f_aux, has_aux=True, create_graph=True)
        g, v, aux = gav(_rm_tensor(np.array([1.0, 2.0, 3.0])))
        # g = 3x^2, 对 g.sum() 再求 grad 得 6x
        g2 = func.grad(lambda a: gav(a)[0].sum())(_rm_tensor(np.array([1.0, 2.0, 3.0])))
        return aux == "meta_info" and _close(g2, 6.0 * np.array([1.0, 2.0, 3.0]))

    # ---------------------------------------------------------------
    # jvp
    # ---------------------------------------------------------------
    def test_jvp(self):
        cases = [
            ("scalar_output",          lambda x: (x ** 2).sum() + x.sum(),  np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5]), 20.0,         7.5),
            ("vector_output",          lambda x: x ** 2,                     np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 9.0]), np.array([2.0, 8.0, 18.0])),
            ("matrix_input_to_scalar", lambda x: (x ** 2).sum(),             np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 1.0], [1.0, 1.0]]), 30.0, 20.0),
        ]
        for desc, f, p_np, t_np, exp_out, exp_jvp in cases:
            _run_case(f"jvp - {desc}",
                      lambda d=desc, ff=f, pp=p_np, tt=t_np, eo=exp_out, ej=exp_jvp:
                          self._check_jvp(ff, pp, tt, eo, ej))

        _run_case("jvp - matches_finite_diff", lambda: self._check_jvp_fd())

        # 多输入 primals/tangents
        _run_case("jvp - multi_input",
                  lambda: self._check_jvp_multi_input())

        # create_graph=True
        _run_case("jvp - create_graph",
                  lambda: self._check_jvp_create_graph())

        # tangents=None：单输入时方向取标量 1.0
        _run_case("jvp - default_tangent", lambda: self._check_jvp_default_tangent())

        # 多输出 func
        _run_case("jvp - multi_output", lambda: self._check_jvp_multi_output())

        # 复数：两次转置需抵消共轭
        _run_case("jvp - complex", lambda: self._check_jvp_complex())

        # strict=True：输出与输入不连通应报错
        _run_case("jvp - strict_independent_raises", lambda: self._check_jvp_strict())

    def _check_jvp(self, f, p_np, t_np, exp_out, exp_jvp):
        p = _rm_tensor(p_np)
        t = _rm_tensor(t_np, requires_grad=False)
        out, jvp_res = func.jvp(f, p, t)
        return _close(out, exp_out) and _close(jvp_res, exp_jvp)

    def _check_jvp_fd(self):
        f = lambda x: (x ** 2 + 3 * x).sum()
        x_np = np.array([1.0, 2.0, 3.0])
        v_np = np.array([0.5, 1.0, 0.0])
        x = _rm_tensor(x_np)
        v = _rm_tensor(v_np, requires_grad=False)
        _, jvp_res = func.jvp(f, x, v)
        eps = 1e-5
        fd = ((x_np + eps * v_np) ** 2 + 3 * (x_np + eps * v_np)).sum() - \
             ((x_np - eps * v_np) ** 2 + 3 * (x_np - eps * v_np)).sum()
        fd /= (2 * eps)
        return _close(jvp_res, fd, atol=1e-4)

    def _check_jvp_multi_input(self):
        f = lambda x, y: (x * y).sum()
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        vx = _rm_tensor(np.array([1.0, 0.0]), requires_grad=False)
        vy = _rm_tensor(np.array([0.0, 1.0]), requires_grad=False)
        out, jvp_res = func.jvp(f, (x, y), (vx, vy))
        # out = 1*3 + 2*4 = 11
        # jvp = y·vx + x·vy = 3 + 2 = 5
        return _close(out, 11.0) and _close(jvp_res, 5.0)

    def _check_jvp_create_graph(self):
        f = lambda x: (x ** 2).sum()
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        v = _rm_tensor(np.array([0.5, 1.0, 0.5]), requires_grad=False)
        _, jvp_res = func.jvp(f, x, v, create_graph=True)
        # jvp = 2x·v = 1 + 4 + 3 = 8
        return _close(jvp_res, 8.0)

    def _check_jvp_default_tangent(self):
        f = lambda x: (x ** 3).sum()
        out, jvp_res = func.jvp(f, _rm_tensor(np.array([1.0, 2.0])), None)
        # out = 1 + 8 = 9；jvp = 3*(1 + 4) = 15（方向取标量 1.0）
        return _close(out, 9.0) and _close(jvp_res, 15.0)

    def _check_jvp_multi_output(self):
        f = lambda x: (x.sum(), (x ** 2).sum())
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        v = _rm_tensor(np.array([1.0, 0.0, 1.0]), requires_grad=False)
        out, jvp_res = func.jvp(f, x, v)
        # out = (6, 14)；jvp = (sum(v)=2, 2x·v=8)
        return (isinstance(out, tuple) and isinstance(jvp_res, tuple)
                and len(out) == 2 and len(jvp_res) == 2
                and _close(out[0], 6.0) and _close(out[1], 14.0)
                and _close(jvp_res[0], 2.0) and _close(jvp_res[1], 8.0))

    def _check_jvp_complex(self):
        z = _rm_ctensor(np.array([1.0 + 2.0j, 3.0 - 1.0j]))
        v = _rm_ctensor(np.array([1.0 + 0.0j, 1.0 + 0.0j]), requires_grad=False)
        out, jvp_res = func.jvp(lambda t: (t ** 2).sum(), z, v)
        # sum(z^2) = 5 - 2j；J·v = 2z = 8 + 2j（两次共轭抵消）
        return _close(out, 5.0 - 2.0j) and _close(jvp_res, 8.0 + 2.0j)

    def _check_jvp_strict(self):
        x = _rm_tensor(np.array([1.0, 2.0]))
        v = _rm_tensor(np.array([1.0, 1.0]), requires_grad=False)
        try:
            func.jvp(lambda _: tensor(3.0), x, v, strict=True)
            return False
        except RuntimeError:
            return True

    # ---------------------------------------------------------------
    # vjp
    # ---------------------------------------------------------------
    def test_vjp(self):
        cases = [
            ("scalar_default_v",          lambda x: (x ** 2).sum() + x.sum(),  np.array([1.0, 2.0, 3.0]), None,             20.0, np.array([3.0, 5.0, 7.0])),
            ("scalar_custom_v",           lambda x: (x ** 2).sum() + x.sum(),  np.array([1.0, 2.0, 3.0]), _rm_tensor(np.array(2.0), requires_grad=False), 20.0, np.array([6.0, 10.0, 14.0])),
            ("vector_default_cotangent",  lambda x: x ** 2,                   np.array([1.0, 2.0, 3.0]), None,             np.array([1.0, 4.0, 9.0]), np.array([2.0, 4.0, 6.0])),
        ]
        for desc, f, p_np, v, exp_out, exp_vjp in cases:
            _run_case(f"vjp - {desc}",
                      lambda ff=f, pp=p_np, vv=v, eo=exp_out, ej=exp_vjp:
                          self._check_vjp(ff, pp, vv, eo, ej))

        _run_case("vjp - matches_grad", lambda: self._check_vjp_grad())

        # 向量输出 + 自定义 cotangents
        _run_case("vjp - vector_output_custom_cotangents",
                  lambda: self._check_vjp_vector_custom())

        # 多输入：返回与输入一一对应的 tuple
        _run_case("vjp - multi_input", lambda: self._check_vjp_multi_input())

        # 多输出 + 多 cotangent：各输出贡献累加
        _run_case("vjp - multi_output_cotangents", lambda: self._check_vjp_multi_output())

        # cotangents 数量与输出不匹配
        _run_case("vjp - cotangent_count_mismatch", lambda: self._check_vjp_cotangent_mismatch())

        # 常量输出：strict=False 返回零块，strict=True 报错
        _run_case("vjp - constant_output_zeros", lambda: self._check_vjp_constant())

    def _check_vjp(self, f, p_np, v, exp_out, exp_vjp):
        p = _rm_tensor(p_np)
        out, vjp_res = func.vjp(f, p, v)
        return _close(out, exp_out) and _close(vjp_res, exp_vjp)

    def _check_vjp_grad(self):
        f = lambda x: (x ** 2 + 3 * x).sum()
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        _, vjp_res = func.vjp(f, x)
        grad_direct = func.grad(f)(x)
        return _close(vjp_res, grad_direct)

    def _check_vjp_vector_custom(self):
        f = lambda x: x ** 2  # 向量输出
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        cot = _rm_tensor(np.array([0.5, 1.0, 2.0]), requires_grad=False)
        out, vjp_res = func.vjp(f, x, cot)
        # out = [1,4,9]; vjp = 2*x * cot = [1, 4, 12]
        return _close(out, np.array([1.0, 4.0, 9.0])) and _close(vjp_res, np.array([1.0, 4.0, 12.0]))

    def _check_vjp_multi_input(self):
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        out, res = func.vjp(lambda a, b: (a * b).sum(), (x, y))
        # out = 11；∂f/∂x = y，∂f/∂y = x
        return (_close(out, 11.0) and isinstance(res, tuple) and len(res) == 2
                and _close(res[0], np.array([3.0, 4.0]))
                and _close(res[1], np.array([1.0, 2.0])))

    def _check_vjp_multi_output(self):
        f = lambda x: (x.sum(), (x ** 2).sum())
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        out, res = func.vjp(f, x, (_rm_tensor(2.0, requires_grad=False),
                                   _rm_tensor(3.0, requires_grad=False)))
        # grad = 2*ones + 3*2x = [8, 14, 20]
        return (isinstance(out, tuple) and _close(out[0], 6.0) and _close(out[1], 14.0)
                and _close(res, np.array([8.0, 14.0, 20.0])))

    def _check_vjp_cotangent_mismatch(self):
        f = lambda x: (x.sum(), (x ** 2).sum())
        x = _rm_tensor(np.array([1.0, 2.0]))
        try:
            func.vjp(f, x, tensor(1.0))   # 单余切 vs 双输出
            return False
        except RuntimeError:
            return True

    def _check_vjp_constant(self):
        x = _rm_tensor(np.array([1.0, 2.0]))
        out, res = func.vjp(lambda _: tensor(3.0), x)
        zeros_ok = _close(out, 3.0) and _close(res, np.zeros(2))
        try:
            func.vjp(lambda _: tensor(3.0), x, strict=True)
            return False
        except RuntimeError:
            return zeros_ok

    # ---------------------------------------------------------------
    # jacrev
    # ---------------------------------------------------------------
    def test_jacrev(self):
        cases = [
            ("diagonal",          lambda x: x ** 2,                                    np.array([1.0, 2.0, 3.0]),         0,        np.diag([2.0, 4.0, 6.0])),
            ("linear_transform",  lambda x: x @ np.array([[1.0, 2.0], [3.0, 4.0]]),  np.array([1.0, 2.0]),              0,        np.array([[1.0, 3.0], [2.0, 4.0]])),
            ("sum_to_scalar",     lambda x: x.sum(),                                   np.array([1.0, 2.0, 3.0]),         0,        np.array([1.0, 1.0, 1.0])),
            ("cubic_diagonal",    lambda x: x ** 3,                                    np.array([1.0, 2.0, 3.0]),         0,        np.diag([3.0, 12.0, 27.0])),
        ]
        for desc, f, x_np, argnums, expected in cases:
            _run_case(f"jacrev - {desc}",
                      lambda ff=f, xx=x_np, aa=argnums, ee=expected:
                          self._check_jacrev(ff, xx, aa, ee))

        _run_case("jacrev - argnums_filter", lambda: self._check_jacrev_argnums())

        # 2D 输入 -> 标量输出，Jacobian 形状应该是扁平化 input_shape
        _run_case("jacrev - 2d_input_shape", lambda: self._check_jacrev_2d_shape())

        # 多输出：每个输出一个 Jacobian（tuple）
        _run_case("jacrev - multi_output", lambda: self._check_jacrev_multi_output())

        # 多输出 + 多输入：外层按输出、内层按 argnums
        _run_case("jacrev - multi_output_multi_input",
                  lambda: self._check_jacrev_multi_output_multi_input())

        # 多输出 + 多输入 + 非连续 argnums 子集
        _run_case("jacrev - multi_output_argnums_filter",
                  lambda: self._check_jacrev_multi_output_argnums())

        # 独立输入：strict=False 返回零块，strict=True 报错
        _run_case("jacrev - independent_input", lambda: self._check_jacrev_independent())

        # 0 维输入：Jacobian 为 0 维
        _run_case("jacrev - scalar_input", lambda: self._check_jacrev_scalar_input())

    def _check_jacrev(self, f, x_np, argnums, expected):
        x = _rm_tensor(x_np)
        jac = func.jacrev(f, argnums=argnums)(x)
        return _close(jac, expected)

    def _check_jacrev_argnums(self):
        def f(x, y, z): return x * y + z
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        z = _rm_tensor(np.array([5.0, 6.0]))
        jac = func.jacrev(f, argnums=(0, 2))(x, y, z)
        return isinstance(jac, tuple) and len(jac) == 2

    def _check_jacrev_2d_shape(self):
        # (2,2) 输入 -> 标量输出，Jacobian 形状应该是 (4,)  —— 扁平化后的形状
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        f = lambda x: (x ** 2).sum()
        x = _rm_tensor(x_np)
        jac = func.jacrev(f)(x)
        expected_flat = np.array([2.0, 4.0, 6.0, 8.0])
        # jac 可能是 (4,) 也可能是 (2,2)，取决于底层 flatten 策略
        jac_arr = jac._array if hasattr(jac, '_array') else np.asarray(jac)
        return jac_arr.flatten().shape == expected_flat.shape and np.allclose(jac_arr.flatten(), expected_flat)

    def _check_jacrev_multi_output(self):
        f = lambda x: (x.sum(), (x ** 2).sum())
        jac = func.jacrev(f)(_rm_tensor(np.array([1.0, 2.0, 3.0])))
        # 两个输出各自的 Jacobian：ones 与 2x
        return (isinstance(jac, tuple) and len(jac) == 2
                and _close(jac[0], np.ones(3))
                and _close(jac[1], np.array([2.0, 4.0, 6.0])))

    def _check_jacrev_multi_output_multi_input(self):
        # 结构约定与 torch.func.jacrev / jacfwd 一致：外层按输出、内层按 argnums
        f = lambda x, y: ((x * y).sum(), (x * y ** 2).sum())
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        jac = func.jacrev(f, argnums=(0, 1))(x, y)
        return (isinstance(jac, tuple) and len(jac) == 2
                and all(isinstance(blk, tuple) and len(blk) == 2 for blk in jac)
                # out0 = sum(x*y)：d/dx = y，d/dy = x
                and _close(jac[0][0], np.array([3.0, 4.0]))
                and _close(jac[0][1], np.array([1.0, 2.0]))
                # out1 = sum(x*y^2)：d/dx = y^2，d/dy = 2*x*y
                and _close(jac[1][0], np.array([9.0, 16.0]))
                and _close(jac[1][1], np.array([6.0, 16.0])))

    def _check_jacrev_multi_output_argnums(self):
        # 每个输出都依赖 w，但 w 不在 argnums 中，用于验证内层按 argnums 过滤
        f = lambda x, w, y: ((x * y).sum() + w.sum(), (x * y ** 2).sum() + w.sum())
        x = _rm_tensor(np.array([1.0, 2.0]))
        w = _rm_tensor(np.array([1.0, 1.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        jac = func.jacrev(f, argnums=(0, 2))(x, w, y)
        return (isinstance(jac, tuple) and len(jac) == 2
                and all(isinstance(blk, tuple) and len(blk) == 2 for blk in jac)
                and _close(jac[0][0], np.array([3.0, 4.0]))
                and _close(jac[0][1], np.array([1.0, 2.0]))
                and _close(jac[1][0], np.array([9.0, 16.0]))
                and _close(jac[1][1], np.array([6.0, 16.0])))

    def _check_jacrev_independent(self):
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        zeros_ok = _close(
            func.jacrev(lambda a, b: (b ** 2).sum(), argnums=0, strict=False)(x, y),
            np.zeros(2))
        try:
            func.jacrev(lambda a, b: (b ** 2).sum(), argnums=0, strict=True)(x, y)
            return False
        except RuntimeError:
            return zeros_ok

    def _check_jacrev_scalar_input(self):
        x = _rm_tensor(2.0)
        jac = func.jacrev(lambda t: (t ** 3).sum())(x)
        # d(x^3)/dx = 3x^2 = 12，0 维输入对应 0 维 Jacobian
        return _close(jac, 12.0)

    # ---------------------------------------------------------------
    # jacfwd
    # ---------------------------------------------------------------
    def test_jacfwd(self):
        cases = [
            ("diagonal",  lambda x: x ** 2,                                   np.array([1.0, 2.0, 3.0]),  np.diag([2.0, 4.0, 6.0])),
            ("cubic",     lambda x: x ** 3,                                   np.array([1.0, 2.0, 3.0]),  np.diag([3.0, 12.0, 27.0])),
            ("linear",    lambda x: x @ np.array([[1.0, 2.0], [3.0, 4.0]]),  np.array([1.0, 2.0]),       np.array([[1.0, 3.0], [2.0, 4.0]])),
        ]
        for desc, f, x_np, expected in cases:
            _run_case(f"jacfwd - {desc}",
                      lambda ff=f, xx=x_np, ee=expected: self._check_jacfwd(ff, xx, ee))

        _run_case("jacfwd - equals_jacrev", lambda: self._check_jacfwd_eq_rev())

        # 多输入场景
        _run_case("jacfwd - multi_input", lambda: self._check_jacfwd_multi_input())

        # 多输出：每个输出一个 Jacobian（tuple）
        _run_case("jacfwd - multi_output", lambda: self._check_jacfwd_multi_output())

        # 多输出 + 多输入：与 jacrev 相同的嵌套约定（外层输出、内层 argnums）
        _run_case("jacfwd - multi_output_multi_input",
                  lambda: self._check_jacfwd_multi_output_multi_input())

        # 复数前向：J·v 用 analytic Jacobian（.mH 两次共轭抵消）
        _run_case("jacfwd - complex", lambda: self._check_jacfwd_complex())

        # 0 维输入：Jacobian 为 0 维
        _run_case("jacfwd - scalar_input", lambda: self._check_jacfwd_scalar_input())

    def _check_jacfwd(self, f, x_np, expected):
        x = _rm_tensor(x_np)
        jac = func.jacfwd(f)(x)
        return _close(jac, expected)

    def _check_jacfwd_eq_rev(self):
        f = lambda x: x ** 2 + 0.5 * x
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        return _close(func.jacrev(f)(x), func.jacfwd(f)(x))

    def _check_jacfwd_multi_input(self):
        f = lambda x, y: (x * y).sum()  # 标量输出，对两个输入各算 Jacobian
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        j = func.jacfwd(f, argnums=(0, 1))(x, y)
        # 标量输出的 Jacobian 形状就是各输入形状
        # ∂f/∂x = y, ∂f/∂y = x
        return (isinstance(j, tuple) and len(j) == 2
                and _close(j[0], np.array([3.0, 4.0]))
                and _close(j[1], np.array([1.0, 2.0])))

    def _check_jacfwd_multi_output(self):
        f = lambda x: (x.sum(), (x ** 2).sum())
        jac = func.jacfwd(f)(_rm_tensor(np.array([1.0, 2.0, 3.0])))
        return (isinstance(jac, tuple) and len(jac) == 2
                and _close(jac[0], np.ones(3))
                and _close(jac[1], np.array([2.0, 4.0, 6.0])))

    def _check_jacfwd_multi_output_multi_input(self):
        f = lambda x, y: ((x * y).sum(), (x * y ** 2).sum())
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        jac = func.jacfwd(f, argnums=(0, 1))(x, y)
        return (isinstance(jac, tuple) and len(jac) == 2
                and all(isinstance(blk, tuple) and len(blk) == 2 for blk in jac)
                and _close(jac[0][0], np.array([3.0, 4.0]))
                and _close(jac[0][1], np.array([1.0, 2.0]))
                and _close(jac[1][0], np.array([9.0, 16.0]))
                and _close(jac[1][1], np.array([6.0, 16.0])))

    def _check_jacfwd_complex(self):
        z = _rm_ctensor(np.array([1.0 + 2.0j, 3.0 - 1.0j]))
        jac = func.jacfwd(lambda t: (t ** 2).sum())(z)
        # 解析 Jacobian：2z = [2+4j, 6-2j]
        return _close(jac, np.array([2.0 + 4.0j, 6.0 - 2.0j]))

    def _check_jacfwd_scalar_input(self):
        x = _rm_tensor(2.0)
        jac = func.jacfwd(lambda t: (t ** 3).sum())(x)
        return _close(jac, 12.0)

    # ---------------------------------------------------------------
    # hessian
    # ---------------------------------------------------------------
    def test_hessian(self):
        cases = [
            ("cubic_sum",   lambda x: (x ** 3).sum(),                                 np.array([1.0, 2.0, 3.0]),  np.diag([6.0, 12.0, 18.0])),
            ("quadratic",   lambda x: x @ np.array([[1.0, 0.5], [0.5, 2.0]]) @ x,     np.array([1.0, 2.0]),       np.array([[2.0, 1.0], [1.0, 4.0]])),
            ("quartic_sum", lambda x: (x ** 4).sum(),                                 np.array([1.0, 2.0]),       np.diag([12.0, 48.0])),
        ]
        for desc, f, x_np, expected in cases:
            _run_case(f"hessian - {desc}",
                      lambda ff=f, xx=x_np, ee=expected: self._check_hessian(ff, xx, ee))

        _run_case("hessian - argnums_filter", lambda: self._check_hessian_argnums())

        # create_graph=True
        _run_case("hessian - create_graph", lambda: self._check_hessian_create_graph())

        # 非标量函数应抛异常
        _run_case("hessian - non_scalar_raises", lambda: self._check_hessian_non_scalar())

        # 多 argnums：返回各输入的 Hessian 块
        _run_case("hessian - multi_argnums", lambda: self._check_hessian_multi_argnums())

        # 非连续 argnums (0, 2)
        _run_case("hessian - noncontiguous_argnums",
                  lambda: self._check_hessian_noncontiguous())

        # argnums 子集 + 与其余参数无关：strict 自动放宽
        _run_case("hessian - strict_relaxed_subset",
                  lambda: self._check_hessian_strict_relaxed())

    def _check_hessian(self, f, x_np, expected):
        x = _rm_tensor(x_np)
        hess = func.hessian(f)(x)
        return _close(hess, expected)

    def _check_hessian_argnums(self):
        # 不可分离函数：f = x * y²，∂²f/∂x² = 0，∂²f/∂x∂y = 2y
        # cross-Hessian 非零 → strict 检查通过
        def f(x, y): return (x * y ** 2).sum()
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        h = func.hessian(f, argnums=(0,))(x, y)
        # argnums=(0,) 只应返回 1 个 Hessian（对 x 的），不应是 tuple-of-tuples
        return not isinstance(h, tuple)

    def _check_hessian_create_graph(self):
        f = lambda x: (x ** 4).sum()
        h = func.hessian(f, create_graph=True)(_rm_tensor(np.array([1.0, 2.0, 3.0])))
        # Hessian of x^4: diag(12x^2) = diag([12, 48, 108])
        return _close(h, np.diag([12.0, 48.0, 108.0]))

    def _check_hessian_non_scalar(self):
        f = lambda x: x ** 2  # 向量输出，不是标量
        try:
            func.hessian(f)(_rm_tensor(np.array([1.0, 2.0, 3.0])))
            return False
        except (RuntimeError, ValueError):
            return True

    def _check_hessian_multi_argnums(self):
        f = lambda x, y: (x ** 2 * y ** 2).sum()
        h = func.hessian(f, argnums=(0, 1))(
            _rm_tensor(np.array([1.0, 2.0])), _rm_tensor(np.array([3.0, 4.0])))
        # H_xx = diag(2y²) = diag([18, 32])；H_yy = diag(2x²) = diag([2, 8])
        return (isinstance(h, tuple) and len(h) == 2
                and _close(h[0], np.diag([18.0, 32.0]))
                and _close(h[1], np.diag([2.0, 8.0])))

    def _check_hessian_noncontiguous(self):
        f = lambda x, y, w: (x ** 2 * w).sum() + (w ** 3).sum()
        h = func.hessian(f, argnums=(0, 2))(
            _rm_tensor(np.array([1.0, 2.0])),     # x
            _rm_tensor(np.array([3.0])),          # y（不参与求导）
            _rm_tensor(np.array([4.0, 5.0])))     # w
        # H_xx = diag(2w) = diag([8, 10])；H_ww = diag(6w) = diag([24, 30])
        return (isinstance(h, tuple) and len(h) == 2
                and _close(h[0], np.diag([8.0, 10.0]))
                and _close(h[1], np.diag([24.0, 30.0])))

    def _check_hessian_strict_relaxed(self):
        # 梯度 ∂f/∂x = 3x² 与 y 无关；argnums=(0,) 时 strict 应自动放宽
        h = func.hessian(lambda x, y: (x ** 3).sum() + (y ** 3).sum(), argnums=(0,))(
            _rm_tensor(np.array([1.0, 2.0])), _rm_tensor(np.array([3.0, 4.0])))
        return not isinstance(h, tuple) and _close(h, np.diag([6.0, 12.0]))

    # ---------------------------------------------------------------
    # vmap
    # ---------------------------------------------------------------
    def test_vmap(self):
        cases = [
            ("scalar_output_batch",  lambda x: (x ** 3).sum(),                     np.array([[1.0, 2.0], [3.0, 4.0]]),  0,    np.array([9.0, 91.0])),
            ("vector_output_batch",  lambda x: x * 2 + 1,                          np.array([[1.0, 2.0], [3.0, 4.0]]),  0,    np.array([[3.0, 5.0], [7.0, 9.0]])),
            ("single_dim_batch",     lambda x: x ** 2,                             np.array([1.0, 2.0, 3.0]),           0,    np.array([1.0, 4.0, 9.0])),
            ("constant_output",      lambda x: tensor(5.0, requires_grad=False),  np.array([[1.0], [2.0]]),            0,    np.array([5.0, 5.0])),
        ]
        for desc, f, inp_np, in_dims, expected in cases:
            _run_case(f"vmap - {desc}",
                      lambda ff=f, ii=inp_np, dd=in_dims, ee=expected:
                          self._check_vmap(ff, ii, dd, ee))

        _run_case("vmap - no_batch_passthrough", lambda: self._check_vmap_passthrough())

        # func 返回 tuple（多输出）
        _run_case("vmap - func_returns_tuple", lambda: self._check_vmap_tuple_out())

        # out_dims 不同于 0
        _run_case("vmap - out_dims_nonzero", lambda: self._check_vmap_out_dims())

        # 多输入 + in_dims 元组
        _run_case("vmap - multi_input_in_dims_tuple", lambda: self._check_vmap_multi_tuple())

        # 多输入 + in_dims=None 某输入不批处理（广播语义）
        _run_case("vmap - multi_input_with_none_dim", lambda: self._check_vmap_multi_none())

        # 多输入 + in_dims=0 共享维度（标量输出）
        _run_case("vmap - multi_input_shared_dim", lambda: self._check_vmap_multi_shared())

        # no_grad 下的纯前向路径（输出缓冲预分配）
        _run_case("vmap - no_grad_pure_forward", lambda: self._check_vmap_no_grad())

        # 梯度可穿过输出堆叠（有梯度时回退 stack）
        _run_case("vmap - gradient_flows", lambda: self._check_vmap_grad_flow())

        # 负 in_dims
        _run_case("vmap - negative_in_dims", lambda: self._check_vmap_negative_dim())

        # 多输出 + out_dims 元组
        _run_case("vmap - out_dims_tuple", lambda: self._check_vmap_out_dims_tuple())

        # 非张量参数透传
        _run_case("vmap - non_tensor_arg", lambda: self._check_vmap_non_tensor_arg())

    def _check_vmap(self, f, inp_np, in_dims, expected):
        inp = _rm_tensor(inp_np)
        out = func.vmap(f, in_dims=in_dims)(inp)
        return _close(out, expected)

    def _check_vmap_passthrough(self):
        f = lambda x: (x ** 2).sum()
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        out = func.vmap(f, in_dims=None)(x)
        return _close(out, 14.0)

    def _check_vmap_tuple_out(self):
        f = lambda x: ((x ** 2).sum(), (x ** 3).sum())
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        out = func.vmap(f)(x)
        return (isinstance(out, tuple) and len(out) == 2
                and _close(out[0], np.array([5.0, 25.0]))
                and _close(out[1], np.array([9.0, 91.0])))

    def _check_vmap_out_dims(self):
        f = lambda x: x * 2
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        out = func.vmap(f, out_dims=1)(x)
        # 每个样本输出形状 (2,)，沿 dim=1 堆叠后整体形状应还是 (2, 2)
        return out.shape == (2, 2)

    def _check_vmap_multi_tuple(self):
        f = lambda x, y: x + y
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        y = _rm_tensor(np.array([[10.0, 20.0], [30.0, 40.0]]))
        out = func.vmap(f, in_dims=(0, 0))(x, y)
        return _close(out, np.array([[11.0, 22.0], [33.0, 44.0]]))

    def _check_vmap_multi_none(self):
        f = lambda x, y: x + y
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))   # batch=0
        y = _rm_tensor(np.array([10.0, 20.0]))                # 不批处理，直接透传
        out = func.vmap(f, in_dims=(0, None))(x, y)
        # 每个样本加上同一个 y
        return _close(out, np.array([[11.0, 22.0], [13.0, 24.0]]))

    def _check_vmap_multi_shared(self):
        f = lambda x, y: (x * y).sum()
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        y = _rm_tensor(np.array([[0.5, 1.0], [2.0, 3.0]]))
        # in_dims=0 是 int，所有输入共享 dim=0
        out = func.vmap(f, in_dims=0)(x, y)
        # 样本 1: 1*0.5+2*1 = 2.5; 样本 2: 3*2+4*3 = 18
        return _close(out, np.array([2.5, 18.0]))

    def _check_vmap_no_grad(self):
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with rm.no_grad():
            out = func.vmap(lambda row: (row ** 2).sum())(x)
        return _close(out, np.array([5.0, 25.0]))

    def _check_vmap_grad_flow(self):
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        g = func.grad(lambda a: func.vmap(lambda row: (row ** 3).sum())(a).sum())(x)
        # d(sum(a^3))/da = 3a²
        return _close(g, 3.0 * np.array([[1.0, 4.0], [9.0, 16.0]]))

    def _check_vmap_negative_dim(self):
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
        out = func.vmap(lambda col: (col ** 2).sum(), in_dims=-1)(x)
        # 沿最后一维批处理：列 [1,3]->10，列 [2,4]->20
        return _close(out, np.array([10.0, 20.0]))

    def _check_vmap_out_dims_tuple(self):
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        out = func.vmap(lambda row: (row.sum(), row * 2), out_dims=(0, 1))(x)
        return (isinstance(out, tuple) and len(out) == 2
                and _close(out[0], np.array([3.0, 7.0]))
                and _close(out[1], np.array([[2.0, 6.0], [4.0, 8.0]])))

    def _check_vmap_non_tensor_arg(self):
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        out = func.vmap(lambda row, c: c * (row ** 2).sum())(x, 2.0)
        return _close(out, np.array([10.0, 50.0]))

    # ---------------------------------------------------------------
    # functional_call
    # ---------------------------------------------------------------
    def test_functional_call(self):
        _run_case("functional_call - default_params",   lambda: self._check_fc_default())
        _run_case("functional_call - custom_params_and_restore", lambda: self._check_fc_custom())
        _run_case("functional_call - no_params_module", lambda: self._check_fc_noparams())
        _run_case("functional_call - wrong_param_count", lambda: self._check_fc_count_error())
        _run_case("functional_call - param_type_error",  lambda: self._check_fc_type_error())

        # kwargs 透传
        _run_case("functional_call - kwargs_passthrough", lambda: self._check_fc_kwargs())

        # 多参数 named_parameters
        _run_case("functional_call - multi_params", lambda: self._check_fc_multi_params())

        # 单个 TN 作为 parameter_values（免包 list）
        _run_case("functional_call - single TN parameter_values",
                  lambda: self._check_fc_single_tn())

        # 前向抛异常时参数必须恢复（finally 分支）
        _run_case("functional_call - restore_on_forward_error",
                  lambda: self._check_fc_restore_on_error())

        # 仅提供 _named_parameters 的旧式模块（setattr 回退分支）
        _run_case("functional_call - _named_parameters fallback",
                  lambda: self._check_fc_named_parameters_fallback())

        # ---- rm.nn.Module 真实 Module 场景（今天修复的 _parameters 分支） ----
        _run_case("functional_call - rm.nn.Linear default",
                  lambda: self._check_fc_rmnn_default())
        _run_case("functional_call - rm.nn.Linear custom_and_restore",
                  lambda: self._check_fc_rmnn_custom())
        _run_case("functional_call - rm.nn.Linear single TN args (not tuple)",
                  lambda: self._check_fc_rmnn_single_args())

        # ---- 与 func.grad 组合：对替代参数求梯度 ----
        _run_case("functional_call + grad - differentiate w.r.t. substitute params",
                  lambda: self._check_fc_grad_combo())

    def _make_linear_module(self):
        class Linear:
            def __init__(self, w): self.w = w
            def forward(self, x): return x @ self.w
            def named_parameters(self): yield "w", self.w
        return Linear

    def _check_fc_default(self):
        Linear = self._make_linear_module()
        w = _rm_tensor(np.random.randn(4, 5))
        x = _rm_tensor(np.random.randn(2, 4), requires_grad=False)
        model = Linear(w)
        y_default = func.functional_call(model, None, (x,))
        y_direct = model.forward(x)
        return _close(y_default, y_direct)

    def _check_fc_custom(self):
        Linear = self._make_linear_module()
        w0 = _rm_tensor(np.random.randn(4, 5))
        w_new = _rm_tensor(np.random.randn(4, 5))
        x = _rm_tensor(np.random.randn(2, 4), requires_grad=False)
        model = Linear(w0)
        y_custom = func.functional_call(model, [w_new], (x,))
        return _close(y_custom, x @ w_new) and _close(model.w, w0)  # 属性恢复

    def _check_fc_noparams(self):
        class Const:
            def forward(self, x): return x * 2.0
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]), requires_grad=False)
        y = func.functional_call(Const(), None, (x,))
        return _close(y, x * 2.0)

    def _check_fc_count_error(self):
        Linear = self._make_linear_module()
        w = _rm_tensor(np.random.randn(4, 5))
        model = Linear(w)
        try:
            func.functional_call(model, [w, w], (_rm_tensor(np.random.randn(2, 4), requires_grad=False),))
            return False
        except ValueError:
            return True

    def _check_fc_type_error(self):
        Linear = self._make_linear_module()
        w = _rm_tensor(np.random.randn(4, 5))
        model = Linear(w)
        try:
            func.functional_call(model, "not_a_tensor", ())
            return False
        except TypeError:
            return True

    def _check_fc_kwargs(self):
        class LinearScale:
            def __init__(self, w): self.w = w
            def forward(self, x, scale=1.0): return (x @ self.w) * scale
            def named_parameters(self): yield "w", self.w
        w = _rm_tensor(np.random.randn(4, 5))
        x = _rm_tensor(np.random.randn(2, 4), requires_grad=False)
        model = LinearScale(w)
        y_fc = func.functional_call(model, None, (x,), {"scale": 2.0})
        y_direct = model.forward(x, scale=2.0)
        return _close(y_fc, y_direct)

    def _check_fc_multi_params(self):
        class LinearBias:
            def __init__(self, w, b): self.w = w; self.b = b
            def forward(self, x): return x @ self.w + self.b
            def named_parameters(self):
                yield "w", self.w
                yield "b", self.b
        w0 = _rm_tensor(np.random.randn(4, 5))
        b0 = _rm_tensor(np.random.randn(5))
        w_new = _rm_tensor(np.random.randn(4, 5))
        b_new = _rm_tensor(np.random.randn(5))
        x = _rm_tensor(np.random.randn(2, 4), requires_grad=False)
        model = LinearBias(w0, b0)
        y_custom = func.functional_call(model, [w_new, b_new], (x,))
        # 结果应该等于 x @ w_new + b_new
        y_expected = x @ w_new + b_new
        # 同时验证 model 属性没有被修改
        return (_close(y_custom, y_expected)
                and _close(model.w, w0)
                and _close(model.b, b0))

    def _check_fc_single_tn(self):
        Linear = self._make_linear_module()
        w0 = _rm_tensor(np.random.randn(4, 5))
        w_new = _rm_tensor(np.random.randn(4, 5))
        x = _rm_tensor(np.random.randn(2, 4), requires_grad=False)
        model = Linear(w0)
        y = func.functional_call(model, w_new, (x,))
        return _close(y, x @ w_new) and _close(model.w, w0)

    def _check_fc_restore_on_error(self):
        class Boom:
            def __init__(self, w): self.w = w
            def forward(self, x): raise RuntimeError("boom")
            def named_parameters(self): yield "w", self.w
        w0 = _rm_tensor(np.array([1.0]))
        model = Boom(w0)
        try:
            func.functional_call(model, [_rm_tensor(np.array([2.0]))],
                                 (_rm_tensor(np.array([1.0]), requires_grad=False),))
            return False
        except RuntimeError:
            return _close(model.w, w0)   # finally 中已恢复

    def _check_fc_named_parameters_fallback(self):
        class Legacy:
            def __init__(self, w): self.w = w
            def forward(self, x): return x * self.w
            def _named_parameters(self): yield "w", self.w
        w0 = _rm_tensor(np.array([1.0, 2.0]))
        w_new = _rm_tensor(np.array([3.0, 4.0]))
        x = _rm_tensor(np.array([1.0, 1.0]), requires_grad=False)
        model = Legacy(w0)
        y = func.functional_call(model, [w_new], (x,))
        return _close(y, np.array([3.0, 4.0])) and _close(model.w, w0)

    # ---- rm.nn.Module 真实 Module 场景 ----
    def _check_fc_rmnn_default(self):
        """rm.nn.Linear 使用自身参数调用（parameter_values=None）"""
        layer = rm.nn.Linear(4, 5)
        x = _rm_tensor(np.random.randn(2, 4), requires_grad=False)
        # functional_call 不替换参数
        y_fc = func.functional_call(layer, None, (x,))
        y_direct = layer(x)
        return _close(y_fc, y_direct)

    def _check_fc_rmnn_custom(self):
        """rm.nn.Linear 用替代参数调用，验证 _parameters 分支替换 + 恢复"""
        layer = rm.nn.Linear(4, 5)
        orig_weight = layer.weight.clone()
        orig_bias = layer.bias.clone()

        w_new = _rm_tensor(np.random.randn(5, 4), requires_grad=True)
        b_new = _rm_tensor(np.random.randn(5), requires_grad=True)
        x = _rm_tensor(np.random.randn(2, 4), requires_grad=False)

        y_fc = func.functional_call(layer, [w_new, b_new], (x,))
        # 手动计算期望值：x @ w_new.T + b_new
        y_expected = x @ w_new.T + b_new
        # Module 自身参数必须恢复原值
        restored = (_close(layer.weight, orig_weight)
                    and _close(layer.bias, orig_bias))
        return _close(y_fc, y_expected) and restored

    def _check_fc_rmnn_single_args(self):
        """args 直接传单个 TN（非 tuple），验证不会被错误解包"""
        layer = rm.nn.Linear(4, 5)
        x = _rm_tensor(np.random.randn(2, 4), requires_grad=False)
        # 单 TN 作为 args（不是 (x,)）
        y_fc = func.functional_call(layer, None, x)
        y_direct = layer(x)
        return _close(y_fc, y_direct)

    # ---- functional_call + func.grad 组合 ----
    def _check_fc_grad_combo(self):
        """functional_call 与 func.grad_and_value 组合，对替代参数求梯度"""
        layer = rm.nn.Linear(4, 5)
        x = _rm_tensor(np.random.randn(2, 4), requires_grad=False)

        alt_w = _rm_tensor(np.random.randn(5, 4), requires_grad=True)
        alt_b = _rm_tensor(np.random.randn(5), requires_grad=True)

        # argnums=(0, 1) 同时对 w 和 b 求梯度
        loss_fn = func.grad_and_value(
            lambda w, b: (func.functional_call(layer, [w, b], x) ** 2).sum(),
            argnums=(0, 1)
        )
        g, v = loss_fn(alt_w, alt_b)   # g = (g_w, g_b)

        # 手动数值验证：
        # L = sum((x @ alt_w.T + alt_b)^2)
        # dL/d(alt_w) = 2 * (x @ alt_w.T + alt_b).T @ x
        # dL/d(alt_b) = 2 * sum(x @ alt_w.T + alt_b, dim=0)
        y = x @ alt_w.T + alt_b   # (2, 5)
        expected_w = (2.0 * y.T @ x)  # (5, 4)
        expected_b = (2.0 * y).sum(dim=0)  # (5,)
        return (_close(g[0], expected_w)
                and _close(g[1], expected_b)
                and _close(v, (y ** 2).sum()))

    # ---------------------------------------------------------------
    # 函数组合（跨原语组合能力）
    # ---------------------------------------------------------------
    def test_compositions(self):
        _run_case("compositions - grad_of_grad", lambda: self._check_comp_grad_grad())
        _run_case("compositions - hessian_via_jacrev_of_grad", lambda: self._check_comp_hess_chain())
        _run_case("compositions - jvp_via_finite_diff", lambda: self._check_comp_jvp_fd())

        # jvp + vjp 互逆关系
        _run_case("compositions - jvp_vjp_consistency", lambda: self._check_comp_jvp_vjp())

        # vmap + jacrev 组合（底层 requires_grad_ 修复后可用）
        _run_case("compositions - vmap_jacrev", lambda: self._check_comp_vmap_jacrev())

        # vmap + jacfwd 组合
        _run_case("compositions - vmap_jacfwd", lambda: self._check_comp_vmap_jacfwd())

        # 标量输出 f 下：jvp(f,x,v) = vjp(f,x, default=ones) · v
        _run_case("compositions - jvp_vjp_scalar_math", lambda: self._check_comp_jvp_vjp_scalar())

    def _check_comp_grad_grad(self):
        f = lambda x: (x ** 4).sum()
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        d1f = func.grad(f, create_graph=True)
        grad_norm = lambda a: d1f(a).sum()
        d2y = func.grad(grad_norm)(x)
        return _close(d2y, 12.0 * x._array ** 2)

    def _check_comp_hess_chain(self):
        f = lambda x: (x ** 4 + x ** 2).sum()
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        hess_direct = func.hessian(f)(x)
        grad_f = func.grad(f, create_graph=True)
        hess_via_jacrev = func.jacrev(grad_f)(x)
        return _close(hess_via_jacrev, hess_direct)

    def _check_comp_jvp_fd(self):
        f = lambda x: (x ** 3 + x ** 2).sum()
        x_np = np.array([1.0, 2.0, 3.0])
        v_np = np.array([0.3, 0.7, 0.0])
        x = _rm_tensor(x_np)
        v = _rm_tensor(v_np, requires_grad=False)
        _, jvp_res = func.jvp(f, x, v)
        eps = 1e-5
        f_p = (x_np + eps * v_np) ** 3 + (x_np + eps * v_np) ** 2
        f_m = (x_np - eps * v_np) ** 3 + (x_np - eps * v_np) ** 2
        fd = (f_p.sum() - f_m.sum()) / (2 * eps)
        return _close(jvp_res, fd, atol=1e-4)

    def _check_comp_jvp_vjp(self):
        # 对于 f: R^n -> R^m，JVP 和 VJP 使用相同的 Jacobian J
        # JVP(x, v) = J·v；VJP(x, c) = J^T·c
        # 取 c = JVP(x, v)，则 VJP(x, JVP(x,v)) = J^T J v
        f = lambda x: x ** 3  # f: R^3 -> R^3, 向量输出
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        v = _rm_tensor(np.array([0.5, 1.0, 0.5]), requires_grad=False)
        _, jvp_val = func.jvp(f, x, v)
        _, vjp_val = func.vjp(f, x, jvp_val)
        # 验证形状一致（VJP 应回到输入空间 R^3）
        return vjp_val.shape == (3,)

    def _check_comp_jvp_vjp_scalar(self):
        # 标量输出 f: jvp(f,x,v) = ∇f·v，vjp(f,x)（默认 cot=ones）= ∇f
        f = lambda x: (x ** 2).sum() + x.sum()  # ∇f = 2x+1
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        v = _rm_tensor(np.array([0.5, 1.0, 0.5]), requires_grad=False)
        _, jvpr = func.jvp(f, x, v)
        _, vjpr = func.vjp(f, x)
        expected_grad = np.array([3.0, 5.0, 7.0])  # 2x+1
        # jvpr = grad · v = 3*0.5 + 5*1 + 7*0.5 = 1.5+5+3.5 = 10
        return _close(vjpr, expected_grad) and _close(jvpr, 10.0)

    def _check_comp_vmap_jacrev(self):
        # vmap(jacrev(f)): 批量对每个样本独立求 Jacobian
        f = lambda x: (x ** 2).sum()
        x = _rm_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        out = func.vmap(func.jacrev(f))(x)
        # 每个样本的 ∇f = 2x → 形状 (batch, in_dim) = (2, 3)
        return (out.shape == (2, 3)
                and _close(out[0], np.array([2.0, 4.0, 6.0]))
                and _close(out[1], np.array([8.0, 10.0, 12.0])))

    def _check_comp_vmap_jacfwd(self):
        f = lambda x: (x ** 2).sum()
        x = _rm_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        out = func.vmap(func.jacfwd(f))(x)
        return (out.shape == (2, 3)
                and _close(out[0], np.array([2.0, 4.0, 6.0]))
                and _close(out[1], np.array([8.0, 10.0, 12.0])))

    # ---------------------------------------------------------------
    # argnums 校验（错误信息与 torch.func 一致）
    # ---------------------------------------------------------------
    def test_argnums_validation(self):
        f = lambda x, y: (x ** 2).sum() + x @ y

        _run_case("argnums - grad () raises non-empty",
                  lambda: self._argnums_raises(func.grad, f, (), "argnums must be non-empty"))
        _run_case("argnums - grad None raises type error",
                  lambda: self._argnums_raises(func.grad, f, None,
                                               "argnums must be int or Tuple[int, ...], got: <class 'NoneType'>"))
        _run_case("argnums - grad list raises type error",
                  lambda: self._argnums_raises(func.grad, f, [0],
                                               "argnums must be int or Tuple[int, ...], got: <class 'list'>"))
        _run_case("argnums - grad non-int element raises",
                  lambda: self._argnums_raises(func.grad, f, (0, "a"),
                                               "argnum must be int, got: <class 'str'>"))
        _run_case("argnums - grad_and_value None raises",
                  lambda: self._argnums_raises(func.grad_and_value, f, None,
                                               "argnums must be int or Tuple[int, ...], got: <class 'NoneType'>"))
        _run_case("argnums - jacrev () raises",
                  lambda: self._argnums_raises(func.jacrev, f, (), "argnums must be non-empty"))
        _run_case("argnums - jacrev None raises",
                  lambda: self._argnums_raises(func.jacrev, f, None,
                                               "argnums must be int or Tuple[int, ...], got: <class 'NoneType'>"))
        _run_case("argnums - jacfwd () raises",
                  lambda: self._argnums_raises(func.jacfwd, f, (), "argnums must be non-empty"))
        _run_case("argnums - hessian () raises",
                  lambda: self._argnums_raises(func.hessian, f, (), "argnums must be non-empty"))
        _run_case("argnums - hessian None raises",
                  lambda: self._argnums_raises(func.hessian, f, None,
                                               "argnums must be int or Tuple[int, ...], got: <class 'NoneType'>"))
        # 校验在调用导函数时进行（构造导函数本身不报错），与 torch.func 一致
        _run_case("argnums - validation deferred to call",
                  lambda: self._check_argnums_deferred(f))

        # jacfwd 的 None 表示“全部参数”（torch.func.jacfwd 语义），且始终返回元组
        _run_case("argnums - jacfwd None all args (single input)",
                  lambda: self._check_jacfwd_none_single())
        _run_case("argnums - jacfwd None all args (two inputs)",
                  lambda: self._check_jacfwd_none_multi())

    def _argnums_raises(self, fn_factory, f, argnums, expected_msg, **kwargs):
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        wrapped = fn_factory(f, argnums=argnums, **kwargs)
        try:
            wrapped(x, y)
            return False
        except RuntimeError as e:
            return expected_msg in str(e)
        except Exception:
            return False

    def _check_argnums_deferred(self, f):
        wrapped = func.grad(f, argnums=())
        if not callable(wrapped):
            return False
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        try:
            wrapped(x, y)
            return False
        except RuntimeError:
            return True

    def _check_jacfwd_none_single(self):
        g = lambda x: (x ** 3).sum()
        x = _rm_tensor(np.array([1.0, 2.0]))
        res = func.jacfwd(g, argnums=None)(x)
        # 单参数时也返回元组（与 torch.func.jacfwd 一致）
        return (isinstance(res, tuple) and len(res) == 1
                and _close(res[0], np.array([3.0, 12.0])))

    def _check_jacfwd_none_multi(self):
        f = lambda x, y: (x ** 2).sum() + x @ y
        x = _rm_tensor(np.array([1.0, 2.0]))
        y = _rm_tensor(np.array([3.0, 4.0]))
        res_none = func.jacfwd(f, argnums=None)(x, y)
        res_both = func.jacfwd(f, argnums=(0, 1))(x, y)
        return (isinstance(res_none, tuple) and len(res_none) == 2
                and _close(res_none[0], res_both[0])
                and _close(res_none[1], res_both[1]))

    # ---------------------------------------------------------------
    # 高阶导数（对同一函数反复应用本模块原语）
    # ---------------------------------------------------------------
    def test_high_order_derivatives(self):
        cases = [
            ("grad - 3rd order",                lambda: self._ho_grad_3rd()),
            ("grad_and_value - 2nd order",      lambda: self._ho_gav_2nd()),
            ("jvp - 2nd order",                 lambda: self._ho_jvp_2nd()),
            ("vjp - 2nd order",                 lambda: self._ho_vjp_2nd()),
            ("jacrev - 2nd order",              lambda: self._ho_jacrev_2nd()),
            ("jacfwd - 2nd order",              lambda: self._ho_jacfwd_2nd()),
            ("jacfwd(jacrev) - equals hessian", lambda: self._ho_jacfwd_jacrev()),
            ("jacrev(jacfwd) - equals hessian", lambda: self._ho_jacrev_jacfwd()),
            ("hessian - 3rd order",             lambda: self._ho_hessian_3rd()),
            ("vmap(grad) - 2nd order",          lambda: self._ho_vmap_grad_2nd()),
            ("vmap(jacrev) - 2nd order",        lambda: self._ho_vmap_jacrev_2nd()),
            ("functional_call - 2nd order",     lambda: self._ho_functional_call_2nd()),
        ]
        for desc, fn in cases:
            _run_case(f"high_order - {desc}", fn)

    def _ho_grad_3rd(self):
        # f = sum(x^4): f'=4x^3, f''=12x^2, f'''=24x
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        d1 = func.grad(lambda a: (a ** 4).sum(), create_graph=True)
        d2 = func.grad(lambda a: d1(a).sum(), create_graph=True)
        d3 = func.grad(lambda a: d2(a).sum())(x)
        return _close(d3, 24.0 * np.array([1.0, 2.0, 3.0]))

    def _ho_gav_2nd(self):
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        gav = func.grad_and_value(lambda a: (a ** 4).sum(), create_graph=True)
        _, v = gav(x)
        d2 = func.grad(lambda a: gav(a)[0].sum())(x)
        return _close(v, 98.0) and _close(d2, 12.0 * np.array([1.0, 4.0, 9.0]))

    def _ho_jvp_2nd(self):
        # h(a) = jvp(sum(z^3), a, v)[1] = 3*sum(a^2*v); jvp(h, x, v)[1] = 6*sum(x*v^2)
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        v = _rm_tensor(np.array([1.0, 1.0, 1.0]), requires_grad=False)
        h = lambda a: func.jvp(lambda z: (z ** 3).sum(), (a,), (v,), create_graph=True)[1]
        _, res = func.jvp(h, (x,), (v,), create_graph=True)
        return _close(res, 36.0)

    def _ho_vjp_2nd(self):
        # vjp 结果为 4x^3，再对输入求导应得 12x^2
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        _, g = func.vjp(lambda z: (z ** 4).sum(), (x,),
                        _rm_tensor(np.array(1.0), requires_grad=False), create_graph=True)
        d2 = func.grad(lambda a: g.sum())(x)
        return _close(d2, 12.0 * np.array([1.0, 4.0, 9.0]))

    def _ho_jacrev_2nd(self):
        # jacrev(jacrev(sum(x^3)) 对输入求导 = 6*diag(x)
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        f = lambda b: (b ** 3).sum()
        h = func.jacrev(lambda a: func.jacrev(f, create_graph=True)(a), create_graph=True)(x)
        return _close(h, np.diag(6.0 * x._array))

    def _ho_jacfwd_2nd(self):
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        f = lambda b: (b ** 3).sum()
        h = func.jacfwd(lambda a: func.jacfwd(f, create_graph=True)(a), create_graph=True)(x)
        return _close(h, np.diag(6.0 * x._array))

    def _ho_jacfwd_jacrev(self):
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        f = lambda b: (b ** 3).sum()
        return _close(func.jacfwd(lambda a: func.jacrev(f, create_graph=True)(a),
                                  create_graph=True)(x), np.diag(6.0 * x._array))

    def _ho_jacrev_jacfwd(self):
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        f = lambda b: (b ** 3).sum()
        return _close(func.jacrev(lambda a: func.jacfwd(f, create_graph=True)(a),
                                  create_graph=True)(x), np.diag(6.0 * x._array))

    def _ho_hessian_3rd(self):
        # hessian(sum(x^3)) = 6*diag(x)，对其和再求导得 6*ones
        x = _rm_tensor(np.array([1.0, 2.0, 3.0]))
        f = lambda b: (b ** 3).sum()
        d3 = func.jacrev(lambda a: func.hessian(f, create_graph=True)(a).sum(),
                         create_graph=True)(x)
        return _close(d3, 6.0 * np.ones(3))

    def _ho_vmap_grad_2nd(self):
        # 逐样本 grad of sum(z^3) = 3z^2，再对输入求导得 6z
        z = _rm_tensor(np.array([1.0, 2.0]))
        per_sample = lambda v: (v ** 3).sum()
        res = func.grad(lambda a: func.vmap(func.grad(per_sample, create_graph=True))(a).sum())(z)
        return _close(res, 6.0 * z._array)

    def _ho_vmap_jacrev_2nd(self):
        x = _rm_tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        f = lambda b: (b ** 3).sum()
        res = func.jacrev(lambda p: func.vmap(lambda r: func.jacrev(f, create_graph=True)(r))(p).sum())(x)
        return _close(res, 6.0 * x._array)

    def _ho_functional_call_2nd(self):
        # y = x @ W, f = sum(y^3) => H = 6 * W diag(x @ W) W^T
        class Linear:
            def __init__(self, w): self.w = w
            def forward(self, x): return x @ self.w
            def named_parameters(self): yield "w", self.w

        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        model = Linear(_rm_tensor(W, requires_grad=False))
        w = _rm_tensor(W)
        x = _rm_tensor(np.array([1.0, 2.0]))
        f = lambda a: (func.functional_call(model, [w], (a,)) ** 3).sum()
        h = func.jacrev(lambda a: func.grad(f, create_graph=True)(a), create_graph=True)(x)
        return _close(h, np.array([[282.0, 606.0], [606.0, 1338.0]]))


# ---------------------------------------------------------------------------
# 主入口（独立脚本运行）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    IS_RUNNING_AS_SCRIPT = True

    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")

    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行 riemann.func 模块测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestFuncs)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    stats.print_summary()
    sys.exit(0 if result.wasSuccessful() else 1)
