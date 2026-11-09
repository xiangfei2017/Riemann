"""
tests/test_011_func_vs_torch.py
==============================
riemann.func 与 torch.func 的数值对照测试（含高阶导数）。

目的:
    逐个接口比较 riemann.func 与 torch.func 的输出，验证两者在以下场景下
    数值一致（float64，atol/rtol = 1e-9）:
        grad / grad_and_value, jvp, vjp, jacrev, jacfwd,
        hessian, vmap, functional_call
    以及由这些原语嵌套得到的高阶导数：
        grad(grad), grad(grad(grad)), jacrev(jacfwd), jacfwd(jacrev),
        jacrev(jacrev), jacrev(hessian), grad(jvp), grad(vjp)。

已知语义差异（在 test_api_differences 中显式核对，而不是当作失败）:
    1. riemann 的 grad 默认 create_graph=False，嵌套求导必须显式传
       create_graph=True；torch.func 的变换天然可再求导。
    2. riemann 的 jacrev/jacfwd 默认 strict=True，输入与输出不连通时抛
       RuntimeError；torch.func 返回零块。传 strict=False 后数值一致。
    3. torch.func.hessian(argnums=(0,1)) 返回完整的分块矩阵（含交叉块）；
       riemann.func.hessian 只返回各 argnums 位置的对角块。
    4. torch.func.vjp 对单输入也返回长度为 1 的 tuple；riemann.func.vjp
       单输入时直接返回 TN（多输入才返回 tuple）。

运行:
    独立脚本 → 详细打印 + 统计表；pytest → 仅断言。
    未安装 torch 时整个测试类被跳过。
"""

import os
import sys
import time
import unittest
import warnings

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import riemann as rm
from riemann import func, tensor

try:
    with warnings.catch_warnings():
        # 屏蔽导入 torch 及其内部初始化阶段的一次性弃用告警
        warnings.simplefilter("ignore")
        import torch
        import torch.nn as tnn

        # 预热 torch.func 的 JVP 分解缓存：首次调用 JVP 路径会懒加载
        # torch._decomp 的 JVP 分解，并在内部批量调用 torch.jit.script，
        # 从而产生与测试无关的 FutureWarning（Python 3.11 为 "deprecated"，
        # 3.14+ 为 "not supported"）。在此一次性触发，后续用例不再告警。
        _warmup_x = torch.tensor([0.0])
        torch.func.jvp(lambda t: t * 2, (_warmup_x,), (_warmup_x,))
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    tnn = None
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colors / StatisticsCollector / 运行控制（与 test_010 保持一致）
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

    def add_result(self, passed):
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
            pt = f"{st['passed']}/{st['total']}"
            pt_w = self._display_width(pt)
            pr_s = f"{pr:.2f}"
            pr_w = self._display_width(pr_s)
            ti = f"{st['time']:.4f}"
            ti_w = self._display_width(ti)
            print(
                f"{fn}{' ' * (cw[0] - fn_w)}"
                f"{pt}{' ' * (cw[1] - pt_w)}"
                f"{sc}{pr_s}{' ' * (cw[2] - pr_w)}{Colors.ENDC}"
                f"{ti}{' ' * (cw[3] - ti_w)}"
            )

        print("=" * tw)


stats = StatisticsCollector()
IS_RUNNING_AS_SCRIPT = False


# ---------------------------------------------------------------------------
# 张量构造与结构比对
# ---------------------------------------------------------------------------

def _rm(arr, requires_grad=True):
    """构造 riemann float64 张量（arr 可为标量、list 或 ndarray）。"""
    return tensor(np.asarray(arr, dtype=np.float64), requires_grad=requires_grad)


def _tt(arr, requires_grad=True):
    """构造 torch float64 张量（arr 可为标量、list 或 ndarray）。"""
    return torch.tensor(np.asarray(arr, dtype=np.float64), requires_grad=requires_grad)


def _to_numpy(value):
    """把 TN / torch.Tensor / tuple / list / 标量统一转换为 numpy 结构。"""
    if hasattr(value, "_array"):
        value = value._array
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (tuple, list)):
        return tuple(_to_numpy(v) for v in value)
    return np.asarray(value, dtype=np.float64)


def _tree_close(a, b, atol=1e-9, rtol=1e-9):
    """递归比较两个结果树（tuple 结构与元素形状、数值均需一致）。"""
    a = _to_numpy(a)
    b = _to_numpy(b)
    if isinstance(a, tuple) or isinstance(b, tuple):
        if not (isinstance(a, tuple) and isinstance(b, tuple)):
            return False
        return len(a) == len(b) and all(
            _tree_close(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b)
        )
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=atol, rtol=rtol))


def _unwrap_single(value):
    """把长度为 1 的 tuple 解包（torch.func.vjp 单输入固定返回 tuple）。"""
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


def _run_case(case_name, fn):
    """执行单个测试用例：返回 bool，失败时抛 AssertionError 供 pytest 捕获。"""
    t0 = time.time()
    passed = False
    err = None
    try:
        passed = bool(fn())
    except Exception as e:
        err = str(e)

    time_taken = time.time() - t0
    stats.add_result(passed)

    if IS_RUNNING_AS_SCRIPT:
        if passed:
            print(f"  {Colors.OKGREEN}通过{Colors.ENDC}  {case_name}  ({time_taken:.4f}s)")
        else:
            print(f"  {Colors.FAIL}失败{Colors.ENDC}  {case_name}  ({time_taken:.4f}s)"
                  + (f"  — {err}" if err else ""))

    if not passed:
        raise AssertionError(f"{case_name}: {err or '与 torch.func 结果不一致'}")
    return passed


# ---------------------------------------------------------------------------
# 对照测试
# ---------------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "未安装 torch，跳过 riemann.func 与 torch.func 的对照测试")
class TestFuncVsTorch(unittest.TestCase):
    """riemann.func 与 torch.func 的逐接口数值对照。"""

    def setUp(self):
        self.current_test_name = self._testMethodName
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")

    def tearDown(self):
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()

    # ---------------------------------------------------------------
    # grad / grad_and_value
    # ---------------------------------------------------------------
    def test_grad(self):
        _run_case("grad - vector", lambda: self._cmp_grad_vector())
        _run_case("grad - matrix", lambda: self._cmp_grad_matrix())
        _run_case("grad - multi_argnums", lambda: self._cmp_grad_multi_argnums())
        _run_case("grad - argnums_second", lambda: self._cmp_grad_argnums_second())
        _run_case("grad - has_aux", lambda: self._cmp_grad_has_aux())
        _run_case("grad - create_graph_value", lambda: self._cmp_grad_create_graph())
        _run_case("grad_and_value - grad_part", lambda: self._cmp_grad_and_value())

    def _cmp_grad_vector(self):
        f = lambda x: (x ** 3).sum()
        rm_x = _rm([1.0, 2.0, 3.0])
        tt_x = _tt([1.0, 2.0, 3.0])
        return _tree_close(func.grad(f)(rm_x), torch.func.grad(f)(tt_x))

    def _cmp_grad_matrix(self):
        f = lambda m: (m ** 3).sum()
        rm_m = _rm([[1.0, 2.0], [3.0, 4.0]])
        tt_m = _tt([[1.0, 2.0], [3.0, 4.0]])
        return _tree_close(func.grad(f)(rm_m), torch.func.grad(f)(tt_m))

    def _cmp_grad_multi_argnums(self):
        f = lambda x, w: (x * w).sum()
        rm_x, rm_w = _rm([1.0, 2.0]), _rm([0.5, 1.0])
        tt_x, tt_w = _tt([1.0, 2.0]), _tt([0.5, 1.0])
        return _tree_close(
            func.grad(f, argnums=(0, 1))(rm_x, rm_w),
            torch.func.grad(f, argnums=(0, 1))(tt_x, tt_w),
        )

    def _cmp_grad_argnums_second(self):
        f = lambda x, w: (x**2 * w).sum()
        rm_x, rm_w = _rm([1.0, 2.0]), _rm([0.5, 1.0])
        tt_x, tt_w = _tt([1.0, 2.0]), _tt([0.5, 1.0])
        return _tree_close(
            func.grad(f, argnums=1)(rm_x, rm_w),
            torch.func.grad(f, argnums=1)(tt_x, tt_w),
        )

    def _cmp_grad_has_aux(self):
        def f(x):
            # torch.func 的 has_aux 要求 aux 是张量（或张量组成的 pytree）
            return (x ** 2).sum(), x.sum()

        rm_g, rm_aux = func.grad(f, has_aux=True)(_rm([1.0, 2.0]))
        tt_g, tt_aux = torch.func.grad(f, has_aux=True)(_tt([1.0, 2.0]))
        return _tree_close(rm_g, tt_g) and _tree_close(rm_aux, tt_aux)

    def _cmp_grad_create_graph(self):
        f = lambda x: (x ** 3).sum()
        rm_g = func.grad(f, create_graph=True)(_rm([1.0, 2.0]))
        tt_g = torch.func.grad(f)(_tt([1.0, 2.0]))
        return _tree_close(rm_g, tt_g) and rm_g.requires_grad

    def _cmp_grad_and_value(self):
        f = lambda x: (x ** 3).sum()
        rm_x, tt_x = _rm([1.0, 2.0]), _tt([1.0, 2.0])
        rm_g, rm_v = func.grad_and_value(f)(rm_x)
        return _tree_close(rm_g, torch.func.grad(f)(tt_x)) and _tree_close(rm_v, f(tt_x))

    # ---------------------------------------------------------------
    # jvp
    # ---------------------------------------------------------------
    def test_jvp(self):
        _run_case("jvp - scalar_output", lambda: self._cmp_jvp_scalar_output())
        _run_case("jvp - vector_output", lambda: self._cmp_jvp_vector_output())
        _run_case("jvp - multi_input", lambda: self._cmp_jvp_multi_input())
        _run_case("jvp - multi_output", lambda: self._cmp_jvp_multi_output())
        _run_case("jvp - default_tangent", lambda: self._cmp_jvp_default_tangent())
        _run_case("jvp - create_graph_value", lambda: self._cmp_jvp_create_graph())

    def _cmp_jvp_scalar_output(self):
        f = lambda x: (x ** 3).sum()
        rm_x, rm_v = _rm([1.0, 2.0]), _rm([1.0, -1.0], requires_grad=False)
        tt_x, tt_v = _tt([1.0, 2.0]), _tt([1.0, -1.0], requires_grad=False)
        rm_out, rm_jvp = func.jvp(f, rm_x, rm_v)
        tt_out, tt_jvp = torch.func.jvp(f, (tt_x,), (tt_v,))
        return _tree_close((rm_out, rm_jvp), (tt_out, tt_jvp))

    def _cmp_jvp_vector_output(self):
        f = lambda x: x ** 3
        rm_x, rm_v = _rm([1.0, 2.0]), _rm([0.5, 2.0], requires_grad=False)
        tt_x, tt_v = _tt([1.0, 2.0]), _tt([0.5, 2.0], requires_grad=False)
        rm_out, rm_jvp = func.jvp(f, rm_x, rm_v)
        tt_out, tt_jvp = torch.func.jvp(f, (tt_x,), (tt_v,))
        return _tree_close((rm_out, rm_jvp), (tt_out, tt_jvp))

    def _cmp_jvp_multi_input(self):
        f = lambda x, y: (x * y).sum()
        rm_x, rm_y = _rm([1.0, 2.0]), _rm([3.0, 4.0])
        tt_x, tt_y = _tt([1.0, 2.0]), _tt([3.0, 4.0])
        rm_vx = _rm([1.0, 0.0], requires_grad=False)
        rm_vy = _rm([0.0, 1.0], requires_grad=False)
        tt_vx = _tt([1.0, 0.0], requires_grad=False)
        tt_vy = _tt([0.0, 1.0], requires_grad=False)
        rm_out, rm_jvp = func.jvp(f, (rm_x, rm_y), (rm_vx, rm_vy))
        tt_out, tt_jvp = torch.func.jvp(f, (tt_x, tt_y), (tt_vx, tt_vy))
        return _tree_close((rm_out, rm_jvp), (tt_out, tt_jvp))

    def _cmp_jvp_multi_output(self):
        f = lambda x: (x.sum(), (x ** 2).sum())
        rm_x, rm_v = _rm([1.0, 2.0, 3.0]), _rm([1.0, 0.0, 1.0], requires_grad=False)
        tt_x, tt_v = _tt([1.0, 2.0, 3.0]), _tt([1.0, 0.0, 1.0], requires_grad=False)
        rm_out, rm_jvp = func.jvp(f, rm_x, rm_v)
        tt_out, tt_jvp = torch.func.jvp(f, (tt_x,), (tt_v,))
        return _tree_close((rm_out, rm_jvp), (tt_out, tt_jvp))

    def _cmp_jvp_default_tangent(self):
        # riemann 扩展：tangents=None 表示全 1 方向；torch 需显式传入 ones
        f = lambda x: (x ** 3).sum()
        rm_x, tt_x = _rm([1.0, 2.0]), _tt([1.0, 2.0])
        rm_out, rm_jvp = func.jvp(f, rm_x, None)
        tt_out, tt_jvp = torch.func.jvp(
            f, (tt_x,), (torch.ones_like(tt_x, requires_grad=False),)
        )
        return _tree_close((rm_out, rm_jvp), (tt_out, tt_jvp))

    def _cmp_jvp_create_graph(self):
        f = lambda x: (x ** 3).sum()
        rm_x, rm_v = _rm([1.0, 2.0]), _rm([1.0, -1.0], requires_grad=False)
        tt_x, tt_v = _tt([1.0, 2.0]), _tt([1.0, -1.0], requires_grad=False)
        _, rm_jvp = func.jvp(f, rm_x, rm_v, create_graph=True)
        _, tt_jvp = torch.func.jvp(f, (tt_x,), (tt_v,))
        return _tree_close(rm_jvp, tt_jvp) and rm_jvp.requires_grad

    # ---------------------------------------------------------------
    # vjp
    # ---------------------------------------------------------------
    def test_vjp(self):
        _run_case("vjp - scalar_output", lambda: self._cmp_vjp_scalar_output())
        _run_case("vjp - vector_output", lambda: self._cmp_vjp_vector_output())
        _run_case("vjp - multi_input", lambda: self._cmp_vjp_multi_input())
        _run_case("vjp - multi_output", lambda: self._cmp_vjp_multi_output())
        _run_case("vjp - default_cotangent", lambda: self._cmp_vjp_default_cotangent())
        _run_case("vjp - create_graph_value", lambda: self._cmp_vjp_create_graph())

    def _cmp_vjp_scalar_output(self):
        f = lambda x: (x ** 3).sum()
        rm_x, tt_x = _rm([1.0, 2.0]), _tt([1.0, 2.0])
        rm_out, rm_g = func.vjp(f, rm_x, _rm(2.0, requires_grad=False))
        tt_out, tt_vfn = torch.func.vjp(f, tt_x)
        tt_g = tt_vfn(_tt(2.0, requires_grad=False))
        return _tree_close((rm_out, rm_g), (tt_out, _unwrap_single(tt_g)))

    def _cmp_vjp_vector_output(self):
        f = lambda x: x ** 3
        rm_x, tt_x = _rm([1.0, 2.0]), _tt([1.0, 2.0])
        rm_cot = _rm([1.0, 2.0], requires_grad=False)
        tt_cot = _tt([1.0, 2.0], requires_grad=False)
        rm_out, rm_g = func.vjp(f, rm_x, rm_cot)
        tt_out, tt_vfn = torch.func.vjp(f, tt_x)
        tt_g = tt_vfn(tt_cot)
        return _tree_close((rm_out, rm_g), (tt_out, _unwrap_single(tt_g)))

    def _cmp_vjp_multi_input(self):
        f = lambda x, y: (x * y).sum()
        rm_x, rm_y = _rm([1.0, 2.0]), _rm([3.0, 4.0])
        tt_x, tt_y = _tt([1.0, 2.0]), _tt([3.0, 4.0])
        rm_out, rm_g = func.vjp(f, (rm_x, rm_y), _rm(1.0, requires_grad=False))
        tt_out, tt_vfn = torch.func.vjp(f, tt_x, tt_y)
        tt_g = tt_vfn(_tt(1.0, requires_grad=False))
        return _tree_close((rm_out, rm_g), (tt_out, tt_g))

    def _cmp_vjp_multi_output(self):
        f = lambda x: (x.sum(), (x ** 2).sum())
        rm_x, tt_x = _rm([1.0, 2.0]), _tt([1.0, 2.0])
        rm_cot = (_rm(2.0, requires_grad=False), _rm(3.0, requires_grad=False))
        tt_cot = (_tt(2.0, requires_grad=False), _tt(3.0, requires_grad=False))
        rm_out, rm_g = func.vjp(f, rm_x, rm_cot)
        tt_out, tt_vfn = torch.func.vjp(f, tt_x)
        tt_g = tt_vfn(tt_cot)
        return _tree_close((rm_out, rm_g), (tt_out, _unwrap_single(tt_g)))

    def _cmp_vjp_default_cotangent(self):
        # cotangents=None 等价于全 1；torch 需显式传入 ones_like
        f = lambda x: x ** 3
        rm_x, tt_x = _rm([1.0, 2.0]), _tt([1.0, 2.0])
        rm_out, rm_g = func.vjp(f, rm_x, None)
        tt_out, tt_vfn = torch.func.vjp(f, tt_x)
        tt_g = tt_vfn(torch.ones_like(tt_x, requires_grad=False))
        return _tree_close((rm_out, rm_g), (tt_out, _unwrap_single(tt_g)))

    def _cmp_vjp_create_graph(self):
        f = lambda x: (x ** 3).sum()
        rm_x, tt_x = _rm([1.0, 2.0]), _tt([1.0, 2.0])
        rm_out, rm_g = func.vjp(f, rm_x, _rm(2.0, requires_grad=False), create_graph=True)
        tt_out, tt_vfn = torch.func.vjp(f, tt_x)
        tt_g = _unwrap_single(tt_vfn(_tt(2.0, requires_grad=False)))
        return _tree_close((rm_out, rm_g), (tt_out, tt_g)) and rm_g.requires_grad

    # ---------------------------------------------------------------
    # jacrev
    # ---------------------------------------------------------------
    def test_jacrev(self):
        _run_case("jacrev - vector_to_vector", lambda: self._cmp_jacrev_vector())
        _run_case("jacrev - matrix_input", lambda: self._cmp_jacrev_matrix_input())
        _run_case("jacrev - multi_output", lambda: self._cmp_jacrev_multi_output())
        _run_case("jacrev - multi_input", lambda: self._cmp_jacrev_multi_input())
        _run_case("jacrev - noncontiguous_argnums",
                  lambda: self._cmp_jacrev_noncontiguous())
        _run_case("jacrev - scalar_input", lambda: self._cmp_jacrev_scalar_input())

    def _cmp_jacrev_vector(self):
        f = lambda x: rm.stack([x[0] ** 2 * x[1], x[1] ** 3])
        tf = lambda x: torch.stack([x[0] ** 2 * x[1], x[1] ** 3])
        return _tree_close(func.jacrev(f)(_rm([1.0, 2.0])) , torch.func.jacrev(tf)(_tt([1.0, 2.0])))

    def _cmp_jacrev_matrix_input(self):
        f = lambda m: (m ** 3).sum()
        return _tree_close(
            func.jacrev(f)(_rm([[1.0, 2.0], [3.0, 4.0]])),
            torch.func.jacrev(f)(_tt([[1.0, 2.0], [3.0, 4.0]])),
        )

    def _cmp_jacrev_multi_output(self):
        f = lambda x: (x.sum(), (x ** 2).sum())
        return _tree_close(
            func.jacrev(f)(_rm([1.0, 2.0])),
            torch.func.jacrev(f)(_tt([1.0, 2.0])),
        )

    def _cmp_jacrev_multi_input(self):
        # 两个输出同时依赖 x 与 y，避免触发 strict 的未使用输入检查
        f = lambda x, y: ((x * y).sum(), (x * y ** 2).sum())
        rm_x, rm_y = _rm([1.0, 2.0]), _rm([3.0, 4.0])
        tt_x, tt_y = _tt([1.0, 2.0]), _tt([3.0, 4.0])
        return _tree_close(
            func.jacrev(f, argnums=(0, 1))(rm_x, rm_y),
            torch.func.jacrev(f, argnums=(0, 1))(tt_x, tt_y),
        )

    def _cmp_jacrev_noncontiguous(self):
        # 中间参数 w 未被使用，riemann 需要 strict=False（差异见 test_api_differences）
        f = lambda x, w, z: (x ** 2 * z).sum() + (z ** 3).sum()
        rm_args = (_rm([1.0, 2.0]), _rm([1.0, 1.0]), _rm([2.0, 3.0]))
        tt_args = (_tt([1.0, 2.0]), _tt([1.0, 1.0]), _tt([2.0, 3.0]))
        return _tree_close(
            func.jacrev(f, argnums=(0, 2), strict=False)(*rm_args),
            torch.func.jacrev(f, argnums=(0, 2))(*tt_args),
        )

    def _cmp_jacrev_scalar_input(self):
        f = lambda x: x ** 3
        return _tree_close(func.jacrev(f)(_rm(2.0)), torch.func.jacrev(f)(_tt(2.0)))

    # ---------------------------------------------------------------
    # jacfwd
    # ---------------------------------------------------------------
    def test_jacfwd(self):
        _run_case("jacfwd - vector_to_vector", lambda: self._cmp_jacfwd_vector())
        _run_case("jacfwd - matrix_input", lambda: self._cmp_jacfwd_matrix_input())
        _run_case("jacfwd - multi_output", lambda: self._cmp_jacfwd_multi_output())
        _run_case("jacfwd - multi_input", lambda: self._cmp_jacfwd_multi_input())
        _run_case("jacfwd - noncontiguous_argnums",
                  lambda: self._cmp_jacfwd_noncontiguous())
        _run_case("jacfwd - scalar_input", lambda: self._cmp_jacfwd_scalar_input())
        _run_case("jacfwd - equals_jacrev", lambda: self._cmp_jacfwd_equals_jacrev())

    def _cmp_jacfwd_vector(self):
        f = lambda x: rm.stack([x[0] ** 2 * x[1], x[1] ** 3])
        tf = lambda x: torch.stack([x[0] ** 2 * x[1], x[1] ** 3])
        return _tree_close(func.jacfwd(f)(_rm([1.0, 2.0])) , torch.func.jacfwd(tf)(_tt([1.0, 2.0])))

    def _cmp_jacfwd_matrix_input(self):
        f = lambda m: (m ** 3).sum()
        return _tree_close(
            func.jacfwd(f)(_rm([[1.0, 2.0], [3.0, 4.0]])),
            torch.func.jacfwd(f)(_tt([[1.0, 2.0], [3.0, 4.0]])),
        )

    def _cmp_jacfwd_multi_output(self):
        f = lambda x: (x.sum(), (x ** 2).sum())
        return _tree_close(
            func.jacfwd(f)(_rm([1.0, 2.0])),
            torch.func.jacfwd(f)(_tt([1.0, 2.0])),
        )

    def _cmp_jacfwd_multi_input(self):
        f = lambda x, y: (x.sum(), (x * y).sum())
        rm_x, rm_y = _rm([1.0, 2.0]), _rm([3.0, 4.0])
        tt_x, tt_y = _tt([1.0, 2.0]), _tt([3.0, 4.0])
        return _tree_close(
            func.jacfwd(f, argnums=(0, 1))(rm_x, rm_y),
            torch.func.jacfwd(f, argnums=(0, 1))(tt_x, tt_y),
        )

    def _cmp_jacfwd_noncontiguous(self):
        f = lambda x, w, z: (x ** 2 * z).sum() + (z ** 3).sum()
        rm_args = (_rm([1.0, 2.0]), _rm([1.0, 1.0]), _rm([2.0, 3.0]))
        tt_args = (_tt([1.0, 2.0]), _tt([1.0, 1.0]), _tt([2.0, 3.0]))
        return _tree_close(
            func.jacfwd(f, argnums=(0, 2))(*rm_args),
            torch.func.jacfwd(f, argnums=(0, 2))(*tt_args),
        )

    def _cmp_jacfwd_scalar_input(self):
        f = lambda x: x ** 3
        return _tree_close(func.jacfwd(f)(_rm(2.0)), torch.func.jacfwd(f)(_tt(2.0)))

    def _cmp_jacfwd_equals_jacrev(self):
        f = lambda x: rm.stack([x[0] ** 2 * x[1], x[1] ** 3])
        tf = lambda x: torch.stack([x[0] ** 2 * x[1], x[1] ** 3])
        rm_x, tt_x = _rm([1.0, 2.0]), _tt([1.0, 2.0])
        return (_tree_close(func.jacfwd(f)(rm_x), func.jacrev(f)(rm_x))
                and _tree_close(func.jacfwd(f)(rm_x), torch.func.jacfwd(tf)(tt_x)))

    # ---------------------------------------------------------------
    # hessian
    # ---------------------------------------------------------------
    def test_hessian(self):
        _run_case("hessian - quadratic", lambda: self._cmp_hess_quadratic())
        _run_case("hessian - cubic", lambda: self._cmp_hess_cubic())
        _run_case("hessian - matrix_input", lambda: self._cmp_hess_matrix_input())
        _run_case("hessian - scalar_input", lambda: self._cmp_hess_scalar_input())
        _run_case("hessian - multi_argnums", lambda: self._cmp_hess_multi_argnums())
        _run_case("hessian - noncontiguous_argnums",
                  lambda: self._cmp_hess_noncontiguous())

    def _cmp_hess_quadratic(self):
        f = lambda x: (x ** 2).sum()
        rm_h = func.hessian(f)(_rm([1.0, 2.0]))
        tt_h = torch.func.hessian(f)(_tt([1.0, 2.0]))
        return _tree_close(rm_h, tt_h)

    def _cmp_hess_cubic(self):
        f = lambda x: (x ** 3).sum()
        rm_h = func.hessian(f)(_rm([1.0, 2.0]))
        tt_h = torch.func.hessian(f)(_tt([1.0, 2.0]))
        return _tree_close(rm_h, tt_h)

    def _cmp_hess_matrix_input(self):
        f = lambda m: (m ** 3).sum()
        rm_h = func.hessian(f)(_rm([[1.0, 2.0], [3.0, 4.0]]))
        tt_h = torch.func.hessian(f)(_tt([[1.0, 2.0], [3.0, 4.0]]))
        return _tree_close(rm_h, tt_h)

    def _cmp_hess_scalar_input(self):
        f = lambda x: x ** 4
        return _tree_close(func.hessian(f)(_rm(2.0)), torch.func.hessian(f)(_tt(2.0)))

    def _cmp_hess_multi_argnums(self):
        # torch 返回完整分块矩阵，riemann 只返回对角块（见 test_api_differences）
        f = lambda x, y: (x ** 2 * y ** 2).sum()
        rm_h = func.hessian(f, argnums=(0, 1))(_rm([1.0, 2.0]), _rm([3.0, 4.0]))
        tt_h = torch.func.hessian(f, argnums=(0, 1))(_tt([1.0, 2.0]), _tt([3.0, 4.0]))
        return (_tree_close(rm_h[0], tt_h[0][0]) and _tree_close(rm_h[1], tt_h[1][1]))

    def _cmp_hess_noncontiguous(self):
        f = lambda x, w, z: (x ** 2 * z).sum() + (z ** 3).sum()
        rm_args = (_rm([1.0, 2.0]), _rm([1.0, 1.0]), _rm([2.0, 3.0]))
        tt_args = (_tt([1.0, 2.0]), _tt([1.0, 1.0]), _tt([2.0, 3.0]))
        rm_h = func.hessian(f, argnums=(0, 2))(*rm_args)
        tt_h = torch.func.hessian(f, argnums=(0, 2))(*tt_args)
        return (_tree_close(rm_h[0], tt_h[0][0]) and _tree_close(rm_h[1], tt_h[1][1]))

    # ---------------------------------------------------------------
    # vmap
    # ---------------------------------------------------------------
    def test_vmap(self):
        _run_case("vmap - unary", lambda: self._cmp_vmap_unary())
        _run_case("vmap - multi_output", lambda: self._cmp_vmap_multi_output())
        _run_case("vmap - multi_input_out_dims", lambda: self._cmp_vmap_multi_in_out())
        _run_case("vmap - in_dims_negative", lambda: self._cmp_vmap_negative_dim())
        _run_case("vmap - unbatched_input", lambda: self._cmp_vmap_unbatched())
        _run_case("vmap - grad_composition", lambda: self._cmp_vmap_grad())
        _run_case("vmap - jacrev_composition", lambda: self._cmp_vmap_jacrev())
        _run_case("vmap - hessian_composition", lambda: self._cmp_vmap_hessian())

    def _cmp_vmap_unary(self):
        f = lambda row: (row ** 3).sum()
        return _tree_close(
            func.vmap(f)(_rm([[1.0, 2.0], [3.0, 4.0]])),
            torch.func.vmap(f)(_tt([[1.0, 2.0], [3.0, 4.0]])),
        )

    def _cmp_vmap_multi_output(self):
        f = lambda row: (row.sum(), row ** 2)
        return _tree_close(
            func.vmap(f)(_rm([[1.0, 2.0], [3.0, 4.0]])),
            torch.func.vmap(f)(_tt([[1.0, 2.0], [3.0, 4.0]])),
        )

    def _cmp_vmap_multi_in_out(self):
        f = lambda x, y: (x + y, x * y)
        rm_a = _rm([[1.0, 2.0], [3.0, 4.0]])
        rm_b = _rm([[5.0, 6.0], [7.0, 8.0]])
        tt_a = _tt([[1.0, 2.0], [3.0, 4.0]])
        tt_b = _tt([[5.0, 6.0], [7.0, 8.0]])
        return _tree_close(
            func.vmap(f, in_dims=(0, 0), out_dims=(0, 1))(rm_a, rm_b),
            torch.func.vmap(f, in_dims=(0, 0), out_dims=(0, 1))(tt_a, tt_b),
        )

    def _cmp_vmap_negative_dim(self):
        f = lambda x: x ** 2
        return _tree_close(
            func.vmap(f, in_dims=-1)(_rm([[1.0, 2.0], [3.0, 4.0]])),
            torch.func.vmap(f, in_dims=-1)(_tt([[1.0, 2.0], [3.0, 4.0]])),
        )

    def _cmp_vmap_unbatched(self):
        f = lambda x, c: x * c
        rm_x, rm_c = _rm([[1.0, 2.0], [3.0, 4.0]]), _rm(2.0)
        tt_x, tt_c = _tt([[1.0, 2.0], [3.0, 4.0]]), _tt(2.0)
        return _tree_close(
            func.vmap(f, in_dims=(0, None))(rm_x, rm_c),
            torch.func.vmap(f, in_dims=(0, None))(tt_x, tt_c),
        )

    def _cmp_vmap_grad(self):
        f = lambda m: func.vmap(lambda row: (row ** 3).sum())(m).sum()
        tf = lambda m: torch.func.vmap(lambda row: (row ** 3).sum())(m).sum()
        rm_g = func.grad(f)(_rm([[1.0, 2.0], [3.0, 4.0]]))
        tt_g = torch.func.grad(tf)(_tt([[1.0, 2.0], [3.0, 4.0]]))
        return _tree_close(rm_g, tt_g)

    def _cmp_vmap_jacrev(self):
        f = lambda row: (row ** 2).sum()
        rm_j = func.vmap(func.jacrev(f))(_rm([[1.0, 2.0], [3.0, 4.0]]))
        tt_j = torch.func.vmap(torch.func.jacrev(f))(_tt([[1.0, 2.0], [3.0, 4.0]]))
        return _tree_close(rm_j, tt_j)

    def _cmp_vmap_hessian(self):
        f = lambda row: (row ** 3).sum()
        rm_h = func.vmap(func.hessian(f))(_rm([[1.0, 2.0], [3.0, 4.0]]))
        tt_h = torch.func.vmap(torch.func.hessian(f))(_tt([[1.0, 2.0], [3.0, 4.0]]))
        return _tree_close(rm_h, tt_h)

    # ---------------------------------------------------------------
    # functional_call
    # ---------------------------------------------------------------
    def test_functional_call(self):
        _run_case("functional_call - default_params", lambda: self._cmp_fc_default())
        _run_case("functional_call - custom_params", lambda: self._cmp_fc_custom())
        _run_case("functional_call - grad_wrt_params", lambda: self._cmp_fc_grad())
        _run_case("functional_call - kwargs", lambda: self._cmp_fc_kwargs())

    @staticmethod
    def _build_linear_pair():
        """构造参数完全一致的 riemann / torch 线性层。"""
        rm_layer = rm.nn.Linear(2, 3)
        tt_layer = tnn.Linear(2, 3).double()
        with torch.no_grad():
            tt_layer.weight.copy_(torch.tensor(np.asarray(rm_layer.weight._array,
                                                          dtype=np.float64)))
            tt_layer.bias.copy_(torch.tensor(np.asarray(rm_layer.bias._array,
                                                        dtype=np.float64)))
        return rm_layer, tt_layer

    def _cmp_fc_default(self):
        rm_layer, tt_layer = self._build_linear_pair()
        rm_x, tt_x = _rm([[1.0, 2.0], [3.0, 4.0]]), _tt([[1.0, 2.0], [3.0, 4.0]])
        rm_y = func.functional_call(rm_layer, None, (rm_x,))
        tt_y = torch.func.functional_call(tt_layer, dict(tt_layer.named_parameters()),
                                          (tt_x,))
        return _tree_close(rm_y, tt_y)

    def _cmp_fc_custom(self):
        rm_layer, tt_layer = self._build_linear_pair()
        w_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b_np = np.array([1.0, 1.0, 1.0])
        rm_x, tt_x = _rm([[1.0, 2.0]]), _tt([[1.0, 2.0]])
        rm_y = func.functional_call(rm_layer, [_rm(w_np), _rm(b_np)], (rm_x,))
        tt_y = torch.func.functional_call(tt_layer, {"weight": _tt(w_np),
                                                     "bias": _tt(b_np)}, (tt_x,))
        return _tree_close(rm_y, tt_y)

    def _cmp_fc_grad(self):
        rm_layer, tt_layer = self._build_linear_pair()
        w_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b_np = np.array([1.0, 1.0, 1.0])
        rm_x = _rm([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
        tt_x = _tt([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)

        def rm_loss(w, b):
            return (func.functional_call(rm_layer, [w, b], (rm_x,)) ** 2).sum()

        def tt_loss(w, b):
            return (torch.func.functional_call(
                tt_layer, {"weight": w, "bias": b}, (tt_x,)) ** 2).sum()

        rm_g = func.grad(rm_loss, argnums=(0, 1))(_rm(w_np), _rm(b_np))
        tt_g = torch.func.grad(tt_loss, argnums=(0, 1))(_tt(w_np), _tt(b_np))
        return _tree_close(rm_g, tt_g)

    def _cmp_fc_kwargs(self):
        class RmScale:
            def forward(self, x, scale):
                return x * scale

        class TtScale(tnn.Module):
            def forward(self, x, scale):
                return x * scale

        rm_x, tt_x = _rm([[1.0, 2.0]]), _tt([[1.0, 2.0]])
        rm_y = func.functional_call(RmScale(), None, (rm_x,), {"scale": 2.0})
        tt_y = torch.func.functional_call(TtScale(), {}, (tt_x,), {"scale": 2.0})
        return _tree_close(rm_y, tt_y)

    # ---------------------------------------------------------------
    # 高阶导数（嵌套变换）
    # ---------------------------------------------------------------
    def test_high_order(self):
        _run_case("higher_order - grad2_scalar", lambda: self._ho_grad2_scalar())
        _run_case("higher_order - grad3_scalar", lambda: self._ho_grad3_scalar())
        _run_case("higher_order - jacrev_jacfwd", lambda: self._ho_jacrev_jacfwd())
        _run_case("higher_order - jacfwd_jacrev", lambda: self._ho_jacfwd_jacrev())
        _run_case("higher_order - jacrev_jacrev", lambda: self._ho_jacrev_jacrev())
        _run_case("higher_order - jacrev_hessian", lambda: self._ho_jacrev_hessian())
        _run_case("higher_order - hessian_cross_3rd", lambda: self._ho_hessian_cross())
        _run_case("higher_order - grad_jvp", lambda: self._ho_grad_jvp())
        _run_case("higher_order - grad_vjp", lambda: self._ho_grad_vjp())

    def _ho_grad2_scalar(self):
        f = lambda x: x ** 4
        rm_g2 = func.grad(func.grad(f, create_graph=True))(_rm(1.5))
        tt_g2 = torch.func.grad(torch.func.grad(f))(_tt(1.5))
        return _tree_close(rm_g2, tt_g2)

    def _ho_grad3_scalar(self):
        f = lambda x: x ** 4
        rm_g3 = func.grad(func.grad(func.grad(f, create_graph=True),
                                    create_graph=True))(_rm(1.5))
        tt_g3 = torch.func.grad(torch.func.grad(torch.func.grad(f)))(_tt(1.5))
        return _tree_close(rm_g3, tt_g3)

    @staticmethod
    def _vector_func_pair():
        rm_f = lambda x: rm.stack([x[0] ** 2 * x[1], x[1] ** 3])
        tt_f = lambda x: torch.stack([x[0] ** 2 * x[1], x[1] ** 3])
        return rm_f, tt_f

    def _ho_jacrev_jacfwd(self):
        rm_f, tt_f = self._vector_func_pair()
        return _tree_close(
            func.jacrev(func.jacfwd(rm_f, create_graph=True))(_rm([1.0, 2.0])),
            torch.func.jacrev(torch.func.jacfwd(tt_f))(_tt([1.0, 2.0])),
        )

    def _ho_jacfwd_jacrev(self):
        rm_f, tt_f = self._vector_func_pair()
        return _tree_close(
            func.jacfwd(func.jacrev(rm_f, create_graph=True))(_rm([1.0, 2.0])),
            torch.func.jacfwd(torch.func.jacrev(tt_f))(_tt([1.0, 2.0])),
        )

    def _ho_jacrev_jacrev(self):
        rm_f, tt_f = self._vector_func_pair()
        return _tree_close(
            func.jacrev(func.jacrev(rm_f, create_graph=True))(_rm([1.0, 2.0])),
            torch.func.jacrev(torch.func.jacrev(tt_f))(_tt([1.0, 2.0])),
        )

    def _ho_jacrev_hessian(self):
        f = lambda x: (x ** 3).sum()
        return _tree_close(
            func.jacrev(func.hessian(f, create_graph=True))(_rm([1.0, 2.0])),
            torch.func.jacrev(torch.func.hessian(f))(_tt([1.0, 2.0])),
        )

    def _ho_hessian_cross(self):
        # 含交叉项的标量函数，三阶张量中会出现非对角元
        f = lambda x: (x[0] ** 2 * x[1] + x[1] ** 3)
        return _tree_close(
            func.jacrev(func.hessian(f, create_graph=True))(_rm([2.0, 3.0])),
            torch.func.jacrev(torch.func.hessian(f))(_tt([2.0, 3.0])),
        )

    def _ho_grad_jvp(self):
        f = lambda x: (x ** 3).sum()
        v_np = np.array([1.0, 0.0])
        rm_g = func.grad(lambda x: func.jvp(
            f, x, _rm(v_np, requires_grad=False), create_graph=True)[1])(_rm([1.0, 2.0]))
        tt_g = torch.func.grad(lambda x: torch.func.jvp(
            f, (x,), (_tt(v_np, requires_grad=False),))[1])(_tt([1.0, 2.0]))
        return _tree_close(rm_g, tt_g)

    def _ho_grad_vjp(self):
        f = lambda x: (x ** 3).sum()
        rm_g = func.grad(lambda x: func.vjp(
            f, x, _rm(1.0, requires_grad=False), create_graph=True)[1].sum())(
            _rm([1.0, 2.0]))
        tt_g = torch.func.grad(lambda x: torch.func.vjp(f, x)[1](
            _tt(1.0, requires_grad=False))[0].sum())(_tt([1.0, 2.0]))
        return _tree_close(rm_g, tt_g)

    # ---------------------------------------------------------------
    # 显式核对的 API 语义差异
    # ---------------------------------------------------------------
    def test_api_differences(self):
        _run_case("api_diff - grad_nesting_needs_create_graph",
                  lambda: self._diff_grad_nesting())
        _run_case("api_diff - jacrev_strict_default",
                  lambda: self._diff_jacrev_strict_default())
        _run_case("api_diff - hessian_multi_argnums_structure",
                  lambda: self._diff_hessian_structure())
        _run_case("api_diff - vjp_single_input_container",
                  lambda: self._diff_vjp_container())

    def _diff_grad_nesting(self):
        """差异 1：riemann 嵌套求导必须显式 create_graph=True。"""
        f = lambda x: x ** 4
        rm_x = _rm(1.5)
        with self.assertRaises(RuntimeError):
            func.grad(func.grad(f))(rm_x)

        rm_g2 = func.grad(func.grad(f, create_graph=True))(rm_x)
        tt_g2 = torch.func.grad(torch.func.grad(f))(_tt(1.5))
        return _tree_close(rm_g2, tt_g2)

    def _diff_jacrev_strict_default(self):
        """差异 2：riemann 默认 strict=True，torch 对未使用输入返回零块。"""
        f = lambda x, y: (x ** 2).sum()
        rm_x, rm_y = _rm([1.0, 2.0]), _rm([3.0, 4.0])
        tt_x, tt_y = _tt([1.0, 2.0]), _tt([3.0, 4.0])
        with self.assertRaises(RuntimeError):
            func.jacrev(f, argnums=(0, 1))(rm_x, rm_y)

        rm_res = func.jacrev(f, argnums=(0, 1), strict=False)(rm_x, rm_y)
        tt_res = torch.func.jacrev(f, argnums=(0, 1))(tt_x, tt_y)
        return (_tree_close(rm_res, tt_res)
                and _tree_close(rm_res[1], np.zeros(2)))

    def _diff_hessian_structure(self):
        """差异 3：torch 返回完整分块矩阵，riemann 只返回对角块。"""
        f = lambda x, y: (x ** 2 * y ** 2).sum()
        rm_h = func.hessian(f, argnums=(0, 1))(_rm([1.0, 2.0]), _rm([3.0, 4.0]))
        tt_h = torch.func.hessian(f, argnums=(0, 1))(_tt([1.0, 2.0]), _tt([3.0, 4.0]))

        # 结构差异：torch 2x2 块矩阵，riemann 2 个对角块
        structure_differs = (len(rm_h) == 2 and len(tt_h) == 2 and len(tt_h[0]) == 2)
        # 对角块数值一致
        diag_ok = _tree_close(rm_h[0], tt_h[0][0]) and _tree_close(rm_h[1], tt_h[1][1])
        # 交叉块可用 riemann 的 jacrev(grad) 复现（内层 grad 需 create_graph=True）
        rm_cross = func.jacrev(func.grad(f, argnums=0, create_graph=True), argnums=1)(
            _rm([1.0, 2.0]), _rm([3.0, 4.0]))
        cross_ok = (_tree_close(rm_cross, tt_h[0][1])
                    and _tree_close(rm_cross, tt_h[1][0]))
        return structure_differs and diag_ok and cross_ok

    def _diff_vjp_container(self):
        """差异 4：torch.vjp 单输入返回 1-tuple，riemann 直接返回 TN。"""
        f = lambda x: x ** 3
        rm_x, tt_x = _rm([1.0, 2.0]), _tt([1.0, 2.0])
        rm_cot = _rm([1.0, 2.0], requires_grad=False)
        tt_cot = _tt([1.0, 2.0], requires_grad=False)
        _, rm_g = func.vjp(f, rm_x, rm_cot)
        _, tt_vfn = torch.func.vjp(f, tt_x)
        tt_g = tt_vfn(tt_cot)

        container_differs = isinstance(rm_g, rm.TN) and isinstance(tt_g, tuple) and len(tt_g) == 1
        return container_differs and _tree_close(rm_g, tt_g[0])


# ---------------------------------------------------------------------------
# 主入口（独立脚本运行）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    IS_RUNNING_AS_SCRIPT = True

    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")

    print(f"{Colors.HEADER}{Colors.BOLD}"
          f"===== riemann.func vs torch.func 对照测试 ====="
          f"{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    if not TORCH_AVAILABLE:
        print(f"{Colors.FAIL}未安装 torch，无法执行对照测试{Colors.ENDC}")
        sys.exit(1)
    print(f"{Colors.OKBLUE}torch 版本: {torch.__version__}{Colors.ENDC}")
    print()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestFuncVsTorch)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    stats.print_summary()
    sys.exit(0 if result.wasSuccessful() else 1)
