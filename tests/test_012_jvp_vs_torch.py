"""
tests/test_012_jvp_vs_torch.py
==============================
riemann 算子函数的 JVP（前向自动微分）与 torch.func.jvp 的数值对照测试。

目的:
    按算子的功能类别分组织测试组，每个用例覆盖一个算子的一个 JVP 场景，
    逐项比较 riemann 与 torch 的输出值和 tangent（float64，默认 atol/rtol=1e-9）。
    torch 未提供前向 AD 规则的算子（如部分矩阵分解）自动回退到
    torch.autograd.functional.jvp 或中心差分，并相应放宽容差。

测试组:
    1. 算术与广播       add/sub/mul/div/pow/mod/neg/matmul/where/maximum/minimum
    2. 一元数学         abs/sin/cos/tan/arcsin/arccos/arctan/sinh/cosh/tanh/
                       arcsinh/arccosh/arctanh/log/exp/sign
    3. 规约与统计       sum/mean/sumall/prod/cumsum/max/min/sort/topk
    4. 形状与索引       reshape/transpose/permute/squeeze/unsqueeze/expand/clone/
                       contiguous/flip/getitem/repeat/stack/concatenate/split/tril/triu
    5. 线性代数         cross/det/inv/cholesky/svd/qr/eigh/eig
    6. nn.functional    sigmoid/softmax/log_softmax/relu/leaky_relu/prelu/rrelu/
                       elu/celu/selu/gelu/embedding
    7. 复数             mul/real/imag/conj
    8. 原地操作         add_/sub_/mul_/div_/pow_/remainder_/copy_/fill_/
                       abs_/neg_/切片赋值/切片复合赋值/原地链
    9. jvp 函数语义     嵌套、上下文栈深度、异常后清理、多输入多输出、
                       strict 参数控制

运行:
    独立脚本 → 逐用例打印 + 统计表；pytest → 仅断言。
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
from riemann.core.ad_context import active_primal_frames, forward_ad_trace_active

try:
    with warnings.catch_warnings():
        # 屏蔽导入 torch 及预热阶段的弃用告警
        warnings.simplefilter("ignore")
        import torch
        import torch.nn.functional as tF

        # 预热 torch.func 的 JVP 分解缓存，避免后续用例出现与测试无关的告警
        _warmup_x = torch.tensor([0.0])
        torch.func.jvp(lambda t: t * 2, (_warmup_x,), (_warmup_x,))
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    tF = None
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colors / StatisticsCollector / 运行控制（与 test_010/test_011 保持一致）
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

def _dtype_of(arr):
    return np.complex128 if np.iscomplexobj(arr) else np.float64


def _rm(arr, requires_grad=True):
    """构造 riemann 张量（复数自动用 complex128，其余 float64）。"""
    a = np.asarray(arr)
    return tensor(np.asarray(a, dtype=_dtype_of(a)), requires_grad=requires_grad)


def _tt(arr, requires_grad=True):
    """构造 torch 张量（复数自动用 complex128，其余 float64）。"""
    a = np.asarray(arr)
    return torch.tensor(np.asarray(a, dtype=_dtype_of(a)), requires_grad=requires_grad)


def _to_torch(arr):
    return _tt(arr, requires_grad=False)


def _to_numpy(value):
    """把 TN / torch.Tensor / tuple / list / 标量统一转换为 numpy 结构。"""
    if hasattr(value, "_array"):
        value = value._array
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.is_conj():
            t = t.resolve_conj()
        if t.is_neg():
            t = t.resolve_neg()
        return t.numpy()
    if isinstance(value, (tuple, list)):
        return tuple(_to_numpy(v) for v in value)
    return np.asarray(value)


def _tree_close(a, b, atol=1e-9, rtol=1e-9):
    """递归比较两个结果树（结构与形状、数值均需一致）。"""
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


def _combine_diff(a, b, eps):
    if isinstance(a, (tuple, list)):
        return type(a)(_combine_diff(x, y, eps) for x, y in zip(a, b))
    return (_to_numpy(a) - _to_numpy(b)) / (2.0 * eps)


def _fd_jvp(f, primals, tangents, eps=1e-6):
    """中心差分数值 JVP（torch 侧），用于无前向 AD 规则的算子。"""
    plus = tuple(_to_torch(p + eps * t) for p, t in zip(primals, tangents))
    minus = tuple(_to_torch(p - eps * t) for p, t in zip(primals, tangents))
    out_plus = f(*plus)
    out_minus = f(*minus)
    base = f(*tuple(_to_torch(p) for p in primals))
    return base, _combine_diff(out_plus, out_minus, eps)


def _torch_jvp(f, primals, tangents):
    """返回 ((out, tangent), 容差)：torch.func → autograd.functional → 差分。"""
    tp = tuple(_to_torch(p) for p in primals)
    tt = tuple(_to_torch(t) for t in tangents)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*torch\\.jit\\.script.*", category=FutureWarning)
            return torch.func.jvp(f, (tp,), (tt,)), 1e-9
    except Exception:
        pass
    try:
        out, jvp_out = torch.autograd.functional.jvp(f, tp, tt, create_graph=True)
        if isinstance(out, tuple) and len(out) == 1:
            out = out[0]
        if isinstance(jvp_out, tuple) and len(jvp_out) == 1:
            jvp_out = jvp_out[0]
        return (out, jvp_out), 1e-8
    except Exception:
        return _fd_jvp(f, primals, tangents), 1e-5


def _cmp_jvp(f_rm, primals, tangents, f_tt=None, atol=1e-9, rtol=1e-9):
    """对一组原值与 tangent 比较 riemann / torch 的 (输出, tangent)。"""
    f_tt = f_tt if f_tt is not None else f_rm
    rm_primals = tuple(_rm(p) for p in primals)
    rm_tangents = tuple(_rm(t, requires_grad=False) for t in tangents)
    rm_out, rm_tan = func.jvp(f_rm, rm_primals, rm_tangents)
    (tt_out, tt_tan), tol = _torch_jvp(f_tt, primals, tangents)
    return _tree_close((rm_out, rm_tan), (tt_out, tt_tan),
                       atol=max(atol, tol), rtol=max(rtol, tol))


def _run_case(case_name, fn):
    """执行单个测试用例：返回 bool，失败时抛 AssertionError 供 pytest 捕获。"""
    t0 = time.time()
    passed = False
    err = None
    try:
        passed = bool(fn())
    except Exception as e:
        err = str(e)

    elapsed = time.time() - t0
    stats.add_result(passed)

    if IS_RUNNING_AS_SCRIPT:
        if passed:
            print(f"  {Colors.OKGREEN}通过{Colors.ENDC}  {case_name}  ({elapsed:.4f}s)")
        else:
            print(f"  {Colors.FAIL}失败{Colors.ENDC}  {case_name}  ({elapsed:.4f}s)"
                  + (f"  — {err}" if err else ""))

    if not passed:
        raise AssertionError(f"{case_name}: {err or '与 torch.func 结果不一致'}")
    return passed


# ---------------------------------------------------------------------------
# 测试数据
# ---------------------------------------------------------------------------

A = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float64)
B = np.array([[0.5, 2.0, -1.0], [3.0, -4.0, 2.5]], dtype=np.float64)
VA = np.array([[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]], dtype=np.float64)
VB = np.array([[0.7, -0.1, 0.2], [0.3, 0.5, -0.4]], dtype=np.float64)

POS = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float64)
POS_T = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=np.float64)
SMALL = np.array([[0.2, -0.4, 0.6], [-0.8, 0.3, 0.9]], dtype=np.float64)
SMALL_T = np.array([[0.1, 0.2, -0.1], [0.3, -0.2, 0.4]], dtype=np.float64)

SQUARE = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float64)
SQUARE_T = np.array([[0.1, 0.2], [-0.3, 0.4]], dtype=np.float64)
SPD = np.array([[3.0, 0.5], [0.5, 2.0]], dtype=np.float64)
SPD_T = np.array([[0.2, -0.1], [-0.1, 0.3]], dtype=np.float64)


@unittest.skipUnless(TORCH_AVAILABLE, "未安装 torch，跳过 riemann 算子 JVP 对照测试")
class TestJVPvsTorch(unittest.TestCase):
    """riemann 算子函数 JVP 与 torch.func.jvp 的数值对照。"""

    def setUp(self):
        self.current_test_name = self._testMethodName
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")

    def tearDown(self):
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()

    # ---------------------------------------------------------------
    # 1. 算术与广播
    # ---------------------------------------------------------------
    def test_arithmetic_ops(self):
        _run_case("add - 同形", lambda: _cmp_jvp(
            lambda a, b: a + b, [A, B], [VA, VB]))
        _run_case("add - 广播", lambda: _cmp_jvp(
            lambda a, b: a + b, [A, B[0:1, :]], [VA, VB[0:1, :]]))
        _run_case("add - 常量", lambda: _cmp_jvp(
            lambda a: a + 2.0, [A], [VA]))
        _run_case("sub - 同形", lambda: _cmp_jvp(
            lambda a, b: a - b, [A, B], [VA, VB]))
        _run_case("mul - 同形", lambda: _cmp_jvp(
            lambda a, b: a * b, [A, B], [VA, VB]))
        _run_case("mul - 广播", lambda: _cmp_jvp(
            lambda a, b: a * b, [A, B[0:1, :]], [VA, VB[0:1, :]]))
        _run_case("div - 同形", lambda: _cmp_jvp(
            lambda a, b: a / b, [A, B], [VA, VB]))
        _run_case("pow - 张量指数", lambda: _cmp_jvp(
            lambda a, b: a ** b, [POS, SMALL], [POS_T, SMALL_T],
            atol=1e-8, rtol=1e-8))
        _run_case("pow - 常量指数", lambda: _cmp_jvp(
            lambda a: a ** 3.0, [A], [VA]))
        _run_case("mod - 取余", lambda: _cmp_jvp(
            lambda a, b: a % b, [A, B], [VA, VB],
            f_tt=lambda a, b: torch.remainder(a, b)))
        _run_case("neg - 取负", lambda: _cmp_jvp(
            lambda a: -a, [A], [VA]))
        _run_case("matmul - 矩阵乘", lambda: _cmp_jvp(
            lambda a, b: a @ b,
            [A, np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]])],
            [VA, np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])]))
        _run_case("where - 条件选择", lambda: self._where_case())
        _run_case("maximum - 逐元素最大", lambda: _cmp_jvp(
            rm.maximum, [A, B], [VA, VB],
            f_tt=lambda a, b: torch.maximum(a, b)))
        _run_case("minimum - 逐元素最小", lambda: _cmp_jvp(
            rm.minimum, [A, B], [VA, VB],
            f_tt=lambda a, b: torch.minimum(a, b)))
        _run_case("atan2 - 同形", lambda: _cmp_jvp(
            rm.atan2, [A, POS], [VA, POS_T],
            f_tt=lambda a, b: torch.atan2(a, b)))
        _run_case("atan2 - 广播", lambda: _cmp_jvp(
            rm.atan2, [A, POS[0:1, :]], [VA, POS_T[0:1, :]],
            f_tt=lambda a, b: torch.atan2(a, b)))
        _run_case("atan2 - TN 方法", lambda: _cmp_jvp(
            lambda a, b: a.atan2(b), [A, POS], [VA, POS_T],
            f_tt=lambda a, b: a.atan2(b)))
        _run_case("arctan2 - 别名", lambda: _cmp_jvp(
            rm.arctan2, [A, POS], [VA, POS_T],
            f_tt=lambda a, b: torch.arctan2(a, b)))

    def _where_case(self):
        cond = np.array([[True, False, True], [False, True, False]])
        return _cmp_jvp(
            lambda a, b: rm.where(tensor(cond), a, b),
            [A, B], [VA, VB],
            f_tt=lambda a, b: torch.where(torch.tensor(cond), a, b))

    # ---------------------------------------------------------------
    # 2. 一元数学
    # ---------------------------------------------------------------
    def test_unary_math_ops(self):
        cases = [
            ("abs", rm.abs, torch.abs, SMALL, SMALL_T, 1e-9),
            ("sin", rm.sin, torch.sin, A, VA, 1e-9),
            ("cos", rm.cos, torch.cos, A, VA, 1e-9),
            ("tan", rm.tan, torch.tan, np.array([[0.1, 0.3], [-0.4, 0.6]]),
             np.array([[0.2, -0.1], [0.3, 0.5]]), 1e-9),
            ("arcsin", rm.arcsin, torch.arcsin, SMALL, SMALL_T, 1e-9),
            ("asin - 别名", rm.asin, torch.asin, SMALL, SMALL_T, 1e-9),
            ("arccos", rm.arccos, torch.arccos, SMALL, SMALL_T, 1e-9),
            ("acos - 别名", rm.acos, torch.acos, SMALL, SMALL_T, 1e-9),
            ("arctan", rm.arctan, torch.arctan, A, VA, 1e-9),
            ("atan - 别名", rm.atan, torch.atan, A, VA, 1e-9),
            ("sinh", rm.sinh, torch.sinh, SMALL, SMALL_T, 1e-9),
            ("cosh", rm.cosh, torch.cosh, SMALL, SMALL_T, 1e-9),
            ("tanh", rm.tanh, torch.tanh, A, VA, 1e-9),
            ("arcsinh", rm.arcsinh, torch.arcsinh, A, VA, 1e-9),
            ("asinh - 别名", rm.asinh, torch.asinh, A, VA, 1e-9),
            ("arccosh", rm.arccosh, torch.arccosh,
             np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]), POS_T, 1e-9),
            ("acosh - 别名", rm.acosh, torch.acosh,
             np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]), POS_T, 1e-9),
            ("arctanh", rm.arctanh, torch.arctanh, SMALL, SMALL_T, 1e-9),
            ("atanh - 别名", rm.atanh, torch.atanh, SMALL, SMALL_T, 1e-9),
            ("log", rm.log, torch.log, POS, POS_T, 1e-9),
            ("exp", rm.exp, torch.exp, SMALL, SMALL_T, 1e-9),
            ("sign", rm.sign, torch.sign, A, VA, 1e-9),
        ]
        for name, rm_fn, tt_fn, x, v, tol in cases:
            _run_case(f"{name} - 逐元素", lambda rf=rm_fn, tf=tt_fn, xx=x, vv=v, t=tol:
                      _cmp_jvp(rf, [xx], [vv], f_tt=tf, atol=t, rtol=t))

    # ---------------------------------------------------------------
    # 3. 规约与统计
    # ---------------------------------------------------------------
    def test_reduction_ops(self):
        _run_case("sum - 全量", lambda: _cmp_jvp(
            lambda a: rm.sum(a), [A], [VA],
            f_tt=lambda a: torch.sum(a)))
        _run_case("sum - dim/keepdim", lambda: _cmp_jvp(
            lambda a: rm.sum(a, dim=1, keepdim=True), [A], [VA],
            f_tt=lambda a: torch.sum(a, dim=1, keepdim=True)))
        _run_case("mean - dim", lambda: _cmp_jvp(
            lambda a: rm.mean(a, dim=0), [A], [VA],
            f_tt=lambda a: torch.mean(a, dim=0)))
        _run_case("sumall - 多输入", lambda: _cmp_jvp(
            lambda a, b: rm.sumall(a, b), [A, B], [VA, VB],
            f_tt=lambda a, b: a + b))
        _run_case("prod - 全量", lambda: _cmp_jvp(
            lambda a: rm.prod(a), [POS], [SMALL_T],
            f_tt=lambda a: torch.prod(a)))
        _run_case("cumsum - dim", lambda: _cmp_jvp(
            lambda a: rm.cumsum(a, 1), [A], [VA],
            f_tt=lambda a: torch.cumsum(a, 1)))
        _run_case("max - dim", lambda: _cmp_jvp(
            lambda a: rm.max(a, dim=1).values, [A], [VA],
            f_tt=lambda a: torch.max(a, dim=1).values))
        _run_case("min - dim", lambda: _cmp_jvp(
            lambda a: rm.min(a, dim=0).values, [A], [VA],
            f_tt=lambda a: torch.min(a, dim=0).values))
        _run_case("sort - values", lambda: _cmp_jvp(
            lambda a: rm.sort(a, dim=1)[0], [A], [VA],
            f_tt=lambda a: torch.sort(a, dim=1).values))
        _run_case("topk - values", lambda: _cmp_jvp(
            lambda a: rm.topk(a, 2, dim=1)[0], [A], [VA],
            f_tt=lambda a: torch.topk(a, 2, dim=1).values))
        _run_case("logsumexp - 全量", lambda: _cmp_jvp(
            lambda a: rm.logsumexp(a), [A], [VA],
            f_tt=lambda a: torch.logsumexp(a, (0, 1))))
        _run_case("logsumexp - dim/keepdim", lambda: _cmp_jvp(
            lambda a: rm.logsumexp(a, dim=1, keepdim=True), [A], [VA],
            f_tt=lambda a: torch.logsumexp(a, 1, keepdim=True)))
        _run_case("logsumexp - 元组 dim", lambda: _cmp_jvp(
            lambda a: rm.logsumexp(a, dim=(0, 1)), [A], [VA],
            f_tt=lambda a: torch.logsumexp(a, (0, 1))))
        _run_case("logsumexp - TN 方法", lambda: _cmp_jvp(
            lambda a: a.logsumexp(dim=-1), [A], [VA],
            f_tt=lambda a: torch.logsumexp(a, -1)))

    # ---------------------------------------------------------------
    # 4. 形状变换
    # ---------------------------------------------------------------
    def test_shape_ops(self):
        _run_case("reshape", lambda: _cmp_jvp(
            lambda a: a.reshape(3, 2), [A], [VA],
            f_tt=lambda a: a.reshape(3, 2)))
        _run_case("transpose", lambda: _cmp_jvp(
            lambda a: a.transpose(0, 1), [A], [VA],
            f_tt=lambda a: a.transpose(0, 1)))
        _run_case("permute", lambda: _cmp_jvp(
            lambda a: a.reshape(2, 3, 1).permute(1, 0, 2), [A], [VA],
            f_tt=lambda a: a.reshape(2, 3, 1).permute(1, 0, 2)))
        _run_case("squeeze", lambda: _cmp_jvp(
            lambda a: a.reshape(1, 2, 3).squeeze(0), [A], [VA],
            f_tt=lambda a: a.reshape(1, 2, 3).squeeze(0)))
        _run_case("unsqueeze", lambda: _cmp_jvp(
            lambda a: a.unsqueeze(1), [A], [VA],
            f_tt=lambda a: a.unsqueeze(1)))
        _run_case("expand", lambda: _cmp_jvp(
            lambda a: a.reshape(2, 3, 1).expand(2, 3, 4), [A], [VA],
            f_tt=lambda a: a.reshape(2, 3, 1).expand(2, 3, 4)))
        _run_case("clone", lambda: _cmp_jvp(
            lambda a: a.clone(), [A], [VA],
            f_tt=lambda a: a.clone()))
        _run_case("contiguous", lambda: _cmp_jvp(
            lambda a: a.contiguous(), [A], [VA],
            f_tt=lambda a: a.contiguous()))
        _run_case("flip", lambda: _cmp_jvp(
            lambda a: rm.flip(a, (0, 1)), [A], [VA],
            f_tt=lambda a: torch.flip(a, (0, 1))))
        _run_case("repeat", lambda: _cmp_jvp(
            lambda a: rm.repeat(a, (2, 1)), [A], [VA],
            f_tt=lambda a: a.repeat(2, 1)))
        _run_case("stack", lambda: _cmp_jvp(
            lambda a, b: rm.stack((a, b), dim=0), [A, B], [VA, VB],
            f_tt=lambda a, b: torch.stack((a, b), dim=0)))
        _run_case("concatenate", lambda: _cmp_jvp(
            lambda a, b: rm.concatenate((a, b), dim=0), [A, B], [VA, VB],
            f_tt=lambda a, b: torch.cat((a, b), dim=0)))
        _run_case("split", lambda: _cmp_jvp(
            lambda a: rm.split(a, 2, 1), [A], [VA],
            f_tt=lambda a: torch.split(a, 2, 1)))
        _run_case("tril", lambda: _cmp_jvp(
            lambda a: rm.tril(a, diagonal=-1), [A], [VA],
            f_tt=lambda a: torch.tril(a, diagonal=-1)))
        _run_case("triu", lambda: _cmp_jvp(
            lambda a: rm.triu(a, diagonal=1), [A], [VA],
            f_tt=lambda a: torch.triu(a, diagonal=1)))

    def test_index_read_ops(self):
        rindex = lambda v: rm.tensor(v, dtype=np.int64)
        tindex = lambda v: torch.tensor(v, dtype=torch.int64)
        mask = np.array([[True, False, True], [False, True, True]])

        _run_case("__getitem__ - 切片", lambda: _cmp_jvp(
            lambda a: a[0:2, 1:3], [A], [VA],
            f_tt=lambda a: a[0:2, 1:3]))
        _run_case("__getitem__ - 高级索引/重复读取", lambda: _cmp_jvp(
            lambda a: a[rindex([0, 1, 0]), rindex([2, 0, 1])], [A], [VA],
            f_tt=lambda a: a[tindex([0, 1, 0]), tindex([2, 0, 1])]))
        _run_case("index_select - 重复索引", lambda: _cmp_jvp(
            lambda a: rm.index_select(a, 0, rindex([1, 0, 1])), [A], [VA],
            f_tt=lambda a: torch.index_select(a, 0, tindex([1, 0, 1]))))
        _run_case("masked_select", lambda: _cmp_jvp(
            lambda a: rm.masked_select(a, rm.tensor(mask)), [A], [VA],
            f_tt=lambda a: torch.masked_select(a, torch.tensor(mask))))
        _run_case("take", lambda: _cmp_jvp(
            lambda a: rm.take(a, rindex([5, 0, 4])), [A], [VA],
            f_tt=lambda a: torch.take(a, tindex([5, 0, 4]))))
        _run_case("gather - 重复读取", lambda: _cmp_jvp(
            lambda a: rm.gather(a, 1, rindex([[2, 0], [1, 1]])), [A], [VA],
            f_tt=lambda a: torch.gather(a, 1, tindex([[2, 0], [1, 1]]))))

    def test_non_inplace_index_write_ops(self):
        rindex = lambda v: rm.tensor(v, dtype=np.int64)
        tindex = lambda v: torch.tensor(v, dtype=torch.int64)
        values = np.array([0.5, -1.0, 2.0])
        values_t = np.array([0.2, 0.4, -0.6])
        copy_src = np.array([[0.2, -0.4], [0.6, -0.8]])
        copy_src_t = np.array([[0.1, 0.3], [-0.2, 0.5]])
        add_src = np.arange(1.0, 10.0).reshape(3, 3) / 10.0
        add_src_t = np.linspace(-0.3, 0.3, 9).reshape(3, 3)

        _run_case("index_put - 覆盖", lambda: _cmp_jvp(
            lambda a, v: rm.index_put(
                a, (rindex([0, 1, 0]), rindex([2, 0, 1])), v),
            [A, values], [VA, values_t],
            f_tt=lambda a, v: torch.index_put(
                a, (tindex([0, 1, 0]), tindex([2, 0, 1])), v)))
        _run_case("index_put - 重复索引累加", lambda: _cmp_jvp(
            lambda a, v: rm.index_put(
                a, (rindex([0, 0, 1]), rindex([1, 1, 2])), v, accumulate=True),
            [A, values], [VA, values_t],
            f_tt=lambda a, v: torch.index_put(
                a, (tindex([0, 0, 1]), tindex([1, 1, 2])), v, accumulate=True)))
        _run_case("index_copy", lambda: _cmp_jvp(
            lambda a, s: rm.index_copy(a, 1, rindex([2, 0]), s),
            [A, copy_src], [VA, copy_src_t],
            f_tt=lambda a, s: torch.index_copy(a, 1, tindex([2, 0]), s)))
        _run_case("index_add - alpha/重复索引", lambda: _cmp_jvp(
            lambda a, s: rm.index_add(a, 0, rindex([1, 0, 1]), s, alpha=-0.5),
            [A, add_src], [VA, add_src_t],
            f_tt=lambda a, s: torch.index_add(
                a, 0, tindex([1, 0, 1]), s, alpha=-0.5)))
        _run_case("index_fill - 0维张量值", lambda: _cmp_jvp(
            lambda a, v: rm.index_fill(a, 1, rindex([0, -1]), v),
            [A, np.array(0.75)], [VA, np.array(-0.2)],
            f_tt=lambda a, v: torch.index_fill(a, 1, tindex([0, -1]), v)))
        _run_case("put - 覆盖", lambda: _cmp_jvp(
            lambda a, s: rm.put(a, rindex([4, 0, 2]), s),
            [A, values.reshape(1, 3)], [VA, values_t.reshape(1, 3)],
            f_tt=lambda a, s: torch.put(a, tindex([4, 0, 2]), s)))
        _run_case("put - 重复索引累加", lambda: _cmp_jvp(
            lambda a, s: rm.put(a, rindex([5, 1, 5]), s, accumulate=True),
            [A, values.reshape(1, 3)], [VA, values_t.reshape(1, 3)],
            f_tt=lambda a, s: torch.put(a, tindex([5, 1, 5]), s, accumulate=True)))
        _run_case("select_scatter - 负索引", lambda: _cmp_jvp(
            lambda a, s: rm.select_scatter(a, s, 1, -1),
            [A, np.array([0.7, -0.8])], [VA, np.array([0.2, 0.5])],
            f_tt=lambda a, s: torch.select_scatter(a, s, 1, -1)))
        _run_case("slice_scatter", lambda: _cmp_jvp(
            lambda a, s: rm.slice_scatter(a, s, 1, 0, 3, 2),
            [A, copy_src], [VA, copy_src_t],
            f_tt=lambda a, s: torch.slice_scatter(a, s, 1, 0, 3, 2)))
        _run_case("scatter - src", lambda: _cmp_jvp(
            lambda a, s: rm.scatter(a, 1, rindex([[2, 0], [1, 2]]), s),
            [A, copy_src], [VA, copy_src_t],
            f_tt=lambda a, s: torch.scatter(a, 1, tindex([[2, 0], [1, 2]]), s)))
        _run_case("scatter_add - 重复索引", lambda: _cmp_jvp(
            lambda a, s: rm.scatter_add(a, 1, rindex([[1, 1], [2, 0]]), s),
            [A, copy_src], [VA, copy_src_t],
            f_tt=lambda a, s: torch.scatter_add(a, 1, tindex([[1, 1], [2, 0]]), s)))

        for reduce in ("sum", "prod", "mean", "amax", "amin"):
            for include_self in (True, False):
                _run_case(
                    f"scatter_reduce - {reduce}/include_self={include_self}",
                    lambda reduce=reduce, include_self=include_self: _cmp_jvp(
                        lambda a, s: rm.scatter_reduce(
                            a, 1, rindex([[1, 1], [2, 0]]), s, reduce,
                            include_self=include_self),
                        [A, copy_src], [VA, copy_src_t],
                        f_tt=lambda a, s: torch.scatter_reduce(
                            a, 1, tindex([[1, 1], [2, 0]]), s, reduce,
                            include_self=include_self)))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"index_reduce\(\) is in beta.*",
                category=UserWarning)
            for reduce in ("prod", "mean", "amax", "amin"):
                for include_self in (True, False):
                    _run_case(
                        f"index_reduce - {reduce}/include_self={include_self}",
                        lambda reduce=reduce, include_self=include_self: _cmp_jvp(
                            lambda a, s: rm.index_reduce(
                                a, 0, rindex([1, 0, 1]), s, reduce,
                                include_self=include_self),
                            [A, add_src], [VA, add_src_t],
                            f_tt=lambda a, s: torch.index_reduce(
                                a, 0, tindex([1, 0, 1]), s, reduce,
                                include_self=include_self)))

        mask = np.array([[True, False, True], [False, True, True]])
        mask_input = np.array([1.0, -2.0, 3.0])
        mask_input_t = np.array([0.1, 0.2, 0.3])
        mask_src = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        mask_src_t = np.array([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
        _run_case("masked_scatter - 双向广播/多余来源", lambda: _cmp_jvp(
            lambda a, s: rm.masked_scatter(a, rm.tensor(mask), s),
            [mask_input, mask_src], [mask_input_t, mask_src_t],
            f_tt=lambda a, s: torch.masked_scatter(a, torch.tensor(mask), s)))
        _run_case("masked_fill - mask广播/0维值", lambda: _cmp_jvp(
            lambda a, v: rm.masked_fill(a, rm.tensor(mask[:1]), v),
            [A, np.array(0.75)], [VA, np.array(-0.2)],
            f_tt=lambda a, v: torch.masked_fill(a, torch.tensor(mask[:1]), v)))

    # ---------------------------------------------------------------
    # 5. 线性代数
    # ---------------------------------------------------------------
    def test_linalg_ops(self):
        _run_case("cross", lambda: _cmp_jvp(
            lambda a, b: rm.linalg.cross(a, b, dim=1),
            [A, B], [VA, VB],
            f_tt=lambda a, b: torch.linalg.cross(a, b, dim=1)))
        _run_case("det", lambda: _cmp_jvp(
            lambda a: rm.linalg.det(a), [SQUARE], [SQUARE_T],
            f_tt=lambda a: torch.linalg.det(a)))
        _run_case("inv", lambda: _cmp_jvp(
            lambda a: rm.linalg.inv(a), [SQUARE], [SQUARE_T],
            f_tt=lambda a: torch.linalg.inv(a)))
        _run_case("cholesky", lambda: _cmp_jvp(
            lambda a: rm.linalg.cholesky(a), [SPD], [SPD_T],
            f_tt=lambda a: torch.linalg.cholesky(a)))
        _run_case("svd - 奇异值", lambda: _cmp_jvp(
            lambda a: rm.linalg.svd(a, full_matrices=False)[1],
            [SQUARE], [SQUARE_T],
            f_tt=lambda a: torch.linalg.svd(a, full_matrices=False).S))
        _run_case("svd - 重建 A_t", lambda: self._svd_reconstruct())
        _run_case("qr - 重建 A_t", lambda: self._qr_reconstruct())
        _run_case("eigh - 特征值", lambda: _cmp_jvp(
            lambda a: rm.linalg.eigh(a)[0], [SPD], [SPD_T],
            f_tt=lambda a: torch.linalg.eigh(a).eigenvalues))
        _run_case("eig - 特征值实部和", lambda: _cmp_jvp(
            lambda a: rm.linalg.eig(a)[0].real.sum(), [SQUARE], [SQUARE_T],
            f_tt=lambda a: torch.linalg.eig(a).eigenvalues.real.sum()))
        _run_case("lu - L (方阵)", lambda: _cmp_jvp(
            lambda a: rm.linalg.lu(a)[1], [SQUARE], [SQUARE_T],
            f_tt=lambda a: torch.linalg.lu(a)[1]))
        _run_case("lu - U (方阵)", lambda: _cmp_jvp(
            lambda a: rm.linalg.lu(a)[2], [SQUARE], [SQUARE_T],
            f_tt=lambda a: torch.linalg.lu(a)[2]))
        _run_case("lu - L (宽矩阵)", lambda: _cmp_jvp(
            lambda a: rm.linalg.lu(a)[1], [A], [VA],
            f_tt=lambda a: torch.linalg.lu(a)[1]))
        _run_case("lu - U (宽矩阵)", lambda: _cmp_jvp(
            lambda a: rm.linalg.lu(a)[2], [A], [VA],
            f_tt=lambda a: torch.linalg.lu(a)[2]))
        _run_case("lu - L (高矩阵)", lambda: _cmp_jvp(
            lambda a: rm.linalg.lu(a)[1], [A.T], [VA.T],
            f_tt=lambda a: torch.linalg.lu(a)[1]))
        _run_case("lu - U (高矩阵)", lambda: _cmp_jvp(
            lambda a: rm.linalg.lu(a)[2], [A.T], [VA.T],
            f_tt=lambda a: torch.linalg.lu(a)[2]))
        _run_case("lu - 重建 dA (pivot=False)", lambda: self._lu_reconstruct())

    def _svd_reconstruct(self):
        f = lambda a: rm.linalg.svd(a, full_matrices=False)
        (u, s, vh), (ut, st, vht) = func.jvp(
            f, (_rm(SQUARE),), (_rm(SQUARE_T, requires_grad=False),))
        rec = (ut.array() @ np.diag(s.array()) @ vh.array()
               + u.array() @ np.diag(st.array()) @ vh.array()
               + u.array() @ np.diag(s.array()) @ vht.array())
        return np.allclose(rec, SQUARE_T, atol=1e-8)

    def _qr_reconstruct(self):
        f = lambda a: rm.linalg.qr(a, mode="reduced")
        (q, r), (qt, rt) = func.jvp(
            f, (_rm(SQUARE),), (_rm(SQUARE_T, requires_grad=False),))
        rec = qt.array() @ r.array() + q.array() @ rt.array()
        return np.allclose(rec, SQUARE_T, atol=1e-8)

    def _lu_reconstruct(self):
        # pivot=False 时 A = L·U，切线满足 dA = dL·U + L·dU
        f = lambda a: rm.linalg.lu(a, pivot=False)
        _, (_, lt, ut) = func.jvp(
            f, (_rm(SQUARE),), (_rm(SQUARE_T, requires_grad=False),))
        _, l, u = f(_rm(SQUARE))
        rec = lt.array() @ u.array() + l.array() @ ut.array()
        return np.allclose(rec, SQUARE_T, atol=1e-8)

    # ---------------------------------------------------------------
    # 6. nn.functional
    # ---------------------------------------------------------------
    def test_nn_functional_ops(self):
        _run_case("sigmoid", lambda: _cmp_jvp(
            rm.nn.functional.sigmoid, [A], [VA],
            f_tt=torch.sigmoid))
        _run_case("softmax", lambda: _cmp_jvp(
            lambda a: rm.nn.functional.softmax(a, dim=1), [A], [VA],
            f_tt=lambda a: torch.softmax(a, dim=1)))
        _run_case("log_softmax", lambda: _cmp_jvp(
            lambda a: rm.nn.functional.log_softmax(a, dim=1), [A], [VA],
            f_tt=lambda a: torch.log_softmax(a, dim=1)))
        _run_case("relu", lambda: _cmp_jvp(
            rm.nn.functional.relu, [A], [VA],
            f_tt=torch.relu))
        _run_case("leaky_relu", lambda: _cmp_jvp(
            lambda a: rm.nn.functional.leaky_relu(a, 0.1), [A], [VA],
            f_tt=lambda a: tF.leaky_relu(a, 0.1)))
        _run_case("prelu", lambda: _cmp_jvp(
            lambda a, w: rm.nn.functional.prelu(a, w),
            [A, np.array([0.1, 0.2, 0.3])],
            [VA, np.array([0.5, -0.4, 0.3])],
            f_tt=lambda a, w: tF.prelu(a, w)))
        _run_case("rrelu - eval", lambda: _cmp_jvp(
            lambda a: rm.nn.functional.rrelu(a, training=False), [A], [VA],
            f_tt=lambda a: tF.rrelu(a, training=False)))
        _run_case("elu", lambda: _cmp_jvp(
            lambda a: rm.nn.functional.elu(a, 0.5), [A], [VA],
            f_tt=lambda a: tF.elu(a, 0.5)))
        _run_case("celu", lambda: _cmp_jvp(
            lambda a: rm.nn.functional.celu(a, 1.0), [A], [VA],
            f_tt=lambda a: tF.celu(a, 1.0)))
        _run_case("selu", lambda: _cmp_jvp(
            rm.nn.functional.selu, [A], [VA],
            f_tt=tF.selu))
        _run_case("gelu", lambda: _cmp_jvp(
            rm.nn.functional.gelu, [A], [VA],
            f_tt=lambda a: tF.gelu(a, approximate="tanh")))
        _run_case("tanh - nn", lambda: _cmp_jvp(
            rm.nn.functional.tanh, [A], [VA],
            f_tt=torch.tanh))
        _run_case("silu", lambda: _cmp_jvp(
            rm.nn.functional.silu, [A], [VA],
            f_tt=tF.silu))
        _run_case("softplus", lambda: _cmp_jvp(
            rm.nn.functional.softplus, [A], [VA],
            f_tt=tF.softplus))
        _run_case("embedding", lambda: self._embedding_case())

    def _embedding_case(self):
        idx = np.array([0, 2, 1, 2], dtype=np.int64)
        w = np.array([[0.5, -1.0], [2.0, 0.3], [-0.7, 1.5]], dtype=np.float64)
        vw = np.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6]], dtype=np.float64)
        rm_idx = tensor(idx)
        tt_idx = torch.tensor(idx)
        rm_out, rm_tan = func.jvp(
            lambda weight: rm.nn.functional.embedding(rm_idx, weight),
            (_rm(w),), (_rm(vw, requires_grad=False),))
        try:
            tt_out, tt_tan = torch.func.jvp(
                lambda weight: tF.embedding(tt_idx, weight),
                (_tt(w),), (_tt(vw, requires_grad=False),))
            tol = 1e-9
        except Exception:
            tt_out, tt_tan = (tF.embedding(tt_idx, _tt(w)),
                              tF.embedding(tt_idx, _tt(vw, requires_grad=False)))
            tol = 1e-9
        return _tree_close((rm_out, rm_tan), (tt_out, tt_tan), atol=tol, rtol=tol)

    # ---------------------------------------------------------------
    # 7. 复数
    # ---------------------------------------------------------------
    def test_complex_ops(self):
        z = np.array([[1.0 + 2.0j, -0.5 + 0.3j],
                      [0.7 - 1.2j, 2.5 + 0.1j]])
        vz = np.array([[0.1 - 0.2j, 0.3 + 0.4j],
                       [-0.5 + 0.6j, 0.7 - 0.8j]])
        _run_case("complex mul", lambda: _cmp_jvp(
            lambda a, b: a * b, [z, z.conj()], [vz, vz],
            atol=1e-9, rtol=1e-9))
        _run_case("real", lambda: _cmp_jvp(
            lambda a: a.real, [z], [vz],
            f_tt=lambda a: a.real))
        _run_case("imag", lambda: _cmp_jvp(
            lambda a: a.imag, [z], [vz],
            f_tt=lambda a: a.imag))
        _run_case("conj", lambda: _cmp_jvp(
            rm.conj, [z], [vz],
            f_tt=torch.conj))
        _run_case("logsumexp - 复数", lambda: _cmp_jvp(
            lambda a: rm.logsumexp(a, dim=1), [z], [vz],
            f_tt=lambda a: torch.logsumexp(a, 1)))
        _run_case("asin - 复数", lambda: _cmp_jvp(
            rm.asin, [z / 3.0], [vz / 3.0],
            f_tt=torch.asin))

    # ---------------------------------------------------------------
    # 8. 原地写操作
    # ---------------------------------------------------------------
    def test_inplace_write_ops(self):
        rindex = lambda v: rm.tensor(v, dtype=np.int64)
        tindex = lambda v: torch.tensor(v, dtype=torch.int64)
        values = np.array([0.5, -1.0, 2.0])
        values_t = np.array([0.2, 0.4, -0.6])
        mask = np.array([[True, False, True]])

        _run_case("add_ - 张量", lambda: _cmp_jvp(
            lambda a, b: a.clone().add_(b), [A, B], [VA, VB]))
        _run_case("add_ - 常量", lambda: _cmp_jvp(
            lambda a: a.clone().add_(2.0), [A], [VA]))
        _run_case("sub_ - 张量", lambda: _cmp_jvp(
            lambda a, b: a.clone().sub_(b), [A, B], [VA, VB]))
        _run_case("mul_ - 张量", lambda: _cmp_jvp(
            lambda a, b: a.clone().mul_(b), [A, B], [VA, VB]))
        _run_case("mul_ - 常量", lambda: _cmp_jvp(
            lambda a: a.clone().mul_(3.0), [A], [VA]))
        _run_case("div_ - 张量", lambda: _cmp_jvp(
            lambda a, b: a.clone().div_(b), [A, B], [VA, VB]))
        _run_case("pow_ - 常量指数", lambda: _cmp_jvp(
            lambda a: a.clone().pow_(3.0), [POS], [POS_T]))
        _run_case("remainder_ - 张量", lambda: _cmp_jvp(
            lambda a, b: a.clone().remainder_(b), [A, B], [VA, VB]))
        _run_case("copy_ - 张量覆盖", lambda: _cmp_jvp(
            lambda a, b: a.clone().copy_(b), [A, B], [VA, VB]))
        _run_case("fill_ - 常量填充", lambda: _cmp_jvp(
            lambda a: a.clone().fill_(2.0), [A], [VA]))
        _run_case("abs_ - 取绝对值", lambda: _cmp_jvp(
            lambda a: a.clone().abs_(), [A], [VA]))
        _run_case("neg_ - 取负", lambda: _cmp_jvp(
            lambda a: a.clone().neg_(), [A], [VA]))
        _run_case("masked_fill_ - mask广播/0维值", lambda: _cmp_jvp(
            lambda a, v: a.clone().masked_fill_(rm.tensor(mask), v),
            [A, np.array(0.75)], [VA, np.array(-0.2)],
            f_tt=lambda a, v: a.clone().masked_fill_(torch.tensor(mask), v)))
        _run_case("index_put_ - 覆盖", lambda: _cmp_jvp(
            lambda a, v: a.clone().index_put_(
                (rindex([0, 1, 0]), rindex([2, 0, 1])), v),
            [A, values], [VA, values_t],
            f_tt=lambda a, v: a.clone().index_put_(
                (tindex([0, 1, 0]), tindex([2, 0, 1])), v)))
        _run_case("index_put_ - 重复索引累加", lambda: _cmp_jvp(
            lambda a, v: a.clone().index_put_(
                (rindex([0, 0, 1]), rindex([1, 1, 2])), v, accumulate=True),
            [A, values], [VA, values_t],
            f_tt=lambda a, v: a.clone().index_put_(
                (tindex([0, 0, 1]), tindex([1, 1, 2])), v, accumulate=True)))
        _run_case("setitem - 切片赋值", lambda: _cmp_jvp(
            lambda a, b: self._setitem(a, b, compound=False),
            [A, np.array([[1.0, 2.0], [3.0, 4.0]])],
            [VA, np.array([[0.1, 0.2], [0.3, 0.4]])]))
        _run_case("setitem - 切片复合赋值", lambda: _cmp_jvp(
            lambda a, b: self._setitem(a, b, compound=True),
            [A, np.array([[1.0, 2.0], [3.0, 4.0]])],
            [VA, np.array([[0.1, 0.2], [0.3, 0.4]])]))
        _run_case("原地链", lambda: _cmp_jvp(
            lambda a, b: self._inplace_chain(a, b), [A, B], [VA, VB]))
        _run_case("中间节点原地", lambda: _cmp_jvp(
            lambda a, b: self._inplace_intermediate(a, b), [A, B], [VA, VB]))

    @staticmethod
    def _setitem(a, b, compound):
        z = a.clone()
        if compound:
            z[0:2, 0:2] += b
        else:
            z[0:2, 0:2] = b
        return z

    @staticmethod
    def _inplace_chain(a, b):
        z = a.clone()
        z.add_(b)
        z.mul_(2.0)
        return z

    @staticmethod
    def _inplace_intermediate(a, b):
        z = 2.0 * a
        z.add_(b)
        return z

    # ---------------------------------------------------------------
    # 9. jvp 函数自身语义（不针对单个算子）
    # ---------------------------------------------------------------
    def test_jvp_function_semantics(self):
        _run_case("嵌套 jvp", lambda: self._nested_jvp())
        _run_case("上下文栈深度", lambda: self._context_depth())
        _run_case("异常后上下文恢复", lambda: self._context_after_exception())
        _run_case("多输入多输出", lambda: self._multi_input_output())
        _run_case("连续调用独立", lambda: self._consecutive_calls())
        _run_case("上下文回调外关闭", lambda: self._inactive_outside())
        _run_case("strict 控制", lambda: self._strict_control())

    def _nested_jvp(self):
        inner = lambda x: (x ** 3).sum()
        outer_rm = lambda x: func.jvp(inner, (x,), ((x * 0.5,)))[1]
        outer_tt = lambda x: torch.func.jvp(inner, (x,), (x * 0.5,))[1]
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.3, -0.2, 0.5])
        rm_out, rm_tan = func.jvp(outer_rm, (_rm(x),), (_rm(v, requires_grad=False),))
        tt_out, tt_tan = torch.func.jvp(
            outer_tt, (_tt(x),), (_tt(v, requires_grad=False),))
        return _tree_close((rm_out, rm_tan), (tt_out, tt_tan),
                           atol=1e-8, rtol=1e-8)

    def _context_depth(self):
        depth = {}

        def inner(x):
            depth["inner"] = len(active_primal_frames())
            return (x ** 2).sum()

        def outer(x):
            depth["outer"] = len(active_primal_frames())
            return func.jvp(inner, (x,), (x * 0.5,))[1]

        func.jvp(outer, (_rm([1.0, 2.0]),), (_rm([0.3, -0.2], requires_grad=False),))
        return (depth.get("outer") == 1 and depth.get("inner") == 2
                and not forward_ad_trace_active())

    def _context_after_exception(self):
        def boom(x):
            raise ValueError("boom")

        try:
            func.jvp(boom, (_rm([1.0, 2.0]),), (_rm([0.3, 0.2], requires_grad=False),))
        except ValueError:
            pass
        return not forward_ad_trace_active()

    def _multi_input_output(self):
        f_rm = lambda a, b: (a * b, a + b, (a ** 2).sum())
        x, y = A, B
        vx, vy = VA, VB
        rm_out, rm_tan = func.jvp(f_rm, (_rm(x), _rm(y)),
                                  (_rm(vx, False), _rm(vy, False)))
        tt_out, tt_tan = torch.func.jvp(f_rm, (_tt(x), _tt(y)),
                                        (_tt(vx, False), _tt(vy, False)))
        return _tree_close((rm_out, rm_tan), (tt_out, tt_tan))

    def _consecutive_calls(self):
        f = lambda x: (x ** 3).sum()
        x = _rm([1.0, 2.0, 3.0])
        _, t1 = func.jvp(f, (x,), (_rm([1.0, 0.0, -1.0], requires_grad=False),))
        _, t2 = func.jvp(f, (x,), (_rm([0.5, 0.5, 0.5], requires_grad=False),))
        return (np.allclose(t1.array(), -24.0)
                and np.allclose(t2.array(), 21.0))

    def _inactive_outside(self):
        before = forward_ad_trace_active()
        func.jvp(lambda x: (x ** 2).sum(), (_rm([1.0, 2.0]),),
                 (_rm([0.1, 0.2], requires_grad=False),))
        return (not before) and (not forward_ad_trace_active())

    def _strict_control(self):
        """strict 控制“某个输出与某个输入完全不连通”时的处理方式。

        行为逻辑（与 torch.func.jvp 一致）：
          - strict=False：按数学语义处理，不连通即 J·v 的对应贡献为 0，
            该输出返回同形零 tangent；
          - strict=True：视为调用方假设“输出必须依赖全部输入”，一旦不连通
            立即抛 RuntimeError（错误信息含 "independent of input"）；
          - 输出正常依赖输入时 strict=True 不做任何额外限制，结果应与
            strict=False / torch 完全相同。
        """
        # 常量输出：与输入 x 完全不连通，用于触发“独立性”判定。
        const = lambda x: _rm([3.0], requires_grad=False)
        x = _rm([1.0, 2.0])
        v = _rm([0.3, 0.2], requires_grad=False)

        # ① strict=False：返回零 tangent，数值与 torch 一致。
        rm_res = func.jvp(const, (x,), (v,), strict=False)
        tt_res = torch.func.jvp(
            lambda t: torch.tensor([3.0], dtype=torch.float64),
            (_tt([1.0, 2.0]),), (_tt([0.3, 0.2]),), strict=False)
        if not _tree_close(rm_res, tt_res):
            return False

        # ② strict=True + 不连通输出：必须报错，且错误信息可区分于普通异常。
        try:
            func.jvp(const, (x,), (v,), strict=True)
        except RuntimeError as exc:
            if "independent of input" not in str(exc):
                return False
        else:
            return False

        # ③ strict=True + 连通输出：正常返回，结果与 torch 一致。
        f = lambda t: (t ** 2).sum()
        rm_res = func.jvp(f, (x,), (v,), strict=True)
        tt_res = torch.func.jvp(f, (_tt([1.0, 2.0]),), (_tt([0.3, 0.2]),),
                                strict=True)
        return _tree_close(rm_res, tt_res)


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
          f"===== riemann 算子 JVP vs torch.func.jvp 对照测试 ====="
          f"{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    if not TORCH_AVAILABLE:
        print(f"{Colors.FAIL}未安装 torch，无法执行对照测试{Colors.ENDC}")
        sys.exit(1)
    print(f"{Colors.OKBLUE}torch 版本: {torch.__version__}{Colors.ENDC}")
    print()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestJVPvsTorch)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    stats.print_summary()
    sys.exit(0 if result.wasSuccessful() else 1)
