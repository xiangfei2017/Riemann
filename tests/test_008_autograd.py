"""
tests/test_008_autograd.py
==========================
riemann 自动微分接口统一测试（由 test_010 ~ test_014 合并而来）。

用例组（每个 test_xxx 方法 = 一个用例组）:
    grad        riemann.autograd.grad
    d           Tensor.d（标量张量的高阶 / 混合偏导）
    derivative  riemann.autograd.functional.derivative
    jacobian    riemann.autograd.functional.jacobian
    hessian     riemann.autograd.functional.hessian
    jvp         Jacobian-vector product
    vjp         vector-Jacobian product
    hvp         Hessian-vector product
    vhp         vector-Hessian product（与 hvp 数值一致）

风格与 test_010_func_grad.py 一致：
    - 数据驱动：每个用例组由模块级用例表描述（func / inputs / expected）
    - 独立脚本运行 → 逐用例打印 + 按用例组汇总统计表
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
    from riemann import tensor
    from riemann.autograd.grad import grad as rm_grad
    from riemann.autograd.functional import (
        derivative, jacobian, hessian, jvp, vjp, hvp, vhp,
    )
except ImportError as e:
    print(f"无法导入 riemann 模块: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Colors / StatisticsCollector / IS_RUNNING_AS_SCRIPT（与 test_010 一致）
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

    def add_result(self, case_name, passed):
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
        headers = ['用例组', '通过/总数', '通过率', '耗时(秒)']
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
        print("\n各用例组测试详情:")
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
# 通用辅助（张量构造 / 结果比较 / 用例执行）
# ---------------------------------------------------------------------------

def _t(value, requires_grad=True):
    """由 list / float / ndarray 构造 float64 的 TN 张量。"""
    return tensor(np.asarray(value, dtype=np.float64), requires_grad=requires_grad)


def _inputs(values):
    """批量构造用例输入张量。"""
    return [_t(v) for v in values]


def _np(value):
    """TN / ndarray / 标量 → numpy 数组。"""
    return np.asarray(value._array if hasattr(value, '_array') else value)


def _vector(spec):
    """把用例表中的向量规格转换成 v 参数。

    tuple 表示「按输入 / 输出逐个给出」的列表，其余形式（list / float / ndarray）
    表示单个向量。两者形状不一致时由被测函数自行报错。
    """
    if isinstance(spec, tuple):
        return tuple(_t(x, requires_grad=False) for x in spec)
    return _t(spec, requires_grad=False)


def _assert_tree(actual, expected, name):
    """递归比较嵌套结果（tuple / None / 张量）与期望值的结构与数值。"""
    if expected is None:
        assert actual is None, f"{name}: 期望 None，实际 {type(actual).__name__}"
        return
    if isinstance(expected, tuple):
        assert isinstance(actual, (tuple, list)), \
            f"{name}: 期望 tuple，实际 {type(actual).__name__}"
        assert len(actual) == len(expected), \
            f"{name}: 长度 {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_tree(a, e, f"{name}[{i}]")
        return
    a = _np(actual)
    e = np.asarray(expected, dtype=np.float64)
    assert a.shape == e.shape, f"{name}: 形状 {a.shape} != {e.shape}"
    np.testing.assert_allclose(a, e, atol=1e-8, rtol=1e-8, err_msg=name)


def _assert_requires_grad(value, name):
    """递归断言结果中的所有张量都可求导（create_graph=True 场景）。"""
    if isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            _assert_requires_grad(v, f"{name}[{i}]")
        return
    assert value.requires_grad, f"{name}: 结果未保留计算图"


def _assert_requires_grad_flags(value, flags, name):
    """按逐个张量给出的期望标志断言 requires_grad（常量块不保留计算图）。"""
    for i, (v, flag) in enumerate(zip(value, flags)):
        assert v.requires_grad == flag, \
            f"{name}[{i}]: requires_grad 应为 {flag}，实际 {v.requires_grad}"


def _start_group(name):
    """开始一个用例组的统计；独立脚本模式下打印组标题。"""
    stats.start_function(name)
    if IS_RUNNING_AS_SCRIPT:
        print(f"\n{Colors.OKBLUE}{Colors.BOLD}【{name}】{Colors.ENDC}")


def _end_group():
    """结束当前用例组统计。"""
    stats.end_function()


def _run_case(case_name, fn, expect_error=False, match=None):
    """执行单个子用例并记录统计；独立脚本模式下打印状态。"""
    t0 = time.time()
    err = None
    try:
        fn()
        passed = not expect_error
        if expect_error:
            err = f"未抛出预期异常（期望包含 '{match}'）"
    except Exception as e:
        if expect_error and (match is None or match in str(e)):
            passed = True
        else:
            passed = False
            err = f"{type(e).__name__}: {e}"

    stats.add_result(case_name, passed)

    if IS_RUNNING_AS_SCRIPT:
        used = time.time() - t0
        if passed:
            print(f"  {Colors.OKGREEN}通过{Colors.ENDC}  {case_name}  ({used:.4f}s)")
        else:
            print(f"  {Colors.FAIL}失败{Colors.ENDC}  {case_name}  ({used:.4f}s)  — {err}")

    if not passed:
        raise AssertionError(f"子用例 '{case_name}' 失败: {err}")


# ---------------------------------------------------------------------------
# 用例表：grad
# ---------------------------------------------------------------------------

GRAD_CASES = [
    # 基础：标量输入 / 标量函数
    dict(name='scalar_power', func=lambda x: x ** 2, inputs=[2.0], expected=(4.0,)),
    dict(name='scalar_poly', func=lambda x: x ** 3 + 2 * x, inputs=[3.0], expected=(29.0,)),
    # 多输入
    dict(name='multi_input', func=lambda x, y: 2 * x + y ** 2,
         inputs=[3.0, 4.0], expected=(2.0, 8.0)),
    # 三角函数参与（验证非多项式算子）
    dict(name='trig_mixed', func=lambda x, y: x * y + rm.sin(x),
         inputs=[np.pi, 2.0], expected=(2.0 + np.cos(np.pi), np.pi)),
    # allow_unused：第三个输入不参与计算
    dict(name='allow_unused', func=lambda x, y, z: 2 * x + y ** 2,
         inputs=[3.0, 4.0, 5.0], allow_unused=True, expected=(2.0, 8.0, None)),
    # 张量输入 + 显式 grad_outputs
    dict(name='grad_outputs', func=lambda x: x ** 2, inputs=[[1.0, 2.0, 3.0]],
         grad_outputs=[1.0, 1.0, 1.0], expected=([2.0, 4.0, 6.0],)),
    # 高阶导数
    dict(name='second_order', func=lambda x: x ** 3, inputs=[2.0], order=2, expected=(12.0,)),
    dict(name='third_order', func=lambda x: x ** 4, inputs=[2.0], order=3, expected=(48.0,)),
    # 异常场景
    dict(name='err_input_not_requires_grad', expect_error=True, match='requires grad',
         action=lambda: rm_grad((_t([1.0, 2.0]) ** 2).sum(),
                                [_t([1.0, 2.0], requires_grad=False)])),
    dict(name='err_output_not_scalar', expect_error=True, match='Only a scalar',
         action=lambda: rm_grad(_t([1.0, 2.0]), [_t([1.0, 2.0])])),
    dict(name='err_grad_outputs_shape', expect_error=True, match='shape of grad_outputs',
         action=lambda: rm_grad(_t([1.0, 2.0]) ** 2, [_t([1.0, 2.0])],
                                grad_outputs=_t([1.0]))),
]


# ---------------------------------------------------------------------------
# 用例表：Tensor.d
# 每个 spec 为 (输入下标, 求导阶数, 是否用 tuple 形式)；按顺序串联即为混合偏导
# ---------------------------------------------------------------------------

_D_THIRD = np.zeros((3, 3, 3))
for _i in range(3):
    _D_THIRD[_i, _i, _i] = 24 * np.array([1.0, 2.0, 3.0])[_i]

D_CASES = [
    # 兼容单变量写法 f.d(x)
    dict(name='compat_single_var', func=lambda x, y: x ** 3 * y ** 2,
         inputs=[2.0, 3.0], specs=[(0, 1, False)], expected=108.0),
    # 兼容混合偏导写法 f.d(x, y)
    dict(name='compat_mixed_vars', func=lambda x, y: x ** 3 * y ** 2,
         inputs=[2.0, 3.0], specs=[(0, 1, False), (1, 1, False)], expected=72.0),
    # 兼容显式二阶写法 f.d((x, 2))
    dict(name='compat_tuple_k', func=lambda x, y: x ** 3 * y ** 2,
         inputs=[2.0, 3.0], specs=[(0, 2, True)], expected=108.0),
    # 向量变量一阶
    dict(name='vector_first_order', func=lambda x: (x ** 2).sum(),
         inputs=[[1.0, 2.0]], specs=[(0, 1, False)], expected=[2.0, 4.0]),
    # 向量变量二阶（Hessian）
    dict(name='vector_second_order', func=lambda x: (x ** 2).sum(),
         inputs=[[1.0, 2.0]], specs=[(0, 2, False)], expected=[[2.0, 0.0], [0.0, 2.0]]),
    # 多向量混合偏导矩阵：f=(x·y)^2
    dict(name='multi_vector_mixed', func=lambda x, y: (x * y).sum() ** 2,
         inputs=[[1.0, 2.0], [3.0, 4.0]],
         specs=[(0, 1, False), (1, 1, False)],
         expected=[[28.0, 12.0], [8.0, 38.0]]),
    # 标量 + 向量混合：f = x·Σy² → ∂²f/∂x∂y = 2y
    dict(name='scalar_vector_mixed', func=lambda x, y: x * (y ** 2).sum(),
         inputs=[2.0, [1.0, 2.0, 3.0]],
         specs=[(0, 1, False), (1, 1, False)], expected=[2.0, 4.0, 6.0]),
    # 三阶张量：f = Σx⁴
    dict(name='third_order_tensor', func=lambda x: (x ** 4).sum(),
         inputs=[[1.0, 2.0, 3.0]],
         specs=[(0, 1, False), (0, 1, False), (0, 1, False)], expected=_D_THIRD),
    # 同一变量写两次 == 用 (x, 2) 指定阶数
    dict(name='same_var_twice_equiv', func=lambda x, y: x ** 3 * y ** 2,
         inputs=[2.0, 3.0], specs=[(0, 1, False), (0, 1, False)],
         expected=108.0),
    # tuple 形式指定二阶
    dict(name='tuple_order_two', func=lambda x: (x ** 2).sum(),
         inputs=[[1.0, 2.0]], specs=[(0, 2, True)], expected=[[2.0, 0.0], [0.0, 2.0]]),
    # args 合并：直接传 + tuple 混用，阶数累加
    dict(name='merge_triple_order', func=lambda x, y: x ** 3 * y ** 2,
         inputs=[2.0, 3.0], specs=[(0, 1, False), (0, 2, True)], expected=54.0),
    # 显式多阶规格：x 二阶 + y 一阶
    dict(name='mixed_explicit_orders', func=lambda x, y: x ** 3 * y ** 2,
         inputs=[2.0, 3.0], specs=[(0, 2, True), (1, 1, False)], expected=72.0),
    # 异常场景
    dict(name='err_self_not_scalar', expect_error=True,
         action=lambda: _t([1.0, 2.0, 3.0]).d(_t([1.0, 2.0, 3.0]))),
    dict(name='err_no_args', expect_error=True,
         action=lambda: _t(2.0).d()),
    dict(name='err_k_zero', expect_error=True,
         action=lambda: (_t(2.0) ** 2).d((_t(2.0), 0))),
]


# ---------------------------------------------------------------------------
# 用例表：derivative（迁移自 test_010_func_grad.py）
# ---------------------------------------------------------------------------

DERIVATIVE_CASES = [
    dict(name='scalar_to_scalar', func=lambda x: x ** 3, inputs=[2.0],
         expected=(12.0,)),
    dict(name='vector_to_vector', func=lambda x: x ** 2, inputs=[[1.0, 2.0]],
         expected=(np.diag([2.0, 4.0]),)),
    dict(name='two_inputs', func=lambda x, y: (x * y).sum(),
         inputs=[[1.0, 2.0], [3.0, 4.0]],
         expected=([3.0, 4.0], [1.0, 2.0])),
]


# ---------------------------------------------------------------------------
# 用例表：jacobian
# ---------------------------------------------------------------------------

_JAC_HIGHER3 = np.zeros((3, 3, 3))
for _i in range(3):
    _JAC_HIGHER3[_i, _i, _i] = 6 * np.array([1.0, 2.0, 3.0])[_i]

JACOBIAN_CASES = [
    # 单输入单输出
    dict(name='single_single', func=lambda x: x ** 2, inputs=[[1.0, 2.0, 3.0]],
         expected=np.diag([2.0, 4.0, 6.0])),
    # 多输入单输出（标量输入 → 0 维块）
    dict(name='multi_input_scalar', func=lambda x, y: x * y + y ** 2,
         inputs=[2.0, 3.0], expected=(np.array(3.0), np.array(8.0))),
    # 单输入多输出
    dict(name='single_input_multi_output', func=lambda x: (x ** 2, rm.sin(x)),
         inputs=[[1.0, 2.0]],
         expected=(np.diag([2.0, 4.0]), np.diag([np.cos(1.0), np.cos(2.0)]))),
    # 多输入多输出（外层按输出、内层按输入）
    dict(name='multi_input_multi_output',
         func=lambda x, y: (x ** 2 + y ** 3, x * y),
         inputs=[[1.0, 2.0], [3.0, 4.0]],
         expected=((np.diag([2.0, 4.0]), np.diag([27.0, 48.0])),
                   (np.diag([3.0, 4.0]), np.diag([1.0, 2.0])))),
    # strict=False：未使用输入的块为全零
    dict(name='strict_false_unused', func=lambda x, y: x ** 2,
         inputs=[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], strict=False,
         expected=(np.diag([2.0, 4.0, 6.0]), np.zeros((3, 3)))),
    # strict=True：未使用输入报错
    dict(name='err_strict_true_unused', kind='error', expect_error=True, match='independent',
         action=lambda: jacobian(lambda x, y: x ** 2,
                                 (_t([1.0, 2.0]), _t([1.0, 2.0])), strict=True)),
    # create_graph=True：结果保留计算图
    dict(name='create_graph', func=lambda x: x ** 2, inputs=[[1.0, 2.0]],
         create_graph=True, check_requires_grad=True, expected=np.diag([2.0, 4.0])),
    # 单输入单标量输出
    dict(name='single_input_scalar_output', func=lambda x: (x ** 2).sum(),
         inputs=[[1.0, 2.0, 3.0]], expected=np.array([2.0, 4.0, 6.0])),
    # 多输入单标量输出
    dict(name='multi_input_scalar_output', func=lambda x, y: (x * y).sum(),
         inputs=[[1.0, 2.0], [3.0, 4.0]],
         expected=(np.array([3.0, 4.0]), np.array([1.0, 2.0]))),
    # 单输入多标量输出
    dict(name='single_input_multi_scalar',
         func=lambda x: ((x ** 2).sum(), (x ** 3).sum()),
         inputs=[[1.0, 2.0]],
         expected=(np.array([2.0, 4.0]), np.array([3.0, 12.0]))),
    # 多输入多标量输出
    dict(name='multi_input_multi_scalar',
         func=lambda x, y: ((x * y).sum(), (x * y ** 2).sum()),
         inputs=[[1.0, 2.0], [3.0, 4.0]],
         expected=((np.array([3.0, 4.0]), np.array([1.0, 2.0])),
                   (np.array([9.0, 16.0]), np.array([6.0, 16.0])))),
    # 张量 / 标量混合输出
    dict(name='tensor_and_scalar_output',
         func=lambda x: (x ** 2, x.sum()), inputs=[[1.0, 2.0]],
         expected=(np.diag([2.0, 4.0]), np.array([1.0, 1.0]))),
    # 张量 + 标量混合输入（标量块被展平为 (out_numel,)）
    dict(name='multi_input_tensor_and_scalar',
         func=lambda x, y: x ** 2 + y ** 3, inputs=[[1.0, 2.0], 3.0],
         expected=(np.diag([2.0, 4.0]), np.array([27.0, 27.0]))),
    # 多输出形状不同
    dict(name='multi_output_different_shapes',
         func=lambda x: (x ** 2, x[0:1]), inputs=[[1.0, 2.0]],
         expected=(np.diag([2.0, 4.0]), np.array([[1.0, 0.0]]))),
    # create_graph=True 多输入
    dict(name='create_graph_multi_input',
         func=lambda x, y: (x * y).sum(), inputs=[[1.0, 2.0], [3.0, 4.0]],
         create_graph=True, check_requires_grad=True,
         expected=(np.array([3.0, 4.0]), np.array([1.0, 2.0]))),
    # create_graph=True 多输出：只有依赖 x 的块保留计算图
    dict(name='create_graph_multi_output',
         func=lambda x: (x ** 2, x.sum()), inputs=[[1.0, 2.0]], create_graph=True,
         requires_grad_flags=(True, False),
         expected=(np.diag([2.0, 4.0]), np.array([1.0, 1.0]))),
    # create_graph=True + 未使用输入
    dict(name='create_graph_unused_input', func=lambda x, y: x ** 2,
         inputs=[[1.0, 2.0], [1.0, 2.0]], strict=False, create_graph=True,
         expected=(np.diag([2.0, 4.0]), np.zeros((2, 2)))),
    # 异常：非张量输入
    dict(name='err_non_tensor_input', kind='error', expect_error=True, match='must be a tensor',
         action=lambda: jacobian(lambda x: x ** 2, np.array([1.0, 2.0]))),
    # 高阶：先求一阶 Jacobian，再对其求 Jacobian
    dict(name='higher_order', kind='higher', func=lambda x: x ** 3,
         inputs=[[1.0, 2.0, 3.0]],
         expected_first=np.diag([3.0, 12.0, 27.0]),
         expected=_JAC_HIGHER3),
    # 2D 输入：单输入单输出时展平
    dict(name='2d_input', func=lambda x: x.sum(),
         inputs=[np.ones((2, 3))], expected=np.ones(6)),
    # 空输出
    dict(name='empty_output', kind='empty', func=lambda x: (2 * x)[0:0],
         inputs=[[1.0, 2.0, 3.0]], create_graph=True, expected_shape=(0, 3)),
]


# ---------------------------------------------------------------------------
# 用例表：hessian
# ---------------------------------------------------------------------------

_HESS_2D = 2.0 * np.eye(6).reshape(2, 3, 2, 3)

HESSIAN_CASES = [
    # 标量输入 / 标量输出
    dict(name='scalar_input', func=lambda x: x ** 2, inputs=[2.0], expected=2.0),
    # 张量输入
    dict(name='tensor_input',
         func=lambda x: x[0] ** 2 + x[1] ** 2 + x[0] * x[1],
         inputs=[[1.0, 2.0]], expected=[[2.0, 1.0], [1.0, 2.0]]),
    # 标量多输入（每个块 0 维）
    dict(name='multi_input_scalar', func=lambda x, y: x ** 2 * y + y ** 3,
         inputs=[2.0, 3.0],
         expected=((np.array(6.0), np.array(4.0)), (np.array(4.0), np.array(18.0)))),
    # 张量多输入
    dict(name='multi_input_tensor',
         func=lambda x, y: (x ** 2).sum() + (y ** 3).sum() + (x * y).sum(),
         inputs=[[1.0, 2.0], [3.0, 4.0]],
         expected=((2.0 * np.eye(2), np.eye(2)),
                   (np.eye(2), np.diag([18.0, 24.0])))),
    # 2D 输入：形状为 input.shape + input.shape
    dict(name='2d_input', func=lambda x: (x ** 2).sum(),
         inputs=[np.ones((2, 3))], expected=_HESS_2D),
    # 三元素向量输入
    dict(name='larger_vector',
         func=lambda x: x[0] ** 2 + x[1] ** 3 + x[0] * x[1] * x[2] + x[2] ** 2,
         inputs=[[1.0, 2.0, 3.0]],
         expected=[[2.0, 3.0, 2.0], [3.0, 12.0, 1.0], [2.0, 1.0, 2.0]]),
    # strict=False：未使用输入的块为同形状零块
    dict(name='strict_false_unused', func=lambda x, y: (x ** 2).sum(),
         inputs=[[1.0, 2.0], [1.0, 2.0]], strict=False,
         expected=((2.0 * np.eye(2), np.zeros((2, 2))),
                   (np.zeros((2, 2)), np.zeros((2, 2))))),
    # strict=True：未使用输入报错
    dict(name='err_strict_true_unused', kind='error', expect_error=True, match='independent',
         action=lambda: hessian(lambda x, y: (x ** 2).sum(),
                                (_t([1.0, 2.0]), _t([1.0, 2.0])), strict=True)),
    # create_graph=True
    dict(name='create_graph', func=lambda x: (x ** 2).sum(), inputs=[[1.0, 2.0]],
         create_graph=True, check_requires_grad=True, expected=2.0 * np.eye(2)),
    # create_graph=True 多输入
    dict(name='create_graph_multi_input',
         func=lambda x, y: (x ** 2).sum() + (y ** 3).sum(),
         inputs=[[1.0, 2.0], [3.0, 4.0]], create_graph=True,
         check_requires_grad=True,
         expected=((2.0 * np.eye(2), np.zeros((2, 2))),
                   (np.zeros((2, 2)), np.diag([18.0, 24.0])))),
    # 异常场景
    dict(name='err_non_scalar_output', kind='error', expect_error=True, match='scalar-valued',
         action=lambda: hessian(lambda x: x ** 2, _t([1.0, 2.0]))),
    dict(name='err_non_tensor_input', kind='error', expect_error=True, match='must be a tensor',
         action=lambda: hessian(lambda x: (x ** 2).sum(), np.array([1.0, 2.0]))),
]


# ---------------------------------------------------------------------------
# 用例表：jvp
# ---------------------------------------------------------------------------

JVP_CASES = [
    # 单输入单输出（v 与输入同形）
    dict(name='single_single', func=lambda x: x ** 2, inputs=[[1.0, 2.0]],
         v=[3.0, 4.0], expected=(np.array([1.0, 4.0]), np.array([6.0, 16.0]))),
    # 单输入多输出
    dict(name='single_input_multi_output', func=lambda x: (x ** 2, x.sum()),
         inputs=[[1.0, 2.0]], v=[3.0, 4.0],
         expected=((np.array([1.0, 4.0]), np.array(3.0)),
                   (np.array([6.0, 16.0]), np.array(7.0)))),
    # 多输入单输出
    dict(name='multi_input_single_output', func=lambda x, y: (x * y).sum(),
         inputs=[[1.0, 2.0], [3.0, 4.0]], v=([1.0, 1.0], [2.0, 1.0]),
         expected=(np.array(11.0), np.array(11.0))),
    # 多输入多输出
    dict(name='multi_input_multi_output',
         func=lambda x, y: (x @ y, x.sum() + y.sum()),
         inputs=[[1.0, 2.0], [3.0, 4.0]], v=([1.0, 1.0], [1.0, 1.0]),
         expected=((np.array(11.0), np.array(10.0)), (np.array(10.0), np.array(4.0)))),
    # 标量输入 / 标量输出（v 为标量）
    dict(name='scalar_input', func=lambda x: x ** 3, inputs=[2.0], v=1.0,
         expected=(np.array(8.0), np.array(12.0))),
    # create_graph=True 保留计算图
    dict(name='create_graph', func=lambda x, y: (x * y).sum(),
         inputs=[[1.0, 2.0], [3.0, 4.0]], v=([1.0, 1.0], [1.0, 1.0]),
         create_graph=True, check_requires_grad=True,
         expected=(np.array(11.0), np.array(10.0))),
    # strict=False：未使用输入不报错
    dict(name='strict_false_unused', func=lambda x, y: (x ** 2).sum(),
         inputs=[[1.0, 2.0], [1.0, 2.0]], v=([1.0, 1.0], [1.0, 1.0]),
         strict=False, expected=(np.array(5.0), np.array(6.0))),
    # strict=True：未使用输入报错
    dict(name='err_strict_true_unused', kind='error', expect_error=True, match='independent',
         action=lambda: jvp(lambda x, y: (x ** 2).sum(),
                            (_t([1.0, 2.0]), _t([1.0, 2.0])),
                            (_t([1.0, 1.0]), _t([1.0, 1.0])), strict=True)),
]


# ---------------------------------------------------------------------------
# 用例表：vjp
# ---------------------------------------------------------------------------

VJP_CASES = [
    # 单输入单输出
    dict(name='single_single', func=lambda x: x ** 2, inputs=[[1.0, 2.0]],
         v=[1.0, 1.0], expected=(np.array([1.0, 4.0]), np.array([2.0, 4.0]))),
    # 单输入多输出（v 按输出给出）
    dict(name='single_input_multi_output', func=lambda x: (x ** 2, x.sum()),
         inputs=[[1.0, 2.0]], v=([1.0, 1.0], 2.0),
         expected=((np.array([1.0, 4.0]), np.array(3.0)), np.array([4.0, 6.0]))),
    # 多输入单输出（标量输出可给标量 v）
    dict(name='multi_input_single_output', func=lambda x, y: (x * y).sum(),
         inputs=[[1.0, 2.0], [3.0, 4.0]], v=1.0,
         expected=(np.array(11.0), (np.array([3.0, 4.0]), np.array([1.0, 2.0])))),
    # 多输入多输出
    dict(name='multi_input_multi_output',
         func=lambda x, y: (x @ y, x.sum() + y.sum()),
         inputs=[[1.0, 2.0], [3.0, 4.0]], v=(1.0, 1.0),
         expected=((np.array(11.0), np.array(10.0)),
                   (np.array([4.0, 5.0]), np.array([2.0, 3.0])))),
    # 标量输出可省略 v
    dict(name='scalar_output_optional_v', func=lambda x: (x ** 2).sum(),
         inputs=[[1.0, 2.0]], v=None,
         expected=(np.array(5.0), np.array([2.0, 4.0]))),
    # 标量输入 / 标量输出
    dict(name='scalar_input', func=lambda x: x ** 3, inputs=[2.0], v=1.0,
         expected=(np.array(8.0), np.array(12.0))),
    # create_graph=True 保留计算图
    dict(name='create_graph', func=lambda x: (x ** 2).sum(),
         inputs=[[1.0, 2.0]], v=1.0, create_graph=True, check_requires_grad=True,
         expected=(np.array(5.0), np.array([2.0, 4.0]))),
    # create_graph=True 多输入
    dict(name='create_graph_multi_input', func=lambda x, y: (x * y).sum(),
         inputs=[[1.0, 2.0], [3.0, 4.0]], v=1.0, create_graph=True,
         check_requires_grad=True,
         expected=(np.array(11.0), (np.array([3.0, 4.0]), np.array([1.0, 2.0])))),
    # strict=False：未使用输入的梯度块为零
    dict(name='strict_false_unused', func=lambda x, y: (x ** 2).sum(),
         inputs=[[1.0, 2.0], [1.0, 2.0]], v=1.0, strict=False,
         expected=(np.array(5.0), (np.array([2.0, 4.0]), np.zeros(2)))),
    # strict=True：未使用输入报错
    dict(name='err_strict_true_unused', kind='error', expect_error=True, match='independent',
         action=lambda: vjp(lambda x, y: (x ** 2).sum(),
                            (_t([1.0, 2.0]), _t([1.0, 2.0])), _t(1.0), strict=True)),
]


# ---------------------------------------------------------------------------
# 用例表：hvp / vhp（两者数值一致，共用同一张表）
# ---------------------------------------------------------------------------

_HVP_X3 = np.sin([0.5, 1.0])
_HVP_OUT3 = float(_HVP_X3.sum() + (np.array([0.5, 1.0]) * np.array([1.0, 2.0])).sum())

HVP_CASES = [
    # 单输入单输出（标量输入 → v 为标量）
    dict(name='single_single', func=lambda x: x ** 3, inputs=[2.0], v=1.0,
         expected=(np.array(8.0), np.array(12.0))),
    # 多输入单输出
    dict(name='multi_input_scalar', func=lambda x, y: x ** 2 + y ** 3,
         inputs=[2.0, 3.0], v=(1.0, 1.0),
         expected=(np.array(31.0), (np.array(2.0), np.array(18.0)))),
    # 非平凡计算图：sin + sum + 内积
    dict(name='nonlinear_graph',
         func=lambda x, y: rm.sin(x).sum() + (x * y).sum(),
         inputs=[[0.5, 1.0], [1.0, 2.0]], v=([1.0, 1.0], [1.0, 1.0]),
         expected=(np.array(_HVP_OUT3),
                   (1.0 - _HVP_X3, np.ones(2)))),
    # 标量输入：f=x^4 → f''=12x^2 → HVP=48
    dict(name='scalar_quartic', func=lambda x: x ** 4, inputs=[2.0], v=1.0,
         expected=(np.array(16.0), np.array(48.0))),
    # 2D 输入：H=2I → HVP=2v
    dict(name='2d_input', func=lambda x: (x ** 2).sum(),
         inputs=[np.ones((2, 2))], v=np.ones((2, 2)),
         expected=(np.array(4.0), 2.0 * np.ones((2, 2)))),
    # 多输入张量
    dict(name='multi_input_tensor',
         func=lambda x, y: (x ** 2).sum() + (y ** 3).sum(),
         inputs=[[1.0, 2.0], [3.0, 4.0]], v=([1.0, 1.0], [1.0, 1.0]),
         expected=(np.array(96.0), (np.array([2.0, 2.0]), np.array([18.0, 24.0])))),
    # create_graph=True 保留计算图
    dict(name='create_graph', func=lambda x: (x ** 2).sum(),
         inputs=[[1.0, 2.0]], v=[1.0, 1.0], create_graph=True,
         check_requires_grad=True,
         expected=(np.array(5.0), np.array([2.0, 2.0]))),
    # create_graph=True 多输入
    dict(name='create_graph_multi_input',
         func=lambda x, y: (x ** 2).sum() + (y ** 3).sum(),
         inputs=[[1.0, 2.0], [3.0, 4.0]], v=([1.0, 1.0], [1.0, 1.0]),
         create_graph=True, check_requires_grad=True,
         expected=(np.array(96.0), (np.array([2.0, 2.0]), np.array([18.0, 24.0])))),
    # strict=False：未使用输入的 HVP 块为零
    dict(name='strict_false_unused', func=lambda x, y: (x ** 2).sum(),
         inputs=[[1.0, 2.0], [1.0, 2.0]], v=([1.0, 1.0], [1.0, 1.0]), strict=False,
         expected=(np.array(5.0), (np.array([2.0, 2.0]), np.zeros(2)))),
    # strict=True：未使用输入报错
    dict(name='err_strict_true_unused', kind='error', expect_error=True, match='independent',
         action=lambda: hvp(lambda x, y: (x ** 2).sum(),
                            (_t([1.0, 2.0]), _t([1.0, 2.0])),
                            (_t([1.0, 1.0]), _t([1.0, 1.0])), strict=True)),
]


# ---------------------------------------------------------------------------
# 各组执行器（把用例表翻译成具体调用）
# ---------------------------------------------------------------------------

def _grad_case(case):
    if 'action' in case:
        _run_case(case['name'], case['action'], True, case.get('match'))
        return
    ts = _inputs(case['inputs'])
    order = case.get('order', 1)

    def _call():
        kwargs = {}
        if 'grad_outputs' in case:
            kwargs['grad_outputs'] = _t(case['grad_outputs'], requires_grad=False)
        if case.get('allow_unused'):
            kwargs['allow_unused'] = True
        y = case['func'](*ts)
        targets = ts[0] if len(ts) == 1 else ts
        if order == 1:
            result = rm_grad(y, targets, **kwargs)
        else:
            g = rm_grad(y, targets, create_graph=True, **kwargs)[0]
            for k in range(order - 1):
                g = rm_grad(g, ts[0], create_graph=(k < order - 2))[0]
            result = (g,)
        _assert_tree(result, case['expected'], case['name'])

    _run_case(case['name'], _call)


def _d_case(case):
    if 'action' in case:
        _run_case(case['name'], case['action'], True, case.get('match'))
        return
    ts = _inputs(case['inputs'])
    f = case['func'](*ts)

    def _build(specs):
        # (输入下标, 阶数, 是否用 tuple 形式)；非 tuple 形式按阶数重复传参
        args = []
        for i, k, as_tuple in specs:
            if as_tuple:
                args.append((ts[i], k))
            else:
                args.extend([ts[i]] * k)
        return args

    def _call():
        _assert_tree(f.d(*_build(case['specs'])), case['expected'], case['name'])

    _run_case(case['name'], _call)


def _derivative_case(case):
    ts = _inputs(case['inputs'])

    def _call():
        result = derivative(case['func'])(*ts)
        _assert_tree(result, case['expected'], case['name'])

    _run_case(case['name'], _call)


def _jacobian_case(case):
    kind = case.get('kind', 'call')
    if kind == 'error':
        _run_case(case['name'], case['action'], True, case.get('match'))
        return
    ts = _inputs(case['inputs'])
    arg = ts[0] if len(ts) == 1 else tuple(ts)
    create_graph = case.get('create_graph', False)
    strict = case.get('strict', False)

    def _call():
        if kind == 'higher':
            first = jacobian(case['func'], ts[0], create_graph=True)
            _assert_tree(first, case['expected_first'], f"{case['name']}.first")
            second = jacobian(lambda z: jacobian(case['func'], z, create_graph=True), ts[0])
            _assert_tree(second, case['expected'], case['name'])
            return
        if kind == 'empty':
            result = jacobian(case['func'], ts[0], create_graph=create_graph)
            assert result.shape == case['expected_shape'], \
                f"{case['name']}: 形状 {result.shape} != {case['expected_shape']}"
            assert result.requires_grad == create_graph, \
                f"{case['name']}: requires_grad 应为 {create_graph}"
            assert _np(result).size == 0, f"{case['name']}: 空输出应无数值"
            return
        result = jacobian(case['func'], arg, create_graph=create_graph, strict=strict)
        _assert_tree(result, case['expected'], case['name'])
        if case.get('check_requires_grad'):
            _assert_requires_grad(result, case['name'])
        if 'requires_grad_flags' in case:
            _assert_requires_grad_flags(result, case['requires_grad_flags'], case['name'])

    _run_case(case['name'], _call)


def _hessian_case(case):
    if case.get('kind') == 'error':
        _run_case(case['name'], case['action'], True, case.get('match'))
        return
    ts = _inputs(case['inputs'])
    arg = ts[0] if len(ts) == 1 else tuple(ts)

    def _call():
        result = hessian(case['func'], arg,
                         create_graph=case.get('create_graph', False),
                         strict=case.get('strict', False))
        _assert_tree(result, case['expected'], case['name'])
        if case.get('check_requires_grad'):
            _assert_requires_grad(result, case['name'])

    _run_case(case['name'], _call)


def _jvp_vjp_case(case, api):
    if case.get('kind') == 'error':
        _run_case(case['name'], case['action'], True, case.get('match'))
        return
    ts = _inputs(case['inputs'])
    arg = ts[0] if len(ts) == 1 else tuple(ts)
    v_cfg = case.get('v')
    v_arg = None if v_cfg is None else _vector(v_cfg)

    def _call():
        result = api(case['func'], arg, v_arg,
                     create_graph=case.get('create_graph', False),
                     strict=case.get('strict', False))
        _assert_tree(result, case['expected'], case['name'])
        if case.get('check_requires_grad'):
            _assert_requires_grad(result[1], f"{case['name']}.vector_product")

    _run_case(case['name'], _call)


def _hvp_vhp_case(case, api):
    if case.get('kind') == 'error':
        _run_case(case['name'], case['action'], True, case.get('match'))
        return
    ts = _inputs(case['inputs'])
    arg = ts[0] if len(ts) == 1 else tuple(ts)
    v_arg = _vector(case['v'])

    def _call():
        result = api(case['func'], arg, v_arg,
                     create_graph=case.get('create_graph', False),
                     strict=case.get('strict', False))
        _assert_tree(result, case['expected'], case['name'])
        if case.get('check_requires_grad'):
            _assert_requires_grad(result[1], f"{case['name']}.hv")

    _run_case(case['name'], _call)


# ---------------------------------------------------------------------------
# 测试类：每个 test_xxx 方法 = 一个用例组
# ---------------------------------------------------------------------------

class TestAutograd(unittest.TestCase):

    # ---------------- grad ----------------
    def test_grad(self):
        _start_group('grad')
        for case in GRAD_CASES:
            _grad_case(case)
        _end_group()

    # ---------------- Tensor.d ----------------
    def test_d(self):
        _start_group('d')
        for case in D_CASES:
            _d_case(case)
        _end_group()

    # ---------------- derivative ----------------
    def test_derivative(self):
        _start_group('derivative')
        for case in DERIVATIVE_CASES:
            _derivative_case(case)

        # create_graph=True：一阶结果保留计算图，可再次求导
        def _create_graph():
            x = _t(2.0)
            first = derivative(lambda z: z ** 3, create_graph=True)(x)
            _assert_tree(first, (12.0,), 'derivative.create_graph.first')
            assert first[0].requires_grad, 'derivative.create_graph: 一阶结果未保留计算图'
            second = derivative(
                lambda z: derivative(lambda w: w ** 3, create_graph=True)(z)[0])(x)
            _assert_tree(second, (12.0,), 'derivative.create_graph.second')

        _run_case('create_graph_second_order', _create_graph)
        _end_group()

    # ---------------- jacobian ----------------
    def test_jacobian(self):
        _start_group('jacobian')
        for case in JACOBIAN_CASES:
            _jacobian_case(case)
        _end_group()

    # ---------------- hessian ----------------
    def test_hessian(self):
        _start_group('hessian')
        for case in HESSIAN_CASES:
            _hessian_case(case)
        _end_group()

    # ---------------- jvp ----------------
    def test_jvp(self):
        _start_group('jvp')
        for case in JVP_CASES:
            _jvp_vjp_case(case, jvp)
        _end_group()

    # ---------------- vjp ----------------
    def test_vjp(self):
        _start_group('vjp')
        for case in VJP_CASES:
            _jvp_vjp_case(case, vjp)
        _end_group()

    # ---------------- hvp ----------------
    def test_hvp(self):
        _start_group('hvp')
        for case in HVP_CASES:
            _hvp_vhp_case(case, hvp)
        _end_group()

    # ---------------- vhp ----------------
    def test_vhp(self):
        _start_group('vhp')
        # vhp 与 hvp 数值一致：直接复用同一张用例表
        for case in HVP_CASES:
            _hvp_vhp_case(case, vhp)

        # 显式校验 hvp / vhp 的一致性
        def _symmetry():
            x, y = _t([1.0, 2.0]), _t([3.0, 4.0])
            f = lambda a, b: (a ** 2).sum() + (b ** 3).sum() + (a * b).sum()
            v = (_t([0.5, 1.5]), _t([2.0, 0.5]))
            hv = hvp(f, (x, y), v)[1]
            vh = vhp(f, (x, y), v)[1]
            for a, b in zip(hv, vh):
                _assert_tree(a, _np(b), 'hvp_vhp_symmetry')

        _run_case('hvp_vhp_symmetry', _symmetry)
        _end_group()


# ---------------------------------------------------------------------------
# 脚本入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    IS_RUNNING_AS_SCRIPT = True
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}===== test_008_autograd.py ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAutograd))

    result = unittest.TextTestRunner(verbosity=0).run(suite)
    stats.print_summary()
    sys.exit(0 if result.wasSuccessful() else 1)
