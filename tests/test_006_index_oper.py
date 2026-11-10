"""索引操作测试脚本（合并 getitem / scatter / gather / index_select / masked_select / take / index_put_ 用例组）

- 独立运行: python tests/test_006_index_oper.py
  每个用例组输出子用例结果，执行结束打印统计表格
- pytest 调用: pytest tests/test_006_index_oper.py
"""

import unittest
import numpy as np
import time
import sys
import os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.tensordef import tensor as rm_tensor
    from riemann.autograd.grad import grad
    from riemann.autograd.functional import hessian as rm_hessian
    # 从rm.cuda获取cupy引用和CUDA可用性
    CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
    cp = rm.cuda.cp
    RIEMANN_AVAILABLE = True
except ImportError as e:
    print(f"无法导入riemann模块: {e}")
    print("请确保项目路径设置正确")
    RIEMANN_AVAILABLE = False
    CUDA_AVAILABLE = False

# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的索引操作")
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False


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
        total_pass_rate = self.passed_cases / self.total_cases * 100 if self.total_cases else 0.0

        print("\n" + "=" * total_width)
        print(f"{Colors.BOLD}测试统计摘要{Colors.ENDC}")
        print("=" * total_width)
        print(f"总测试用例数: {self.total_cases}")
        print(f"通过测试用例数: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases}{Colors.ENDC}")
        print(f"测试通过率: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{total_pass_rate:.2f}%{Colors.ENDC}")
        print(f"总耗时: {self.total_time:.4f} 秒")
        print("\n各用例测试详情:")
        print("-" * total_width)

        # 打印表头 - 精确计算每个标题的填充
        header_line = ""
        for i, header in enumerate(headers):
            header_width = self._get_display_width(header)
            padding = col_widths[i] - header_width
            header_line += header + " " * padding
        print(header_line)
        print("-" * total_width)

        # 打印数据行 - 精确计算每个值的填充
        for func_name, stat in self.function_stats.items():
            pass_rate = stat["passed"] / stat["total"] * 100 if stat["total"] > 0 else 0
            status_color = Colors.OKGREEN if pass_rate == 100 else Colors.FAIL

            # 计算每个字段的显示宽度并添加适当的填充
            func_name_display = func_name
            func_name_width = self._get_display_width(func_name_display)
            func_name_padding = col_widths[0] - func_name_width

            pass_total_display = f"{stat['passed']}/{stat['total']}"
            pass_total_width = self._get_display_width(pass_total_display)
            pass_total_padding = col_widths[1] - pass_total_width

            # 通过率字段包含颜色代码，但显示宽度只计算实际文本
            pass_rate_display = f"{pass_rate:.2f}"
            pass_rate_width = self._get_display_width(pass_rate_display)
            pass_rate_padding = col_widths[2] - pass_rate_width

            time_display = f"{stat['time']:.4f}"
            time_width = self._get_display_width(time_display)
            time_padding = col_widths[3] - time_width

            # 构建完整的行
            print(
                f"{func_name_display}{' ' * func_name_padding}" +
                f"{pass_total_display}{' ' * pass_total_padding}" +
                f"{status_color}{pass_rate_display}{' ' * pass_rate_padding}{Colors.ENDC}" +
                f"{time_display}{' ' * time_padding}"
            )

        print("=" * total_width)


# 全局统计实例
stats = StatisticsCollector()
# 判断是否作为独立脚本运行
IS_RUNNING_AS_SCRIPT = False


# 比较值的函数
def compare_values(rm_result, torch_result, atol=1e-6, rtol=1e-6):
    """比较Riemann和PyTorch的值是否接近"""
    # 处理None值的情况
    if not TORCH_AVAILABLE:
        # 如果没有PyTorch，只检查riemann结果是否存在
        return rm_result is not None

    if rm_result is None and torch_result is None:
        return True
    if rm_result is None or torch_result is None:
        return False

    # 处理嵌套元组/列表的情况
    if isinstance(rm_result, (list, tuple)) and isinstance(torch_result, (list, tuple)):
        if len(rm_result) != len(torch_result):
            return False

        all_passed = True
        for r, t in zip(rm_result, torch_result):
            if not compare_values(r, t, atol, rtol):
                all_passed = False
                break

        return all_passed

    # 转换为numpy数组
    try:
        # 处理Riemann结果（CUDA张量也通过detach().cpu()统一转换）
        rm_data = rm_result.detach().cpu().numpy()

        # 处理PyTorch结果
        torch_data = torch_result.detach().cpu().numpy()
    except Exception as e:
        print(f"比较值转换错误: {e}")
        return False

    # 处理形状不匹配的情况
    try:
        # 增加容差参数
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False


# =============================================================================
# 数据驱动的用例定义
# =============================================================================

# getitem 用例组：每个成员的 kind 决定执行方式，其余字段为场景参数
GETITEM_CASES = [
    {"name": "标量索引", "shape": (5,), "index": 2},
    {"name": "切片索引", "shape": (5,), "index": slice(1, 4)},
    {"name": "整数数组索引", "shape": (5,), "index": [0, 2, 4]},
    {"name": "布尔索引", "shape": (5,), "mask": True},
    {"name": "二维单索引", "shape": (3, 4), "index": 1},
    {"name": "二维元组索引", "shape": (3, 4), "index": (1, 2)},
    {"name": "二维混合索引（整数+切片）", "shape": (3, 4), "index": (1, slice(0, 3))},
    {"name": "三维数组的多维度索引", "shape": (2, 3, 4), "index": (0, 1, 2)},
    {"name": "重复索引位置（梯度累加）", "shape": (5,), "index": [0, 0, 2, 2]},
    {"name": "负切片索引", "shape": (10,), "index": slice(-5, -1)},
    {"name": "dots索引", "shape": (2, 3, 4), "index": (Ellipsis, 2)},
    {"name": "3级视图索引", "shape": (5, 5, 5), "kind": "view"},
    {"name": "索引操作的二阶导数(Hessian)", "kind": "hessian", "data": [1.0, 2.0]},
]

# scatter 用例组：op 决定调用哪个接口，其余字段为场景参数


def _scatter_case_name(op, entry):
    """根据用例参数自动生成子用例名称，避免逐条手写名称"""
    labels = {
        "scatter_src": "scatter(src)",
        "scatter_value": "scatter(value)",
        "scatter_inplace": "scatter_",
        "scatter_add_inplace": "scatter_add_",
        "scatter_add": "scatter_add",
        "scatter_add_scalar": "scatter_add(scalar)",
        "direct_indices": "_compute_direct_indices",
        "boundary": "边界情况",
    }
    parts = [labels[op]]
    if "dim" in entry:
        parts.append(f"维度={entry['dim']}")
    if "shape" in entry:
        parts.append(f"形状={entry['shape']}")
    if "value" in entry:
        parts.append(f"值={entry['value']}")
    if "tag" in entry:
        parts.append(entry["tag"])
    return " ".join(parts)


def _scatter_cases(op, entries):
    """把同一接口的多组场景参数展开为数据驱动的用例列表"""
    return [{"op": op, "name": _scatter_case_name(op, entry), **entry} for entry in entries]


SCATTER_CASES = (
    _scatter_cases("scatter_src", [
        {"shape": (3, 4), "dim": 0},
        {"shape": (2, 3, 4), "dim": 1},
        {"shape": (1, 5, 3), "dim": 0},
    ])
    + _scatter_cases("scatter_value", [
        {"shape": (3, 4), "dim": 1, "value": 2.0},
        {"shape": (2, 3, 4), "dim": 2, "value": -3.5},
    ])
    + _scatter_cases("scatter_inplace", [
        {"shape": (4, 5), "dim": 0},
        {"shape": (2, 4, 3), "dim": 2},
    ])
    + _scatter_cases("scatter_add_inplace", [
        {"shape": (3, 4), "dim": 1},
        {"shape": (2, 3, 4), "dim": 0},
        {"shape": (1, 5, 3), "dim": 2},
    ])
    + _scatter_cases("scatter_add", [
        {"shape": (3, 4), "dim": 1},
        {"shape": (2, 3, 4), "dim": 0},
        {"shape": (1, 5, 3), "dim": 2},
    ])
    + _scatter_cases("scatter_add_scalar", [
        {"shape": (3, 4), "dim": 0, "value": 1.5},
        {"shape": (2, 3), "dim": 1, "value": -2.0},
    ])
    + _scatter_cases("direct_indices", [
        {"shape": (3, 4), "dim": 0},
        {"shape": (3, 4), "dim": 1},
        {"shape": (2, 3, 4), "dim": 1},
        {"shape": (2, 3, 4), "dim": -1},
        {"shape": (1, 5, 3, 2), "dim": 2},
        {
            "shape": (2, 2), "dim": 0, "tag": "特殊索引模式",
            "tensor": np.zeros((2, 2)),
            "index": [[0, 1], [1, 0]],
            "src": np.ones((2, 2)),
            "expected": np.array([[1, 1], [1, 1]], dtype=np.float32),
        },
    ])
    + _scatter_cases("boundary", [
        {"shape": (0, 0), "dim": 0},
        {"shape": (3, 4), "dim": -1},
        {"shape": (2, 3), "dim": 0, "same_indices": True},
    ])
)


# gather 用例组：输入/索引为固定数据，grad=True 表示需要比较反向传播结果
def _gather_case(name, input_data, index_data, dim, dtype=np.float32, **extra):
    case = {
        "name": name,
        "dim": dim,
        "input": np.asarray(input_data, dtype=dtype),
        "index": np.asarray(index_data, dtype=np.int64),
    }
    case.update(extra)
    return case


GATHER_CASES = [
    _gather_case("2D张量 dim=0", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0, 1], [1, 0]], 0, grad=True),
    _gather_case("2D张量 dim=1", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0, 2], [1, 0]], 1, grad=True),
    _gather_case("3D张量 dim=0",
                 [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                 [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], 0, grad=True),
    # 中间维采集：dim 既非首维也非末维
    _gather_case("3D张量 dim=1",
                 [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                  [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
                 [[[0, 2]], [[1, 1]]], 1, grad=True),
    _gather_case("3D张量 dim=2",
                 [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                 [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], 2, grad=True),
    # 4D 张量：验证高维下多轴网格索引的拼接
    _gather_case("4D张量 dim=2",
                 np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2),
                 [[[[0, 1]], [[1, 0]]], [[[0, 0]], [[1, 1]]]], 2, grad=True),
    # 空张量无有效梯度路径，仅比较前向结果
    _gather_case("空张量", [], [], 0),
    # 非空输入 + 空索引：输出为空，同时校验空输出下的反向传播
    _gather_case("空索引（空输出）",
                 [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                 np.zeros((0, 3), dtype=np.int64), 1, grad=True),
    _gather_case("单元素张量", [[5.0]], [[0]], 1, grad=True),
    _gather_case("最小索引值", [[10.0, 20.0], [30.0, 40.0]], [[0, 0]], 0, grad=True),
    _gather_case("最大索引值", [[10.0, 20.0], [30.0, 40.0]], [[1, 1]], 0, grad=True),
    _gather_case("小形状索引", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0]], 0, grad=True),
    _gather_case("匹配形状索引", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0, 1, 0], [1, 0, 1]], 0, grad=True),
    # 整型输入不参与梯度计算，仅比较前向结果
    _gather_case("每行多元素采集", np.arange(15).reshape(3, 5), [[0, 1], [2, 3], [1, 2]], 1, dtype=np.int64),
    _gather_case("负维度索引",
                 [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
                 [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], -1, grad=True),
    # 负维度非 -1：验证 dim 归一化到中间维
    _gather_case("负维度索引 dim=-2",
                 [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                  [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
                 [[[0, 2]], [[1, 1]]], -2, grad=True),
    _gather_case("分类任务目标概率提取", [[1.2, 0.5, 2.1], [0.1, 3.2, 0.3]], [[2], [1]], 1, grad=True),
    _gather_case("基本反向传播", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0, 1], [1, 0]], 0, grad=True),
    # 重复索引：各元素被采集次数不同，用于验证梯度按次数累加而非覆盖
    _gather_case("重复索引梯度累加（不均匀计数）",
                 [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                 [[0, 0], [0, 1], [1, 0]], 0, grad=True),
    {
        "name": "复杂计算图梯度",
        "kind": "complex_graph",
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "indices": [np.array([[0, 1]], dtype=np.int64), np.array([[0]], dtype=np.int64)],
        "dims": [0, 1],
    },
]


# =============================================================================
# index_select 用例组
# =============================================================================
INDEX_SELECT_CASES = [
    {"name": "1D dim=0 基础选择", "input": [1.0, 2.0, 3.0, 4.0], "dim": 0, "index": [2, 0], "grad": True},
    {"name": "1D 重复索引梯度累加", "input": [1.0, 2.0, 3.0], "dim": 0, "index": [0, 0, 2], "grad": True},
    {"name": "2D dim=0 行选择", "input": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "dim": 0,
     "index": [2, 0, 1], "grad": True},
    {"name": "2D dim=1 列选择", "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "dim": 1,
     "index": [2, 0], "grad": True},
    {"name": "3D dim=1 中间维选择", "input": np.arange(24, dtype=np.float32).reshape(2, 3, 4), "dim": 1,
     "index": [2, 0, 1], "grad": True},
    {"name": "负维度 dim=-1", "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "dim": -1,
     "index": [1, 0], "grad": True},
    {"name": "空索引", "input": [[1.0, 2.0, 3.0]], "dim": 1, "index": [], "grad": True},
    {"name": "单元素索引", "input": [[1.0, 2.0], [3.0, 4.0]], "dim": 0, "index": [1], "grad": True},
    {"name": "整型输入", "input": np.arange(6).reshape(2, 3), "dim": 0, "index": [1, 1],
     "dtype": np.int64},
]


# =============================================================================
# masked_select 用例组
# =============================================================================
MASKED_SELECT_CASES = [
    {"name": "1D 掩码选择", "input": [1.0, -2.0, 3.0, -4.0],
     "mask": [True, False, True, False], "grad": True},
    {"name": "2D 同形状掩码", "input": [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
     "mask": [[True, False, True], [False, True, False]], "grad": True},
    {"name": "行掩码广播", "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
     "mask": [True, False, True], "grad": True},
    {"name": "列掩码广播", "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
     "mask": [[True], [False]], "grad": True},
    {"name": "全False空结果", "input": [[1.0, 2.0], [3.0, 4.0]],
     "mask": [[False, False], [False, False]], "grad": True},
    {"name": "全True", "input": [[1.0, 2.0], [3.0, 4.0]],
     "mask": [[True, True], [True, True]], "grad": True},
    {"name": "3D 掩码", "input": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
     "mask": (np.arange(8).reshape(2, 2, 2) % 2 == 0), "grad": True},
    {"name": "0维输入True", "input": 3.5, "mask": True, "grad": True},
    {"name": "0维输入False", "input": 3.5, "mask": False, "grad": True},
    {"name": "整型输入", "input": np.arange(6).reshape(2, 3),
     "mask": (np.arange(6).reshape(2, 3) % 2 == 0), "dtype": np.int64},
]


# =============================================================================
# take 用例组
# =============================================================================
TAKE_CASES = [
    {"name": "1D 基础", "input": [1.0, 2.0, 3.0, 4.0], "index": [2, 0, 3], "grad": True},
    {"name": "重复索引梯度累加", "input": [1.0, 2.0, 3.0], "index": [0, 0, 2], "grad": True},
    {"name": "2D输入扁平化", "input": [[1.0, 2.0], [3.0, 4.0]], "index": [3, 1, 2], "grad": True},
    {"name": "二维索引输出", "input": np.arange(6, dtype=np.float32).reshape(2, 3),
     "index": [[0, 5], [3, 1]], "grad": True},
    {"name": "0维标量索引", "input": np.arange(6, dtype=np.float32).reshape(2, 3),
     "index": np.array(4, dtype=np.int64), "grad": True},
    {"name": "空索引", "input": [[1.0, 2.0], [3.0, 4.0]], "index": [], "grad": True},
    {"name": "单元素输入", "input": [5.0], "index": [0, 0], "grad": True},
    {"name": "最大索引", "input": [[1.0, 2.0], [3.0, 4.0]], "index": [3], "grad": True},
    {"name": "整型输入", "input": np.arange(6).reshape(2, 3), "index": [4, 0],
     "dtype": np.int64},
]


# =============================================================================
# index_put_ 用例组
# =============================================================================
INDEX_PUT_CASES = [
    {"name": "1D 覆盖写入", "input": [1.0, 2.0, 3.0, 4.0], "indices": [[0, 2]],
     "values": [10.0, 20.0], "accumulate": False, "grad": True},
    {"name": "1D 累加写入重复索引", "input": [1.0, 2.0, 3.0], "indices": [[1, 1]],
     "values": [10.0, 20.0], "accumulate": True, "grad": True},
    {"name": "1D 覆盖重复索引（最后写入生效）", "input": [1.0, 2.0, 3.0],
     "indices": [[1, 1]], "values": [10.0, 20.0], "accumulate": False, "grad": True},
    {"name": "2D 对角写入", "input": [[1.0, 2.0], [3.0, 4.0]],
     "indices": [[0, 1], [1, 0]], "values": [10.0, 20.0], "accumulate": False, "grad": True},
    {"name": "2D 行选择写入", "input": np.arange(12, dtype=np.float32).reshape(3, 4),
     "indices": [[0, 2]], "values": [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
     "accumulate": False, "grad": True},
    {"name": "0维值广播", "input": [[1.0, 2.0], [3.0, 4.0]],
     "indices": [[0, 1], [1, 0]], "values": 5.0, "accumulate": False, "grad": True},
    {"name": "累加时0维值广播到重复行", "input": np.zeros((3, 4), dtype=np.float32),
     "indices": [[1, 1]], "values": 2.0, "accumulate": True, "grad": True},
    {"name": "布尔掩码写入", "input": [[1.0, 2.0], [3.0, 4.0]],
     "indices": [[[True, False], [False, True]], ], "values": 0.0,
     "accumulate": False, "grad": True, "mask_indices": True},
    {"name": "整型输入", "input": np.arange(6).reshape(2, 3),
     "indices": [[0, 1], [2, 0]], "values": [100, 200], "accumulate": False,
     "dtype": np.int64},
]


# =============================================================================
# 新增函数的边界/错误用例组
# =============================================================================
INDEX_SELECT_EDGE_CASES = [
    {"name": "out参数正常写入", "kind": "out",
     "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "dim": 1, "index": [2, 0]},
    {"name": "requires_grad输入与out报错", "kind": "out_grad_error",
     "input": [[1.0, 2.0], [3.0, 4.0]], "dim": 0, "index": [1]},
    {"name": "越界索引报错", "kind": "range_error",
     "input": [[1.0, 2.0], [3.0, 4.0]], "dim": 0, "index": [2]},
    {"name": "负索引报错", "kind": "range_error",
     "input": [[1.0, 2.0], [3.0, 4.0]], "dim": 0, "index": [-1]},
    {"name": "非一维索引报错", "kind": "index_ndim_error",
     "input": [[1.0, 2.0], [3.0, 4.0]], "dim": 0, "index": [[0, 1]]},
    {"name": "非连续输入", "kind": "non_contiguous",
     "input": np.arange(12, dtype=np.float32).reshape(3, 4), "dim": 0, "index": [2, 0]},
]

MASKED_SELECT_EDGE_CASES = [
    {"name": "out参数正常写入", "kind": "out",
     "input": [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
     "mask": [[True, False, True], [False, False, True]]},
    {"name": "requires_grad输入与out报错", "kind": "out_grad_error",
     "input": [[1.0, 2.0], [3.0, 4.0]], "mask": [[True, False], [False, True]]},
    {"name": "非布尔掩码报错", "kind": "mask_dtype_error",
     "input": [[1.0, 2.0], [3.0, 4.0]], "mask": np.array([[1, 0], [0, 1]], dtype=np.int64)},
    {"name": "掩码无法广播报错", "kind": "mask_broadcast_error",
     "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "mask": [True, False]},
    {"name": "非连续输入", "kind": "non_contiguous",
     "input": np.arange(12, dtype=np.float32).reshape(3, 4),
     "mask": (np.arange(12).reshape(3, 4) % 2 == 0)},
]

TAKE_EDGE_CASES = [
    {"name": "负索引与梯度", "kind": "normal",
     "input": np.arange(6, dtype=np.float32).reshape(2, 3), "index": [-1, -3, 2]},
    {"name": "越界索引报错", "kind": "range_error",
     "input": [[1.0, 2.0], [3.0, 4.0]], "index": [4]},
    {"name": "过小负索引报错", "kind": "range_error",
     "input": [[1.0, 2.0], [3.0, 4.0]], "index": [-5]},
    {"name": "非连续输入", "kind": "non_contiguous",
     "input": np.arange(6, dtype=np.float32).reshape(2, 3), "index": [5, 0, 3, 3]},
    {"name": "2D重复索引梯度", "kind": "normal",
     "input": [1.0, 2.0, 3.0], "index": [[0, 0], [2, 2]]},
]

INDEX_PUT_EDGE_CASES = [
    {"name": "索引广播写入", "kind": "broadcast_indices",
     "input": np.arange(12, dtype=np.float32).reshape(3, 4),
     "indices": [np.array([[0], [2]], dtype=np.int64), np.array([[1, 2, 3]], dtype=np.int64)],
     "values": np.arange(6, dtype=np.float32).reshape(2, 3) + 10.0, "accumulate": False},
    {"name": "对base与values的梯度", "kind": "base_grad",
     "input": [[1.0, 2.0], [3.0, 4.0]],
     "indices": [np.array([0, 1], dtype=np.int64), np.array([1, 0], dtype=np.int64)],
     "values": [10.0, 20.0], "accumulate": False},
    {"name": "叶子张量原地写入报错", "kind": "leaf_error",
     "input": [[1.0, 2.0], [3.0, 4.0]],
     "indices": [np.array([0], dtype=np.int64)], "values": 5.0, "accumulate": False},
    {"name": "空索引无操作", "kind": "empty",
     "input": [1.0, 2.0, 3.0], "indices": [np.array([], dtype=np.int64)],
     "values": np.array([], dtype=np.float32), "accumulate": False},
    {"name": "负索引写入", "kind": "negative",
     "input": np.arange(12, dtype=np.float32).reshape(3, 4),
     "indices": [np.array([-1], dtype=np.int64), np.array([-1], dtype=np.int64)],
     "values": [99.0], "accumulate": False},
    {"name": "越界索引报错", "kind": "index_range_error",
     "input": [1.0, 2.0, 3.0], "indices": [np.array([3], dtype=np.int64)],
     "values": [5.0], "accumulate": False},
    {"name": "values形状过大报错", "kind": "values_shape_error",
     "input": [1.0, 2.0, 3.0], "indices": [np.array([0], dtype=np.int64)],
     "values": [1.0, 2.0], "accumulate": False},
    {"name": "values无法广播报错", "kind": "values_shape_error",
     "input": [[1.0, 2.0], [3.0, 4.0]],
     "indices": [np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)],
     "values": [[1.0, 2.0], [3.0, 4.0]], "accumulate": False},
]


# =============================================================================
# 新增函数的中级用例组：dtype、高维与空输入
# =============================================================================
INDEX_SELECT_MEDIUM_CASES = [
    {"name": "int32索引", "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "dim": 1,
     "index": [2, 0], "index_dtype": np.int32, "grad": True},
    {"name": "float64输入", "input": [[1.0, 2.0], [3.0, 4.0]], "dim": 0,
     "index": [1, 0], "dtype": np.float64, "grad": True},
    {"name": "布尔输入", "input": [[True, False], [False, True]], "dim": 0,
     "index": [1, 0], "dtype": np.bool_},
    {"name": "int32输入", "input": np.arange(6, dtype=np.int32).reshape(2, 3), "dim": 0,
     "index": [1, 1], "dtype": np.int32},
    {"name": "4D输入", "input": np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1), "dim": 2,
     "index": [2, 0, 3], "grad": True},
    {"name": "空维度+空索引", "input": np.zeros((2, 0, 3), dtype=np.float32), "dim": 1,
     "index": [], "grad": True},
    {"name": "空维度+非空索引报错", "kind": "range_error",
     "input": np.zeros((2, 0, 3), dtype=np.float32), "dim": 1, "index": [0]},
]

MASKED_SELECT_MEDIUM_CASES = [
    {"name": "float64输入", "input": [[1.0, -2.0], [3.0, -4.0]],
     "mask": [[True, False], [False, True]], "dtype": np.float64, "grad": True},
    {"name": "布尔输入", "input": [[True, False], [False, True]],
     "mask": [[True, False], [False, True]], "dtype": np.bool_},
    {"name": "int32输入", "input": np.arange(6, dtype=np.int32).reshape(2, 3),
     "mask": (np.arange(6).reshape(2, 3) % 2 == 0), "dtype": np.int32},
    {"name": "高维广播", "input": np.arange(24, dtype=np.float32).reshape(4, 2, 3),
     "mask": np.array([[True, False, True], [False, True, False]]), "grad": True},
    {"name": "首维广播", "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
     "mask": [[True, True, True]], "grad": True},
    {"name": "空输入", "input": np.zeros((0,), dtype=np.float32),
     "mask": np.zeros((0,), dtype=np.bool_), "grad": True},
    {"name": "空2D输入", "input": np.zeros((0, 3), dtype=np.float32),
     "mask": np.zeros((0, 3), dtype=np.bool_), "grad": True},
]

TAKE_MEDIUM_CASES = [
    {"name": "float64输入", "input": [[1.0, 2.0], [3.0, 4.0]], "index": [3, 0],
     "dtype": np.float64, "grad": True},
    {"name": "布尔输入", "input": [True, False, True], "index": [2, 0],
     "dtype": np.bool_},
    {"name": "int32输入", "input": np.arange(6, dtype=np.int32).reshape(2, 3),
     "index": [4, 1], "dtype": np.int32},
    {"name": "4D输入", "input": np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1),
     "index": [23, 0, 12], "grad": True},
    {"name": "4D索引", "input": [[1.0, 2.0], [3.0, 4.0]],
     "index": (np.arange(16).reshape(2, 2, 2, 2) % 4), "grad": True},
    {"name": "空输入+空索引", "input": np.zeros((0,), dtype=np.float32),
     "index": [], "grad": True},
    {"name": "0维输入+0索引", "input": 7.0, "index": 0, "grad": True},
]

INDEX_PUT_MEDIUM_CASES = [
    {"name": "3D三元索引", "input": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
     "indices": [np.array([0, 1], dtype=np.int64), np.array([2, 0], dtype=np.int64),
                 np.array([3, 1], dtype=np.int64)],
     "values": [10.0, 20.0], "accumulate": False, "grad": True},
    {"name": "float64输入与values", "input": [[1.0, 2.0], [3.0, 4.0]],
     "indices": [np.array([0, 1], dtype=np.int64), np.array([1, 0], dtype=np.int64)],
     "values": [10.0, 20.0], "dtype": np.float64, "grad": True, "accumulate": False},
    {"name": "int32输入", "input": np.arange(6, dtype=np.int32).reshape(2, 3),
     "indices": [np.array([0, 1], dtype=np.int64), np.array([2, 0], dtype=np.int64)],
     "values": np.array([100, 200], dtype=np.int32), "dtype": np.int32,
     "grad": False, "accumulate": False},
    {"name": "values整型不匹配报错", "kind": "dtype_error",
     "input": [[1.0, 2.0], [3.0, 4.0]], "indices": [np.array([0], dtype=np.int64)],
     "values": [5], "values_dtype": np.int64, "accumulate": False},
    {"name": "values float64不匹配报错", "kind": "dtype_error",
     "input": [[1.0, 2.0], [3.0, 4.0]], "indices": [np.array([0], dtype=np.int64)],
     "values": [5.0], "values_dtype": np.float64, "accumulate": False},
]


# =============================================================================
# 测试基类：统一设备调度、子用例执行、结果记录与统计
# =============================================================================

class IndexOperTestBase(unittest.TestCase):
    """索引操作测试基类，负责用例组的通用执行与输出逻辑"""

    group_name = ""
    requires_torch = False

    # 这些 kind 的用例在 CUDA 上跳过（PyTorch CUDA 对非法索引触发 device-side assert，
    # 不抛 Python 异常且永久损坏 CUDA context，无法正常测试"是否抛异常"的行为）
    _SKIP_CUDA_KINDS = frozenset([
        "range_error", "index_ndim_error", "mask_dtype_error",
        "mask_broadcast_error", "out_grad_error", "leaf_error",
        "index_range_error", "values_shape_error", "dtype_error",
    ])

    def setUp(self):
        # 导入或依赖缺失时跳过
        if not RIEMANN_AVAILABLE:
            self.skipTest("无法导入riemann模块")
        if self.requires_torch and not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")

        # 设置随机种子以确保结果可重复
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch.set_default_dtype(torch.float32)

        # 获取当前测试方法名
        self.current_test_name = self._testMethodName
        # 如果是独立脚本运行，则开始记录用例组统计
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.group_name)
            print(f"\n{Colors.HEADER}开始测试: {self.group_name} ({self.current_test_name}){Colors.ENDC}")
            if self._testMethodDoc:
                print(f"{Colors.OKBLUE}测试描述: {self._testMethodDoc}{Colors.ENDC}")

    def tearDown(self):
        # 如果是独立脚本运行，则结束用例组统计
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.group_name}{Colors.ENDC}")

    # ------------------------- 设备与张量构造 -------------------------

    def _devices(self, require_torch=False):
        """返回本用例组需要覆盖的设备列表

        CUDA 场景需同时满足 Riemann(cupy) 与 PyTorch 的 CUDA 可用性；
        仅当 PyTorch 完全缺失时才允许执行 Riemann 单侧的 CUDA 用例。
        """
        if not CUDA_AVAILABLE:
            return ["cpu"]
        if TORCH_AVAILABLE:
            if not TORCH_CUDA_AVAILABLE:
                return ["cpu"]
        elif require_torch:
            return ["cpu"]
        return ["cpu", "cuda"]

    @staticmethod
    def _ensure_placement(tensor_obj, device, label):
        """构造后立即确认张量确实位于目标设备

        CUDA 用例中若框架静默回落到 CPU，后续数值比较仍可能通过，
        因此这里显式校验，避免 CUDA 场景出现假通过。
        """
        if device == "cuda" and not tensor_obj.is_cuda:
            raise AssertionError(f"{label}未放置在CUDA设备上（实际设备: {tensor_obj.device}）")
        return tensor_obj

    @classmethod
    def _make_rm_float(cls, np_array, device, requires_grad=True):
        return cls._ensure_placement(
            rm_tensor(np_array, requires_grad=requires_grad, device=device),
            device, "Riemann张量"
        )

    @classmethod
    def _make_rm_long(cls, np_array, device):
        return cls._ensure_placement(
            rm_tensor(np_array, device=device),
            device, "Riemann索引张量"
        )

    @classmethod
    def _make_rm_mask(cls, np_array, device):
        return cls._ensure_placement(
            rm_tensor(np.asarray(np_array, dtype=np.bool_), device=device),
            device, "Riemann掩码张量"
        )

    @classmethod
    def _make_torch_bool(cls, np_array, device):
        return cls._ensure_placement(
            torch.tensor(np.asarray(np_array, dtype=np.bool_), dtype=torch.bool, device=device),
            device, "PyTorch掩码张量"
        )

    def _make_input_pair(self, np_input, device, dtype=np.float32, requires_grad=True):
        """按 dtype 构造 Riemann/PyTorch 输入对；整型/布尔输入不参与梯度"""
        dtype = np.dtype(dtype)
        data = np.asarray(np_input)
        if dtype == np.float32:
            rm_input = self._make_rm_float(data.astype(np.float32), device, requires_grad=requires_grad)
            torch_input = (
                self._make_torch_float(data.astype(np.float32), device, requires_grad=requires_grad)
                if TORCH_AVAILABLE else None
            )
            return rm_input, torch_input, bool(requires_grad)

        if dtype == np.float64:
            rm_input = self._ensure_placement(
                rm_tensor(data.astype(np.float64), requires_grad=bool(requires_grad), device=device),
                device, "Riemann输入张量"
            )
            torch_input = (
                torch.tensor(
                    data.astype(np.float64), dtype=torch.float64,
                    requires_grad=bool(requires_grad), device=device
                ) if TORCH_AVAILABLE else None
            )
            return rm_input, torch_input, bool(requires_grad)

        if dtype == np.bool_:
            rm_input = self._ensure_placement(
                rm_tensor(data.astype(np.bool_), device=device), device, "Riemann布尔输入张量"
            )
            torch_input = (
                torch.tensor(data.astype(np.bool_), dtype=torch.bool, device=device)
                if TORCH_AVAILABLE else None
            )
            return rm_input, torch_input, False

        torch_int_dtype = torch.int32 if dtype == np.int32 else torch.long
        rm_input = self._ensure_placement(
            rm_tensor(data.astype(dtype), device=device), device, "Riemann输入张量"
        )
        torch_input = (
            torch.tensor(data.astype(dtype), dtype=torch_int_dtype, device=device)
            if TORCH_AVAILABLE else None
        )
        return rm_input, torch_input, False

    @classmethod
    def _make_torch_float(cls, np_array, device, requires_grad=True):
        return cls._ensure_placement(
            torch.tensor(np_array, requires_grad=requires_grad, dtype=torch.float32, device=device),
            device, "PyTorch张量"
        )

    @classmethod
    def _make_torch_long(cls, np_array, device):
        return cls._ensure_placement(
            torch.tensor(np_array, dtype=torch.long, device=device),
            device, "PyTorch索引张量"
        )

    # ------------------------- 比较与结果记录 -------------------------

    def _check_tensors(self, failures, label, rm_result, torch_result):
        """比较Riemann与PyTorch结果，失败信息写入 failures"""
        if not TORCH_AVAILABLE:
            return
        if not compare_values(rm_result, torch_result):
            failures.append(f"{label}比较失败")

    def _check_backward(self, failures, rm_output, torch_output, grad_pairs):
        """对输出求和后反向传播，并比较指定的梯度对"""
        if not TORCH_AVAILABLE:
            return
        rm_output.sum().backward()
        torch_output.sum().backward()
        for label, rm_input, torch_input in grad_pairs:
            if not compare_values(rm_input.grad, torch_input.grad):
                failures.append(f"{label}梯度比较失败")

    def _check_both_raise(self, failures, label, rm_call, torch_call):
        """校验 Riemann 与 PyTorch 在非法输入下均抛出异常"""
        rm_raised = False
        try:
            rm_call()
        except Exception:
            rm_raised = True
        if not rm_raised:
            failures.append(f"{label}: Riemann未抛出异常")

        if TORCH_AVAILABLE:
            torch_raised = False
            try:
                torch_call()
            except Exception:
                torch_raised = True
            if not torch_raised:
                failures.append(f"{label}: PyTorch未抛出异常")

    def _check_rm_device(self, failures, device, rm_obj, label):
        """检查Riemann侧计算结果（含梯度）仍位于目标设备"""
        if device == "cuda" and (rm_obj is None or not rm_obj.is_cuda):
            failures.append(f"{label}未在CUDA设备上计算")

    def _report(self, name, passed, failures, elapsed):
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result(name, passed, failures if failures else None)
            status = "通过" if passed else "失败"
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            print(f"  {name} - {status_color}{status}{Colors.ENDC} ({elapsed:.4f}秒)")
            for detail in failures:
                print(f"    - {detail}")

    def _run_group(self, cases, prepare, execute, devices):
        """通用用例组执行器：遍历场景与设备，记录每个子用例结果"""
        if IS_RUNNING_AS_SCRIPT:
            print(f"{Colors.OKBLUE}用例组: {self.group_name} | 场景数: {len(cases)} | 设备: {', '.join(devices)}{Colors.ENDC}")

        total = 0
        passed_count = 0

        for case in cases:
            payload = prepare(case) if prepare else case
            # error 类型用例在 CUDA 上跳过（PyTorch CUDA 非法索引触发 device-side assert）
            case_devices = devices
            if case.get("kind") in self._SKIP_CUDA_KINDS and "cuda" in devices:
                case_devices = [d for d in devices if d != "cuda"]
            for device in case_devices:
                subcase_name = f"{case['name']} [{device.upper()}]"
                start_time = time.time()
                try:
                    failures = execute(case, payload, device) or []
                except Exception as e:
                    failures = [f"异常: {e}"]
                elapsed = time.time() - start_time

                total += 1
                passed = not failures
                if passed:
                    passed_count += 1
                self._report(subcase_name, passed, failures, elapsed)

        if IS_RUNNING_AS_SCRIPT:
            color = Colors.OKGREEN if passed_count == total else Colors.FAIL
            print(f"{color}{self.group_name} 子用例结果: {passed_count}/{total} 通过{Colors.ENDC}")

        self.assertTrue(
            passed_count == total,
            f"{self.group_name} 存在失败子用例: {total - passed_count}/{total}"
        )


# =============================================================================
# getitem 用例组
# =============================================================================

class TestGetitemOperations(IndexOperTestBase):
    """getitem 类索引操作用例组"""

    group_name = "getitem 索引操作"

    def test_getitem_index_operations(self):
        """getitem 类：各类索引、视图链索引与二阶导数场景"""
        self._run_group(
            GETITEM_CASES,
            self._prepare_getitem_input,
            self._execute_getitem_case,
            self._devices(require_torch=True)
        )

    @staticmethod
    def _prepare_getitem_input(case):
        """按用例参数生成输入数据（固定数据或随机数据）"""
        if "data" in case:
            return np.array(case["data"], dtype=np.float32)
        return np.random.randn(*case["shape"]).astype(np.float32)

    def _execute_getitem_case(self, case, input_np, device):
        failures = []
        kind = case.get("kind", "getitem")

        if kind == "hessian":
            failures = self._check_getitem_hessian(input_np, device)
        elif kind == "view":
            failures = self._check_view_indexing(input_np, device)
        else:
            index = input_np > 0 if case.get("mask") else case["index"]
            failures = self._check_getitem_grad(input_np, index, device)
        return failures

    def _check_getitem_grad(self, input_np, index, device):
        """索引操作的一阶梯度：Riemann与PyTorch结果比较"""
        failures = []
        rm_x = self._make_rm_float(input_np, device)
        rm_indexed = rm_x[index]
        # 确保只对标量调用backward()
        if rm_indexed.ndim > 0:
            rm_indexed.sum().backward()
        else:
            rm_indexed.backward()

        self._check_rm_device(failures, device, rm_indexed, "索引结果")
        self._check_rm_device(failures, device, rm_x.grad, "梯度")

        if TORCH_AVAILABLE:
            torch_x = self._make_torch_float(input_np, device)
            torch_indexed = torch_x[index]
            if len(torch_indexed.shape) > 0:
                torch_indexed.sum().backward()
            else:
                torch_indexed.backward()
            self._check_tensors(failures, "值", rm_x.grad, torch_x.grad)
        return failures

    def _check_view_indexing(self, input_np, device):
        """3级视图上的索引：梯度经过多层视图回传"""
        failures = []
        rm_x = self._make_rm_float(input_np, device)
        rm_view1 = rm_x[:, 1:4, :]
        rm_view2 = rm_view1[1:3, :, 1:4]
        rm_view3 = rm_view2[:, 1:3, :]
        rm_view3[0, 0, 0].backward()

        self._check_rm_device(failures, device, rm_view3, "视图结果")
        self._check_rm_device(failures, device, rm_x.grad, "梯度")

        if TORCH_AVAILABLE:
            torch_x = self._make_torch_float(input_np, device)
            torch_view1 = torch_x[:, 1:4, :]
            torch_view2 = torch_view1[1:3, :, 1:4]
            torch_view3 = torch_view2[:, 1:3, :]
            torch_view3[0, 0, 0].backward()
            self._check_tensors(failures, "值", rm_x.grad, torch_x.grad)
        return failures

    def _check_getitem_hessian(self, input_np, device):
        """索引操作的二阶导数（Hessian矩阵）"""
        failures = []

        def f(x):
            return x[0] ** 2. + x[1] ** 3.

        rm_x = self._make_rm_float(input_np, device)
        rm_hessian_result = rm_hessian(f, rm_x)
        self._check_rm_device(failures, device, rm_hessian_result, "二阶导数结果")

        if TORCH_AVAILABLE:
            torch_x = self._make_torch_float(input_np, device)
            torch_hessian_result = torch.autograd.functional.hessian(f, torch_x)
            self._check_tensors(failures, "二阶导数", rm_hessian_result, torch_hessian_result)
        return failures


# =============================================================================
# scatter 用例组
# =============================================================================

class TestScatterOperations(IndexOperTestBase):
    """scatter 类索引操作用例组"""

    group_name = "scatter 索引操作"

    def test_scatter_operations(self):
        """scatter 类：scatter / scatter_ / scatter_add / scatter_add_ / 边界场景"""
        self._run_group(
            SCATTER_CASES,
            self._prepare_scatter_payload,
            self._execute_scatter_case,
            self._devices()
        )

    @staticmethod
    def _prepare_scatter_payload(case):
        """按用例参数生成张量、索引与源数据"""
        shape = tuple(case["shape"])
        dim = case["dim"]
        normalized_dim = dim if dim >= 0 else dim + len(shape)

        if "tensor" in case:
            np_tensor = np.asarray(case["tensor"], dtype=np.float32)
        else:
            np_tensor = np.random.randn(*shape).astype(np.float32)

        if "index" in case:
            np_index = np.asarray(case["index"], dtype=np.int64)
        elif case.get("same_indices"):
            np_index = np.zeros(shape, dtype=np.int64)
        elif shape[normalized_dim] == 0:
            np_index = np.zeros(shape, dtype=np.int64)
        else:
            np_index = np.random.randint(0, shape[normalized_dim], size=shape).astype(np.int64)

        if "src" in case:
            np_src = np.asarray(case["src"], dtype=np.float32)
        else:
            np_src = np.random.randn(*shape).astype(np.float32)

        return np_tensor, np_index, np_src

    def _execute_scatter_case(self, case, payload, device):
        np_tensor, np_index, np_src = payload
        dim = case["dim"]
        op = case["op"]

        if op == "scatter_src":
            return self._check_scatter_src(np_tensor, np_index, np_src, dim, device)
        if op == "scatter_value":
            return self._check_scatter_value(np_tensor, np_index, dim, case["value"], device)
        if op == "scatter_inplace":
            return self._check_scatter_inplace(np_tensor, np_index, np_src, dim, device)
        if op == "scatter_add_inplace":
            return self._check_scatter_add_inplace(np_tensor, np_index, np_src, dim, device)
        if op == "scatter_add":
            return self._check_scatter_add(np_tensor, np_index, np_src, dim, device)
        if op == "scatter_add_scalar":
            return self._check_scatter_add_scalar(np_tensor, np_index, dim, case["value"], device)
        if op == "direct_indices":
            return self._check_direct_indices(np_tensor, np_index, np_src, dim, case, device)
        # boundary 与 scatter(src) 的校验方式一致
        return self._check_scatter_src(np_tensor, np_index, np_src, dim, device)

    def _check_scatter_src(self, np_tensor, np_index, np_src, dim, device):
        """scatter(src)：比较函数值与 self/src 梯度"""
        failures = []
        rm_index = self._make_rm_long(np_index, device)
        rm_tensor = self._make_rm_float(np_tensor, device)
        rm_src = self._make_rm_float(np_src, device)
        rm_result = rm_tensor.scatter(dim, rm_index, rm_src)
        self._check_rm_device(failures, device, rm_result, "scatter结果")

        if TORCH_AVAILABLE:
            torch_tensor = self._make_torch_float(np_tensor, device)
            torch_index = self._make_torch_long(np_index, device)
            torch_src = self._make_torch_float(np_src, device)
            torch_result = torch_tensor.scatter(dim, torch_index, torch_src)
            self._check_tensors(failures, "函数值", rm_result, torch_result)
            self._check_backward(
                failures, rm_result, torch_result,
                [("self", rm_tensor, torch_tensor), ("src", rm_src, torch_src)]
            )
        return failures

    def _check_scatter_value(self, np_tensor, np_index, dim, value, device):
        """scatter(value)：使用标量值写入，比较函数值与 self 梯度"""
        failures = []
        rm_index = self._make_rm_long(np_index, device)
        rm_tensor = self._make_rm_float(np_tensor, device)
        rm_result = rm_tensor.scatter(dim, rm_index, value=value)
        self._check_rm_device(failures, device, rm_result, "scatter结果")

        if TORCH_AVAILABLE:
            torch_tensor = self._make_torch_float(np_tensor, device)
            torch_index = self._make_torch_long(np_index, device)
            torch_result = torch_tensor.scatter(dim, torch_index, value=value)
            self._check_tensors(failures, "函数值", rm_result, torch_result)
            self._check_backward(
                failures, rm_result, torch_result,
                [("self", rm_tensor, torch_tensor)]
            )
        return failures

    def _check_scatter_inplace(self, np_tensor, np_index, np_src, dim, device):
        """scatter_（原地操作）：比较函数值与 src 梯度"""
        failures = []
        rm_index = self._make_rm_long(np_index, device)
        rm_tensor = self._make_rm_float(np_tensor, device, requires_grad=False)
        rm_src = self._make_rm_float(np_src, device)
        rm_result = rm_tensor.clone().scatter_(dim, rm_index, rm_src)
        self._check_rm_device(failures, device, rm_result, "scatter_结果")

        if TORCH_AVAILABLE:
            torch_tensor = self._make_torch_float(np_tensor, device, requires_grad=False)
            torch_index = self._make_torch_long(np_index, device)
            torch_src = self._make_torch_float(np_src, device)
            torch_result = torch_tensor.clone().scatter_(dim, torch_index, torch_src)
            self._check_tensors(failures, "函数值", rm_result, torch_result)
            self._check_backward(
                failures, rm_result, torch_result,
                [("src", rm_src, torch_src)]
            )
        return failures

    def _check_scatter_add_inplace(self, np_tensor, np_index, np_src, dim, device):
        """scatter_add_（原地累加）：比较函数值与 self/src 梯度"""
        failures = []
        rm_index = self._make_rm_long(np_index, device)
        rm_tensor = self._make_rm_float(np_tensor, device)
        rm_src = self._make_rm_float(np_src, device)
        rm_result = rm_tensor.clone().scatter_add_(dim, rm_index, rm_src)
        self._check_rm_device(failures, device, rm_result, "scatter_add_结果")

        if TORCH_AVAILABLE:
            torch_tensor = self._make_torch_float(np_tensor, device)
            torch_index = self._make_torch_long(np_index, device)
            torch_src = self._make_torch_float(np_src, device)
            torch_result = torch_tensor.clone().scatter_add_(dim, torch_index, torch_src)
            self._check_tensors(failures, "函数值", rm_result, torch_result)
            self._check_backward(
                failures, rm_result, torch_result,
                [("self", rm_tensor, torch_tensor), ("src", rm_src, torch_src)]
            )
        return failures

    def _check_scatter_add(self, np_tensor, np_index, np_src, dim, device):
        """scatter_add（非原地累加）：额外验证原张量未被修改"""
        failures = []
        rm_index = self._make_rm_long(np_index, device)
        rm_tensor = self._make_rm_float(np_tensor, device)
        rm_tensor_original = rm_tensor.clone()
        rm_src = self._make_rm_float(np_src, device)
        rm_result = rm_tensor.scatter_add(dim, rm_index, rm_src)
        self._check_rm_device(failures, device, rm_result, "scatter_add结果")

        if not compare_values(rm_tensor, rm_tensor_original):
            failures.append("非原地操作验证失败")

        if TORCH_AVAILABLE:
            torch_tensor = self._make_torch_float(np_tensor, device)
            torch_index = self._make_torch_long(np_index, device)
            torch_src = self._make_torch_float(np_src, device)
            # PyTorch无非原地scatter_add，使用clone()+scatter_add_模拟
            torch_result = torch_tensor.clone().scatter_add_(dim, torch_index, torch_src)
            self._check_tensors(failures, "函数值", rm_result, torch_result)
            self._check_backward(
                failures, rm_result, torch_result,
                [("self", rm_tensor, torch_tensor), ("src", rm_src, torch_src)]
            )
        return failures

    def _check_scatter_add_scalar(self, np_tensor, np_index, dim, value, device):
        """scatter_add（标量src）：Riemann自动构造src，PyTorch手动构造同形状src"""
        failures = []
        rm_index = self._make_rm_long(np_index, device)
        rm_tensor = self._make_rm_float(np_tensor, device)
        rm_tensor_original = rm_tensor.clone()
        rm_result = rm_tensor.scatter_add(dim, rm_index, value)
        self._check_rm_device(failures, device, rm_result, "scatter_add结果")

        if not compare_values(rm_tensor, rm_tensor_original):
            failures.append("非原地操作验证失败")

        if TORCH_AVAILABLE:
            torch_tensor = self._make_torch_float(np_tensor, device)
            torch_index = self._make_torch_long(np_index, device)
            torch_src = torch.full(np_tensor.shape, value, dtype=torch.float32, device=device)
            torch_result = torch_tensor.clone().scatter_add_(dim, torch_index, torch_src)
            self._check_tensors(failures, "函数值", rm_result, torch_result)
            self._check_backward(
                failures, rm_result, torch_result,
                [("self", rm_tensor, torch_tensor)]
            )
        return failures

    def _check_direct_indices(self, np_tensor, np_index, np_src, dim, case, device):
        """校验 scatter 内部索引换算（_compute_to_direct_indices）的正确性

        该内部函数只负责把间接索引换算为直接索引（纯 numpy 索引计算，
        与自动微分无关），因此这里只校验形状与结果值；梯度传播已由
        scatter(src) 等用例覆盖，且重复索引下的梯度语义并不确定。
        """
        failures = []
        rm_index = self._make_rm_long(np_index, device)
        rm_tensor = self._make_rm_float(np_tensor, device, requires_grad=False)
        rm_src = self._make_rm_float(np_src, device, requires_grad=False)
        rm_result = rm_tensor.scatter(dim, rm_index, rm_src)
        self._check_rm_device(failures, device, rm_result, "scatter结果")

        expected_shape = tuple(case["shape"])
        if tuple(rm_result.shape) != expected_shape:
            failures.append(f"形状不匹配: 期望{expected_shape}, 得到{tuple(rm_result.shape)}")

        if "expected" in case:
            rm_data = rm_result.detach().cpu().numpy()
            if not np.allclose(rm_data, case["expected"]):
                failures.append("结果验证失败")
        return failures


# =============================================================================
# gather 用例组
# =============================================================================

class TestGatherOperations(IndexOperTestBase):
    """gather 类索引操作用例组"""

    group_name = "gather 索引操作"
    requires_torch = True

    _DTYPE_ALIASES = {
        'torch.long': 'int64', 'long': 'int64', 'int64': 'int64',
        'torch.float32': 'float32', 'float': 'float32', 'float32': 'float32',
        'torch.float64': 'float64', 'double': 'float64', 'float64': 'float64',
    }

    def test_gather_operations(self):
        """gather 类：基础/边界/广播/高级/反向传播/实际应用场景"""
        self._run_group(
            GATHER_CASES,
            None,
            self._execute_gather_case,
            self._devices(require_torch=True)
        )

    @classmethod
    def _normalize_dtype(cls, dtype_str):
        # 去掉 torch./numpy. 前缀后再按别名表归一化
        normalized = dtype_str.replace('torch.', '').replace('numpy.', '')
        return cls._DTYPE_ALIASES.get(normalized, normalized)

    def _check_gather_tensors(self, failures, label, rm_result, torch_result):
        """比较形状、数据类型与数值"""
        if tuple(rm_result.shape) != tuple(torch_result.shape):
            failures.append(
                f"{label}形状不匹配: rm={tuple(rm_result.shape)}, torch={tuple(torch_result.shape)}"
            )
            return
        rm_dtype = self._normalize_dtype(str(rm_result.dtype))
        torch_dtype = self._normalize_dtype(str(torch_result.dtype))
        if rm_dtype != torch_dtype:
            failures.append(f"{label}数据类型不匹配: rm={rm_dtype}, torch={torch_dtype}")
        if not compare_values(rm_result, torch_result):
            failures.append(f"{label}值不匹配")

    def _execute_gather_case(self, case, payload, device):
        if case.get("kind") == "complex_graph":
            return self._check_complex_graph(case, device)
        return self._check_gather(case, device)

    def _check_gather(self, case, device):
        failures = []
        need_grad = bool(case.get("grad"))
        rm_input = self._ensure_placement(
            rm_tensor(case["input"], requires_grad=need_grad, device=device),
            device, "gather输入张量"
        )
        rm_index = self._make_rm_long(case["index"], device)
        rm_output = rm_input.gather(dim=case["dim"], index=rm_index)
        self._check_rm_device(failures, device, rm_output, "gather结果")

        if TORCH_AVAILABLE:
            torch_input = torch.tensor(case["input"], requires_grad=need_grad, device=device)
            torch_index = self._make_torch_long(case["index"], device)
            torch_output = torch_input.gather(dim=case["dim"], index=torch_index)
            self._check_gather_tensors(failures, "函数值", rm_output, torch_output)

            if need_grad:
                rm_grad = grad(rm_output.sum(), rm_input)[0]
                self._check_rm_device(failures, device, rm_grad, "gather梯度")
                torch_output.sum().backward()
                self._check_gather_tensors(failures, "一阶梯度", rm_grad, torch_input.grad)
        return failures

    def _check_complex_graph(self, case, device):
        failures = []
        rm_index0 = self._make_rm_long(case["indices"][0], device)
        rm_index1 = self._make_rm_long(case["indices"][1], device)
        rm_input = self._ensure_placement(
            rm_tensor(case["input"], requires_grad=True, device=device),
            device, "gather输入张量"
        )

        rm_temp = rm_input.gather(dim=case["dims"][0], index=rm_index0)
        rm_loss = rm_temp.gather(dim=case["dims"][1], index=rm_index1).sum()
        rm_grad = grad(rm_loss, rm_input)[0]
        self._check_rm_device(failures, device, rm_temp, "gather中间结果")
        self._check_rm_device(failures, device, rm_grad, "复杂计算图梯度")

        if TORCH_AVAILABLE:
            torch_index0 = self._make_torch_long(case["indices"][0], device)
            torch_index1 = self._make_torch_long(case["indices"][1], device)
            torch_input = torch.tensor(case["input"], requires_grad=True, device=device)

            torch_temp = torch_input.gather(dim=case["dims"][0], index=torch_index0)
            torch_loss = torch_temp.gather(dim=case["dims"][1], index=torch_index1).sum()
            torch_loss.backward()
            self._check_gather_tensors(failures, "复杂计算图梯度", rm_grad, torch_input.grad)
        return failures


# =============================================================================
# index_select 用例组
# =============================================================================

class TestIndexSelectOperations(IndexOperTestBase):
    """index_select 类索引操作用例组"""

    group_name = "index_select 索引操作"
    requires_torch = True

    def test_index_select_operations(self):
        """index_select 类：多维度、负维度、空索引、重复索引与整型输入"""
        self._run_group(
            INDEX_SELECT_CASES,
            None,
            self._execute_index_select_case,
            self._devices(require_torch=True)
        )

    def _execute_index_select_case(self, case, payload, device):
        failures = []
        rm_input, torch_input, need_grad = self._make_input_pair(
            case["input"], device, case.get("dtype", np.float32), case.get("grad", False)
        )
        rm_index = self._make_rm_long(np.asarray(case["index"], dtype=np.int64), device)
        rm_output = rm.index_select(rm_input, case["dim"], rm_index)
        self._check_rm_device(failures, device, rm_output, "index_select结果")

        if TORCH_AVAILABLE:
            torch_index = self._make_torch_long(np.asarray(case["index"], dtype=np.int64), device)
            torch_output = torch.index_select(torch_input, case["dim"], torch_index)
            self._check_tensors(failures, "函数值", rm_output, torch_output)
            if need_grad:
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("input", rm_input, torch_input)]
                )
        return failures

    def test_index_select_edge_cases(self):
        """index_select 边界：out 参数、错误路径与非连续输入"""
        self._run_group(
            INDEX_SELECT_EDGE_CASES,
            None,
            self._execute_index_select_edge_case,
            self._devices(require_torch=True)
        )

    def _execute_index_select_edge_case(self, case, payload, device):
        failures = []
        kind = case["kind"]
        dim = case["dim"]
        np_input = np.asarray(case["input"], dtype=np.float32)
        dim_norm = dim if dim >= 0 else dim + np_input.ndim
        index_np = np.asarray(case["index"], dtype=np.int64)
        out_shape = list(np_input.shape)
        out_shape[dim_norm] = len(index_np)

        if kind == "non_contiguous":
            rm_base = self._make_rm_float(np_input, device)
            rm_input = rm_base.transpose(0, 1)
            rm_index = self._make_rm_long(index_np, device)
            rm_output = rm.index_select(rm_input, dim, rm_index)
            self._check_rm_device(failures, device, rm_output, "index_select结果")

            if TORCH_AVAILABLE:
                torch_base = self._make_torch_float(np_input, device)
                torch_input = torch_base.transpose(0, 1)
                torch_index = self._make_torch_long(index_np, device)
                torch_output = torch.index_select(torch_input, dim, torch_index)
                self._check_tensors(failures, "函数值", rm_output, torch_output)
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("base", rm_base, torch_base)]
                )
            return failures

        need_grad = kind == "out_grad_error"
        rm_input, torch_input, _ = self._make_input_pair(
            np_input, device, requires_grad=need_grad
        )
        rm_index = self._make_rm_long(index_np, device)

        if kind == "out":
            rm_out = self._ensure_placement(
                rm_tensor(np.zeros(out_shape, dtype=np.float32), device=device),
                device, "index_select输出张量"
            )
            rm_output = rm.index_select(rm_input, dim, rm_index, out=rm_out)
            if rm_output is not rm_out:
                failures.append("index_select未返回传入的out张量")

            if TORCH_AVAILABLE:
                torch_index = self._make_torch_long(index_np, device)
                torch_out = self._ensure_placement(
                    torch.zeros(out_shape, dtype=torch.float32, device=device),
                    device, "PyTorch输出张量"
                )
                torch_output = torch.index_select(torch_input, dim, torch_index, out=torch_out)
                self._check_tensors(failures, "out函数值", rm_output, torch_output)
            return failures

        if kind in ("range_error", "index_ndim_error"):
            self._check_both_raise(
                failures, "index_select非法索引",
                lambda: rm.index_select(rm_input, dim, rm_index),
                lambda: torch.index_select(
                    torch_input, dim, self._make_torch_long(index_np, device)
                ) if TORCH_AVAILABLE else None
            )
            return failures

        # out_grad_error：输入需要梯度时 out= 形式应被拒绝
        rm_out = self._ensure_placement(
            rm_tensor(np.zeros(out_shape, dtype=np.float32), device=device),
            device, "index_select输出张量"
        )
        self._check_both_raise(
            failures, "requires_grad输入与out",
            lambda: rm.index_select(rm_input, dim, rm_index, out=rm_out),
            lambda: torch.index_select(
                torch_input, dim, self._make_torch_long(index_np, device),
                out=torch.zeros(out_shape, dtype=torch.float32, device=device)
            ) if TORCH_AVAILABLE else None
        )
        return failures

    def test_index_select_medium_cases(self):
        """index_select 中级：索引 dtype、输入 dtype、高维与空维度"""
        self._run_group(
            INDEX_SELECT_MEDIUM_CASES,
            None,
            self._execute_index_select_medium_case,
            self._devices(require_torch=True)
        )

    def _execute_index_select_medium_case(self, case, payload, device):
        failures = []
        if case.get("kind") == "range_error":
            np_input = np.asarray(case["input"], dtype=np.float32)
            index_np = np.asarray(case["index"], dtype=np.int64)
            rm_input, torch_input, _ = self._make_input_pair(np_input, device)
            rm_index = self._make_rm_long(index_np, device)
            if TORCH_AVAILABLE:
                torch_index = self._make_torch_long(index_np, device)
                self._check_both_raise(
                    failures, "index_select空维度非法索引",
                    lambda: rm.index_select(rm_input, case["dim"], rm_index),
                    lambda: torch.index_select(torch_input, case["dim"], torch_index)
                )
            else:
                try:
                    rm.index_select(rm_input, case["dim"], rm_index)
                except Exception:
                    pass
                else:
                    failures.append("index_select空维度非法索引: Riemann未抛出异常")
            return failures

        dtype = case.get("dtype", np.float32)
        index_dtype = case.get("index_dtype", np.int64)
        rm_input, torch_input, need_grad = self._make_input_pair(
            case["input"], device, dtype, case.get("grad", False)
        )
        index_np = np.asarray(case["index"], dtype=index_dtype)
        rm_index = self._make_rm_long(index_np, device)
        rm_output = rm.index_select(rm_input, case["dim"], rm_index)
        self._check_rm_device(failures, device, rm_output, "index_select结果")

        if TORCH_AVAILABLE:
            torch_index = self._ensure_placement(
                torch.tensor(
                    index_np,
                    dtype=torch.int32 if index_dtype == np.int32 else torch.long,
                    device=device
                ),
                device, "PyTorch索引张量"
            )
            torch_output = torch.index_select(torch_input, case["dim"], torch_index)
            self._check_tensors(failures, "函数值", rm_output, torch_output)
            if need_grad:
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("input", rm_input, torch_input)]
                )
        return failures


# =============================================================================
# masked_select 用例组
# =============================================================================

class TestMaskedSelectOperations(IndexOperTestBase):
    """masked_select 类索引操作用例组"""

    group_name = "masked_select 索引操作"
    requires_torch = True

    def test_masked_select_operations(self):
        """masked_select 类：同形状掩码、广播掩码、空结果、0维输入与整型输入"""
        self._run_group(
            MASKED_SELECT_CASES,
            None,
            self._execute_masked_select_case,
            self._devices(require_torch=True)
        )

    def _execute_masked_select_case(self, case, payload, device):
        failures = []
        rm_input, torch_input, need_grad = self._make_input_pair(
            case["input"], device, case.get("dtype", np.float32), case.get("grad", False)
        )
        rm_mask = self._make_rm_mask(case["mask"], device)
        rm_output = rm.masked_select(rm_input, rm_mask)
        self._check_rm_device(failures, device, rm_output, "masked_select结果")

        if TORCH_AVAILABLE:
            torch_mask = self._make_torch_bool(case["mask"], device)
            torch_output = torch.masked_select(torch_input, torch_mask)
            self._check_tensors(failures, "函数值", rm_output, torch_output)
            if need_grad:
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("input", rm_input, torch_input)]
                )
        return failures

    def test_masked_select_edge_cases(self):
        """masked_select 边界：out 参数、非法掩码与非连续输入"""
        self._run_group(
            MASKED_SELECT_EDGE_CASES,
            None,
            self._execute_masked_select_edge_case,
            self._devices(require_torch=True)
        )

    def _execute_masked_select_edge_case(self, case, payload, device):
        failures = []
        kind = case["kind"]
        np_input = np.asarray(case["input"], dtype=np.float32)
        np_mask = np.asarray(case["mask"])

        if kind == "non_contiguous":
            rm_base = self._make_rm_float(np_input, device)
            rm_input = rm_base.transpose(0, 1)
            rm_mask = self._make_rm_mask(np_mask.T, device)
            rm_output = rm.masked_select(rm_input, rm_mask)
            self._check_rm_device(failures, device, rm_output, "masked_select结果")

            if TORCH_AVAILABLE:
                torch_base = self._make_torch_float(np_input, device)
                torch_input = torch_base.transpose(0, 1)
                torch_mask = self._make_torch_bool(np_mask.T, device)
                torch_output = torch.masked_select(torch_input, torch_mask)
                self._check_tensors(failures, "函数值", rm_output, torch_output)
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("base", rm_base, torch_base)]
                )
            return failures

        need_grad = kind == "out_grad_error"
        rm_input, torch_input, _ = self._make_input_pair(
            np_input, device, requires_grad=need_grad
        )

        if np_mask.dtype == np.bool_:
            rm_mask = self._make_rm_mask(np_mask, device)
            torch_mask = self._make_torch_bool(np_mask, device) if TORCH_AVAILABLE else None
        else:
            rm_mask = self._ensure_placement(
                rm_tensor(np_mask, device=device), device, "非布尔掩码张量"
            )
            torch_mask = (
                torch.tensor(np_mask, device=device) if TORCH_AVAILABLE else None
            )

        n_selected = int(np.count_nonzero(np_mask))

        if kind == "out":
            rm_out = self._ensure_placement(
                rm_tensor(np.zeros((n_selected,), dtype=np.float32), device=device),
                device, "masked_select输出张量"
            )
            rm_output = rm.masked_select(rm_input, rm_mask, out=rm_out)
            if rm_output is not rm_out:
                failures.append("masked_select未返回传入的out张量")

            if TORCH_AVAILABLE:
                torch_out = self._ensure_placement(
                    torch.zeros((n_selected,), dtype=torch.float32, device=device),
                    device, "PyTorch输出张量"
                )
                torch_output = torch.masked_select(torch_input, torch_mask, out=torch_out)
                self._check_tensors(failures, "out函数值", rm_output, torch_output)
            return failures

        if kind in ("mask_dtype_error", "mask_broadcast_error"):
            self._check_both_raise(
                failures, "masked_select非法掩码",
                lambda: rm.masked_select(rm_input, rm_mask),
                lambda: torch.masked_select(torch_input, torch_mask)
                if TORCH_AVAILABLE else None
            )
            return failures

        # out_grad_error：输入需要梯度时 out= 形式应被拒绝
        rm_out = self._ensure_placement(
            rm_tensor(np.zeros((n_selected,), dtype=np.float32), device=device),
            device, "masked_select输出张量"
        )
        self._check_both_raise(
            failures, "requires_grad输入与out",
            lambda: rm.masked_select(rm_input, rm_mask, out=rm_out),
            lambda: torch.masked_select(
                torch_input, torch_mask,
                out=torch.zeros((n_selected,), dtype=torch.float32, device=device)
            ) if TORCH_AVAILABLE else None
        )
        return failures

    def test_masked_select_medium_cases(self):
        """masked_select 中级：输入 dtype、高维广播与空输入"""
        self._run_group(
            MASKED_SELECT_MEDIUM_CASES,
            None,
            self._execute_masked_select_medium_case,
            self._devices(require_torch=True)
        )

    def _execute_masked_select_medium_case(self, case, payload, device):
        failures = []
        dtype = case.get("dtype", np.float32)
        rm_input, torch_input, need_grad = self._make_input_pair(
            case["input"], device, dtype, case.get("grad", False)
        )
        rm_mask = self._make_rm_mask(case["mask"], device)
        rm_output = rm.masked_select(rm_input, rm_mask)
        self._check_rm_device(failures, device, rm_output, "masked_select结果")

        if TORCH_AVAILABLE:
            torch_mask = self._make_torch_bool(case["mask"], device)
            torch_output = torch.masked_select(torch_input, torch_mask)
            self._check_tensors(failures, "函数值", rm_output, torch_output)
            if need_grad:
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("input", rm_input, torch_input)]
                )
        return failures


# =============================================================================
# take 用例组
# =============================================================================

class TestTakeOperations(IndexOperTestBase):
    """take 类索引操作用例组"""

    group_name = "take 索引操作"
    requires_torch = True

    def test_take_operations(self):
        """take 类：扁平化下标、多维/0维索引、空索引、重复索引与整型输入"""
        self._run_group(
            TAKE_CASES,
            None,
            self._execute_take_case,
            self._devices(require_torch=True)
        )

    def _execute_take_case(self, case, payload, device):
        failures = []
        rm_input, torch_input, need_grad = self._make_input_pair(
            case["input"], device, case.get("dtype", np.float32), case.get("grad", False)
        )
        rm_index = self._make_rm_long(np.asarray(case["index"], dtype=np.int64), device)
        rm_output = rm.take(rm_input, rm_index)
        self._check_rm_device(failures, device, rm_output, "take结果")

        if TORCH_AVAILABLE:
            torch_index = self._make_torch_long(np.asarray(case["index"], dtype=np.int64), device)
            torch_output = torch.take(torch_input, torch_index)
            self._check_tensors(failures, "函数值", rm_output, torch_output)
            if need_grad:
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("input", rm_input, torch_input)]
                )
        return failures

    def test_take_edge_cases(self):
        """take 边界：负索引、越界错误、非连续输入与2D重复索引梯度"""
        self._run_group(
            TAKE_EDGE_CASES,
            None,
            self._execute_take_edge_case,
            self._devices(require_torch=True)
        )

    def _execute_take_edge_case(self, case, payload, device):
        failures = []
        kind = case["kind"]
        np_input = np.asarray(case["input"], dtype=np.float32)
        index_np = np.asarray(case["index"], dtype=np.int64)

        if kind == "non_contiguous":
            rm_base = self._make_rm_float(np_input, device)
            rm_input = rm_base.transpose(0, 1)
            rm_index = self._make_rm_long(index_np, device)
            rm_output = rm.take(rm_input, rm_index)
            self._check_rm_device(failures, device, rm_output, "take结果")

            if TORCH_AVAILABLE:
                torch_base = self._make_torch_float(np_input, device)
                torch_input = torch_base.transpose(0, 1)
                torch_index = self._make_torch_long(index_np, device)
                torch_output = torch.take(torch_input, torch_index)
                self._check_tensors(failures, "函数值", rm_output, torch_output)
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("base", rm_base, torch_base)]
                )
            return failures

        rm_input, torch_input, need_grad = self._make_input_pair(np_input, device)
        rm_index = self._make_rm_long(index_np, device)

        if kind == "range_error":
            self._check_both_raise(
                failures, "take非法索引",
                lambda: rm.take(rm_input, rm_index),
                lambda: torch.take(
                    torch_input, self._make_torch_long(index_np, device)
                ) if TORCH_AVAILABLE else None
            )
            return failures

        rm_output = rm.take(rm_input, rm_index)
        self._check_rm_device(failures, device, rm_output, "take结果")
        if TORCH_AVAILABLE:
            torch_index = self._make_torch_long(index_np, device)
            torch_output = torch.take(torch_input, torch_index)
            self._check_tensors(failures, "函数值", rm_output, torch_output)
            self._check_backward(
                failures, rm_output, torch_output,
                [("input", rm_input, torch_input)]
            )
        return failures

    def test_take_medium_cases(self):
        """take 中级：输入 dtype、4D 输入/索引、空输入与 0 维边界"""
        self._run_group(
            TAKE_MEDIUM_CASES,
            None,
            self._execute_take_medium_case,
            self._devices(require_torch=True)
        )

    def _execute_take_medium_case(self, case, payload, device):
        failures = []
        dtype = case.get("dtype", np.float32)
        rm_input, torch_input, need_grad = self._make_input_pair(
            case["input"], device, dtype, case.get("grad", False)
        )
        index_np = np.asarray(case["index"], dtype=np.int64)
        rm_index = self._make_rm_long(index_np, device)
        rm_output = rm.take(rm_input, rm_index)
        self._check_rm_device(failures, device, rm_output, "take结果")

        if TORCH_AVAILABLE:
            torch_index = self._make_torch_long(index_np, device)
            torch_output = torch.take(torch_input, torch_index)
            self._check_tensors(failures, "函数值", rm_output, torch_output)
            if need_grad:
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("input", rm_input, torch_input)]
                )
        return failures


# =============================================================================
# index_put_ 用例组
# =============================================================================

class TestIndexPutOperations(IndexOperTestBase):
    """index_put_ 类原地索引操作用例组"""

    group_name = "index_put_ 索引操作"
    requires_torch = True

    def test_index_put_operations(self):
        """index_put_ 类：覆盖/累加、重复索引、广播值、布尔掩码与整型输入"""
        self._run_group(
            INDEX_PUT_CASES,
            None,
            self._execute_index_put_case,
            self._devices(require_torch=True)
        )

    def _execute_index_put_case(self, case, payload, device):
        failures = []
        dtype = np.dtype(case.get("dtype", np.float32))
        need_grad = bool(case.get("grad")) and np.issubdtype(dtype, np.floating)
        rm_input, torch_input, _ = self._make_input_pair(
            case["input"], device, dtype, requires_grad=False
        )
        rm_values, torch_values, _ = self._make_input_pair(
            case["values"], device, dtype, requires_grad=need_grad
        )

        if case.get("mask_indices"):
            rm_indices = (self._make_rm_mask(case["indices"][0], device),)
            torch_indices = (self._make_torch_bool(case["indices"][0], device),)
        else:
            rm_indices = tuple(
                self._make_rm_long(np.asarray(idx, dtype=np.int64), device)
                for idx in case["indices"]
            )
            torch_indices = tuple(
                self._make_torch_long(np.asarray(idx, dtype=np.int64), device)
                for idx in case["indices"]
            )

        rm_output = rm_input.clone().index_put_(
            rm_indices, rm_values, accumulate=case["accumulate"]
        )
        self._check_rm_device(failures, device, rm_output, "index_put_结果")

        if TORCH_AVAILABLE:
            torch_output = torch_input.clone().index_put_(
                torch_indices, torch_values, accumulate=case["accumulate"]
            )
            self._check_tensors(failures, "函数值", rm_output, torch_output)
            if need_grad:
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("values", rm_values, torch_values)]
                )
        return failures

    def test_index_put_edge_cases(self):
        """index_put_ 边界：索引广播、base 梯度、叶子报错、空索引与错误路径"""
        self._run_group(
            INDEX_PUT_EDGE_CASES,
            None,
            self._execute_index_put_edge_case,
            self._devices(require_torch=True)
        )

    def _execute_index_put_edge_case(self, case, payload, device):
        failures = []
        kind = case["kind"]
        np_input = np.asarray(case["input"], dtype=np.float32)
        np_values = np.asarray(case["values"], dtype=np.float32)
        rm_indices = tuple(
            self._make_rm_long(np.asarray(idx, dtype=np.int64), device)
            for idx in case["indices"]
        )
        accumulate = case["accumulate"]

        if kind == "leaf_error":
            rm_base = self._make_rm_float(np_input, device, requires_grad=True)
            rm_values = self._make_rm_float(np_values, device)
            rm_raised = False
            try:
                rm_base.index_put_(rm_indices, rm_values, accumulate=accumulate)
            except Exception:
                rm_raised = True
            if not rm_raised:
                failures.append("叶子张量原地写入: Riemann未抛出异常")

            if TORCH_AVAILABLE:
                torch_base = self._make_torch_float(np_input, device, requires_grad=True)
                torch_values = self._make_torch_float(np_values, device)
                torch_indices = tuple(
                    self._make_torch_long(np.asarray(idx, dtype=np.int64), device)
                    for idx in case["indices"]
                )
                self._check_both_raise(
                    failures, "叶子张量原地写入",
                    lambda: rm_base.index_put_(rm_indices, rm_values, accumulate=accumulate),
                    lambda: torch_base.index_put_(torch_indices, torch_values, accumulate=accumulate)
                )
            return failures

        need_grad = kind in ("base_grad", "broadcast_indices", "negative")
        rm_base = self._make_rm_float(np_input, device, requires_grad=need_grad)
        rm_values = self._make_rm_float(np_values, device, requires_grad=need_grad)

        if kind in ("index_range_error", "values_shape_error"):
            rm_target = rm_base.clone()
            if TORCH_AVAILABLE:
                torch_base = self._make_torch_float(np_input, device)
                torch_values = self._make_torch_float(np_values, device)
                torch_indices = tuple(
                    self._make_torch_long(np.asarray(idx, dtype=np.int64), device)
                    for idx in case["indices"]
                )
                self._check_both_raise(
                    failures, "index_put_非法写入",
                    lambda: rm_target.index_put_(rm_indices, rm_values, accumulate=accumulate),
                    lambda: torch_base.clone().index_put_(
                        torch_indices, torch_values, accumulate=accumulate
                    )
                )
            else:
                try:
                    rm_target.index_put_(rm_indices, rm_values, accumulate=accumulate)
                except Exception:
                    pass
                else:
                    failures.append("index_put_非法写入: Riemann未抛出异常")
            return failures

        rm_output = rm_base.clone().index_put_(rm_indices, rm_values, accumulate=accumulate)
        self._check_rm_device(failures, device, rm_output, "index_put_结果")

        if TORCH_AVAILABLE:
            torch_base = self._make_torch_float(np_input, device, requires_grad=need_grad)
            torch_values = self._make_torch_float(np_values, device, requires_grad=need_grad)
            torch_indices = tuple(
                self._make_torch_long(np.asarray(idx, dtype=np.int64), device)
                for idx in case["indices"]
            )
            torch_output = torch_base.clone().index_put_(
                torch_indices, torch_values, accumulate=accumulate
            )
            self._check_tensors(failures, "函数值", rm_output, torch_output)
            if kind == "base_grad":
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("base", rm_base, torch_base), ("values", rm_values, torch_values)]
                )
            elif need_grad:
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("values", rm_values, torch_values)]
                )
        return failures

    def test_index_put_medium_cases(self):
        """index_put_ 中级：3D 三元索引、float64/int32 dtype 与 dtype 不匹配报错"""
        self._run_group(
            INDEX_PUT_MEDIUM_CASES,
            None,
            self._execute_index_put_medium_case,
            self._devices(require_torch=True)
        )

    def _execute_index_put_medium_case(self, case, payload, device):
        failures = []
        dtype = np.dtype(case.get("dtype", np.float32))
        values_dtype = np.dtype(case.get("values_dtype", dtype))
        need_grad = bool(case.get("grad")) and np.issubdtype(dtype, np.floating)

        rm_input, torch_input, _ = self._make_input_pair(
            case["input"], device, dtype, requires_grad=False
        )
        rm_values, torch_values, _ = self._make_input_pair(
            case["values"], device, values_dtype, requires_grad=need_grad
        )
        rm_indices = tuple(
            self._make_rm_long(np.asarray(idx, dtype=np.int64), device)
            for idx in case["indices"]
        )

        if case.get("kind") == "dtype_error":
            if TORCH_AVAILABLE:
                torch_indices = tuple(
                    self._make_torch_long(np.asarray(idx, dtype=np.int64), device)
                    for idx in case["indices"]
                )
                self._check_both_raise(
                    failures, "index_put_dtype不匹配",
                    lambda: rm_input.clone().index_put_(
                        rm_indices, rm_values, accumulate=case["accumulate"]
                    ),
                    lambda: torch_input.clone().index_put_(
                        torch_indices, torch_values, accumulate=case["accumulate"]
                    )
                )
            else:
                try:
                    rm_input.clone().index_put_(
                        rm_indices, rm_values, accumulate=case["accumulate"]
                    )
                except Exception:
                    pass
                else:
                    failures.append("index_put_dtype不匹配: Riemann未抛出异常")
            return failures

        rm_output = rm_input.clone().index_put_(
            rm_indices, rm_values, accumulate=case["accumulate"]
        )
        self._check_rm_device(failures, device, rm_output, "index_put_结果")

        if TORCH_AVAILABLE:
            torch_indices = tuple(
                self._make_torch_long(np.asarray(idx, dtype=np.int64), device)
                for idx in case["indices"]
            )
            torch_output = torch_input.clone().index_put_(
                torch_indices, torch_values, accumulate=case["accumulate"]
            )
            self._check_tensors(failures, "函数值", rm_output, torch_output)
            if need_grad:
                self._check_backward(
                    failures, rm_output, torch_output,
                    [("values", rm_values, torch_values)]
                )
        return failures


# 独立运行入口
if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True

    # 清屏
    clear_screen()

    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行索引操作测试（getitem / scatter / gather / index_select / masked_select / take / index_put_） ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Riemann 可用: {RIEMANN_AVAILABLE} | PyTorch 可用: {TORCH_AVAILABLE} | "
          f"Riemann CUDA 可用: {CUDA_AVAILABLE} | PyTorch CUDA 可用: {TORCH_CUDA_AVAILABLE}{Colors.ENDC}")
    print()

    # 创建测试套件（七个用例组）
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    for test_case in (
        TestGetitemOperations,
        TestScatterOperations,
        TestGatherOperations,
        TestIndexSelectOperations,
        TestMaskedSelectOperations,
        TestTakeOperations,
        TestIndexPutOperations,
    ):
        test_suite.addTests(loader.loadTestsFromTestCase(test_case))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)

    # 打印测试统计摘要
    stats.print_summary()

    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)
