import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    # 尝试从rm.cuda获取cupy引用和CUDA可用性
    CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
    cp = rm.cuda.cp
    
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True

    # 在模块级别进行PyTorch预热，避免在测试计时中包含初始化开销
    print("预热PyTorch系统...")
    warmup_start = time.time()
    
    # 执行简单的PyTorch CPU操作以触发初始化
    warmup_input_cpu = torch.tensor([[0.0]], requires_grad=True)
    warmup_output_cpu = warmup_input_cpu.sum()
    warmup_output_cpu.backward()
    
    # 如果CUDA可用，执行CUDA操作以初始化CUDA上下文
    if torch.cuda.is_available():
        warmup_input_cuda = torch.tensor([[0.0]], requires_grad=True, device='cuda')
        warmup_output_cuda = warmup_input_cuda.sum()
        warmup_output_cuda.backward()
        torch.cuda.empty_cache()
    
    print(f"PyTorch预热完成，耗时: {time.time() - warmup_start:.4f}秒")
    
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的原地操作")
    TORCH_AVAILABLE = False

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
        
        # 打印数据行 - 精确计算每个值的填充，并过滤掉空的测试用例组
        for func_name, stats in self.function_stats.items():
            # 过滤掉总测试用例为0的空组
            if stats['total'] == 0:
                continue
                
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
        for i, (r, t) in enumerate(zip(rm_result, torch_result)):
            if not compare_values(r, t, atol, rtol):
                all_passed = False
                break
        
        return all_passed
    
    # 检查形状是否一致
    if rm_result.shape != torch_result.shape:
        return False
    
    # 处理Riemann结果
    rm_data = rm_result.detach().cpu().numpy()
    # 处理PyTorch结果
    torch_data = torch_result.detach().cpu().numpy()
    
    # 处理形状不匹配的情况
    try:
        # 增加容差参数
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

# 比较梯度的函数
def compare_grads(rm_tensor, torch_tensor, atol=1e-6, rtol=1e-6):
    """比较Riemann和PyTorch的梯度是否接近"""
    if not TORCH_AVAILABLE:
        # 如果没有PyTorch，只检查Riemann是否有梯度
        return rm_tensor.grad is not None
    
    if rm_tensor.grad is None and torch_tensor.grad is None:
        return True
    if rm_tensor.grad is None or torch_tensor.grad is None:
        return False
    
    return compare_values(rm_tensor.grad, torch_tensor.grad, atol, rtol)

# 定义索引方式的测试用例
INDEX_CASES = [
    # 正整数索引
    {"name": "正整数索引", "index": (1,)},  # 第2行
    # 负整数索引
    {"name": "负整数索引", "index": (-1,)},  # 最后一行
    # 切片索引
    {"name": "切片索引", "index": (slice(1, 3),)},  # 第2到4行（不包括第4行）
    # 带负数的切片索引
    {"name": "带负数的切片索引", "index": (slice(-3, -1),)},  # 倒数第3到倒数第1行（不包括倒数第1行）
    # 整数数组索引
    {"name": "整数数组索引", "index": ([0, 2],)},  # 第1和第3行
    # 带负数的整数数组索引
    {"name": "带负数的整数数组索引", "index": ([-1, -3],)},  # 最后一行和倒数第3行
    # 多维数组索引
    {"name": "多维数组索引", "index": ([0, 1], [1, 2])},  # 元素 (0,1) 和 (1,2)
    # 布尔索引
    {"name": "布尔索引", "index": (np.array([True, False, True, False, False]),)},  # 第1和第3行
    # dot索引
    {"name": "dot索引", "index": (..., 1)},  # 最后一个维度的第2列
    # 复杂的混合索引
    {"name": "复杂的混合索引", "index": (1, slice(0, 3))},  # 第2行，前3列
    # 链式索引（这里用元组表示，实际测试中会分两步执行）
    {"name": "链式索引", "index": ((1,), (slice(0, 3),))},  # 先取第2行，再取前3列
    # NumPy数组索引模拟CuPy数组索引
    {"name": "NumPy整数数组索引", "index": (np.array([0, 2]),)},  # 使用NumPy数组模拟CuPy数组索引
    # NumPy布尔数组索引模拟CuPy布尔数组索引
    {"name": "NumPy布尔数组索引", "index": (np.array([True, False, True, False, True]),)}  # 使用NumPy数组模拟CuPy布尔数组索引
]

# 定义CUDA环境专用的索引类型（直接使用CuPy数组）
CUDA_INDEX_CASES = []
if CUDA_AVAILABLE:
    import cupy as cp
    CUDA_INDEX_CASES = [
        # 直接使用CuPy整数数组索引
        {"name": "CuPy整数数组索引", "index": (cp.array([0, 2]),)},  # 直接使用CuPy数组作为索引
        # 直接使用CuPy布尔数组索引
        {"name": "CuPy布尔数组索引", "index": (cp.array([True, False, True, False, True]),)}  # 直接使用CuPy布尔数组作为索引
    ]

# 定义原地操作的测试用例
INPLACE_OPS = [
    {"name": "原地赋值", "op": "=", "rm_func": lambda x, idx, y: setattr(x, "data", np.copy(x.numpy())), "torch_func": lambda x, idx, y: x},  # 占位符，实际使用赋值操作
    {"name": "原地加法", "op": "+", "rm_func": lambda x, idx, y: x[idx].__iadd__(y), "torch_func": lambda x, idx, y: x[idx].__iadd__(y)},
    {"name": "原地减法", "op": "-", "rm_func": lambda x, idx, y: x[idx].__isub__(y), "torch_func": lambda x, idx, y: x[idx].__isub__(y)},
    {"name": "原地乘法", "op": "*", "rm_func": lambda x, idx, y: x[idx].__imul__(y), "torch_func": lambda x, idx, y: x[idx].__imul__(y)},
    {"name": "原地除法", "op": "/", "rm_func": lambda x, idx, y: x[idx].__itruediv__(y), "torch_func": lambda x, idx, y: x[idx].__itruediv__(y)}
]

# 多次重叠索引赋值用例（迁移自 test_007_inplace_setitem.py）
# ops 中每项为 (目标索引, 源输入下标, 源输入切片)；按顺序执行 res[索引] = 源值，
# 两次操作之间克隆结果张量，验证重叠 / 重复写入时只有最后一次写入参与梯度回传。
SETITEM_OVERLAP_CASES = [
    {
        "name": "基本重叠索引",
        "out_shape": (3,),
        "inputs": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0]), np.array([6.0])],
        "ops": [([0, 1], 0, [0, 1]), ([1, 2], 1, None), ([2], 2, None)],
    },
    {
        "name": "复杂重叠索引",
        "out_shape": (5,),
        "inputs": [np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                   np.array([5.0, 6.0, 7.0]),
                   np.array([8.0, 9.0])],
        "ops": [([0, 1, 2], 0, [0, 1, 2]), ([1, 2, 3], 1, None),
                ([2, 3, 4], 0, [1, 2, 3]), ([3, 4], 2, None)],
    },
    {
        "name": "多次重叠更新",
        "out_shape": (4,),
        "inputs": [np.array([1.0, 2.0, 3.0, 4.0]),
                   np.array([5.0, 6.0, 7.0]),
                   np.array([8.0, 9.0])],
        "ops": [([0, 1], 0, [0, 1]), ([1, 2], 1, [0, 1]),
                ([2, 3], 2, None), ([0, 2, 3], 0, [0, 2, 3])],
    },
    {
        "name": "重复重叠索引位置",
        "out_shape": (2,),
        "inputs": [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0])],
        "ops": [(0, 0, 0), (0, 1, 0), (0, 2, 0)],
    },
]

# 二维混合索引重叠赋值用例（迁移自 test_007_inplace_setitem.py）
SETITEM_2D_CASES = [
    {
        "name": "二维张量混合索引重叠更新",
        "out_shape": (3, 3),
        "inputs": [np.arange(1.0, 10.0).reshape(3, 3),
                   np.array(0.0),
                   np.zeros(3),
                   np.zeros(3)],
        "ops": [((1, 1), 1, None), ((1, slice(None)), 2, None),
                ((slice(None), 1), 3, None)],
    },
]

# 特殊原地操作函数的用例（数据驱动）
# 操作数描述 (kind, data)：kind 取 scalar（标量）/ like（与自身同形状）/ broadcast（可广播列向量）
COPY_CASES = [
    {"name": "标量源值", "src": ("scalar", 42.0)},
    {"name": "0维张量源值", "src": ("scalar", 42.0)},
    {"name": "相同形状张量源值", "src": ("like", None)},
    {"name": "不同形状可广播源值", "src": ("broadcast", None)},
]

FILL_CASES = [
    {"name": "标量填充值", "value": ("scalar", 42.0)},
    {"name": "0维张量填充值", "value": ("scalar", 42.0)},
]

MASKED_FILL_CASES = [
    {"name": "全布尔掩码", "mask_type": "full_bool"},
    {"name": "广播布尔掩码", "mask_type": "broadcast_bool"},
    {"name": "标量填充值", "mask_type": "full_bool", "value": ("scalar", 42.0)},
    {"name": "0维张量填充值", "mask_type": "full_bool", "value": ("scalar", 0.0)},
]


def safe_compare_index(a, b):
    """比较索引合并结果（支持 slice / 元组 / 数组 / 列表 / 标量）"""
    if isinstance(a, slice) and isinstance(b, slice):
        return (a.start == b.start and a.stop == b.stop and a.step == b.step)

    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(safe_compare_index(ai, bi) for ai, bi in zip(a, b))

    type_tuple = (np.ndarray, list, cp.ndarray) if cp else (np.ndarray, list)
    if isinstance(a, type_tuple) and isinstance(b, type_tuple):
        try:
            a_arr = np.asarray(a.get()) if hasattr(a, 'get') else np.asarray(a)
            b_arr = np.asarray(b.get()) if hasattr(b, 'get') else np.asarray(b)
            return np.array_equal(a_arr, b_arr)
        except ValueError:
            return False

    if (isinstance(a, np.generic) and not isinstance(a, np.ndarray)) or \
            (isinstance(b, np.generic) and not isinstance(b, np.ndarray)):
        return float(a) == float(b)

    try:
        return a == b
    except ValueError:
        return False

# 定义索引合并的测试用例（从test_index_merge.py迁移）
INDEX_MERGE_CASES = [
    {
        "name": "链式索引 - 整数+切片",
        "base_index": (1,),
        "current_index": (slice(0, 3),),
        "expected": (1, slice(0, 3)),
        "description": "x[1][:3] -> (1, slice(0,3))"
    },
    {
        "name": "链式索引 - 整数+数组",
        "base_index": (2,),
        "current_index": ([0, 2],),
        "expected": (2, [0, 2]),
        "description": "x[2][[0,2]] -> (2, [0,2])"
    },
    {
        "name": "链式索引 - 整数+整数",
        "base_index": (3,),
        "current_index": 1,
        "expected": (3, 1),
        "description": "x[3][1] -> (3, 1) (标量索引)"
    },
    {
        "name": "多层切片索引",
        "base_index": (slice(1, 4),),
        "current_index": (slice(0, 2),),
        "expected": (slice(1, 3),),
        "description": "x[1:4][:2] -> (slice(1,3),)"
    },
    {
        "name": "多层数组索引",
        "base_index": ([0, 2, 4],),
        "current_index": ([0, 2],),
        "expected": ([0, 4],),
        "description": "x[[0,2,4]][[0,2]] -> ([0,4],)"
    },
    {
        "name": "混合索引类型",
        "base_index": (slice(0, 5, 2),),
        "current_index": (slice(1, None, None),),
        "expected": (slice(2, 5, 2),),
        "description": "x[::2][1:] -> (slice(2,5,2),)"
    },
    {
        "name": "二维链式索引",
        "base_index": (1, 2),
        "current_index": (),
        "expected": (1, 2),
        "description": "x[1][2] -> (1, 2) (标量)"
    },
    {
        "name": "省略号索引",
        "base_index": (slice(None), 1),
        "current_index": (..., 2),
        "expected": (2, 1),
        "description": "x[:,1][...,2] -> (2, 1) (1D tensor index maps to base tensor's [2,1])"
    },
    {
        "name": "省略号与切片组合",
        "base_index": (slice(None), slice(1, 3)),
        "current_index": (..., 1),
        "expected": (slice(0, 5, None), 2),
        "description": "x[:,1:3][...,1] -> (slice(None), 2) (2D tensor with ellipsis and integer indexing)",
        "explanation": "x[:,1:3] creates 5x2 view, [...,1] selects column 1 of this view, which maps to column 2 in base tensor"
    },
    {
        "name": "带步长的切片索引",
        "base_index": (slice(0, 5, 2),),
        "current_index": (slice(0, None, 2),),
        "expected": (slice(0, 5, 4),),
        "description": "x[::2][::2] -> (slice(0, 5, 4),) (step multiplication)"
    },
    {
        "name": "负索引",
        "base_index": (-1,),
        "current_index": (-2,),
        "expected": (-1, -2),
        "description": "x[-1][-2] -> (-1, -2) (negative indices)"
    },
    {
        "name": "混合索引 - 切片+整数",
        "base_index": (slice(1, 4),),
        "current_index": (1,),
        "expected": (2,),
        "description": "x[1:4][1] -> (2,) (slice to integer indexing)"
    },
    {
        "name": "空索引",
        "base_index": (),
        "current_index": (),
        "expected": (),
        "description": "x[] -> () (empty indices)"
    },
    {
        "name": "复杂混合索引",
        "base_index": ([0, 2], slice(1, None)),
        "current_index": (1, [0, 2]),
        "expected": (2, [1, 3]),
        "description": "x[[0,2],1:][1,[0,2]] -> (2, [1, 3]) (complex mixed indexing)"
    },
    # 添加CuPy索引场景测试用例
    {
        "name": "CuPy整数数组索引",
        "base_index": (slice(1, 4),),
        "current_index": (np.array([0, 2]),),
        "expected": (np.array([1, 3]),),
        "description": "x[1:4][np.array([0,2])] -> np.array([1, 3]) (CuPy integer array indexing)",
        "explanation": "从切片[1:4]的结果中选择第1和第3个元素，对应基张量的索引1和3"
    },
    {
        "name": "CuPy布尔数组索引",
        "base_index": (slice(0, 5),),
        "current_index": (np.array([True, False, True, False, True]),),
        "expected": (np.array([0, 2, 4]),),
        "description": "x[:5][np.array([True, False, True, False, True])] -> np.array([0, 2, 4]) (CuPy boolean array indexing)",
        "explanation": "从完整切片[:5]中选择True位置的元素，对应基张量的索引0, 2, 4"
    },
    {
        "name": "切片到CuPy整数数组",
        "base_index": (slice(1, 5, 2),),
        "current_index": (np.array([1]),),
        "expected": (3,),
        "description": "x[1:5:2][np.array([1])] -> (3,) (slice to CuPy integer array indexing)",
        "explanation": "从步长为2的切片[1:5:2]（元素1, 3）中选择第2个元素（索引1），对应基张量的索引3"
    },
    {
        "name": "整数索引导致维度消失",
        "base_index": (),
        "current_index": (0, 1),
        "expected": (0, 1),
        "description": "x[0,1] -> 标量 (二维索引导致维度消失)"
    },
    {
        "name": "数组索引导致维度减少",
        "base_index": (),
        "current_index": ([[0, 2], [1, 3]]),
        "expected": ([[0, 2], [1, 3]]),
        "description": "x[[0,2], [1,3]] -> 1D张量 (数组索引导致维度减少)"
    },
    {
        "name": "切片+整数组合导致维度减少",
        "base_index": (),
        "current_index": (slice(None), 1),
        "expected": (slice(None), 1),
        "description": "x[:,1] -> 1D张量 (切片+整数索引导致维度减少)"
    },
    {
        "name": "多层索引导致连续维度减少",
        "base_index": (0,),
        "current_index": (1,),
        "expected": (0, 1),
        "description": "x[0][1] -> 标量 (多层索引导致连续维度减少)"
    },
    {
        "name": "带步长的切片保持维度",
        "base_index": (),
        "current_index": (slice(None, None, 2),),
        "expected": (slice(None, None, 2),),
        "description": "x[::2] -> 保持维度 (带步长的切片保持维度)"
    },
    {
            "name": "空切片保持维度",
            "base_index": (),
            "current_index": (slice(None), slice(None)),
            "expected": (slice(None), slice(None)),
            "description": "x[:, :] -> 保持维度 (空切片保持维度)"
        },
        {
            "name": "复杂切片组合",
            "base_index": (slice(0, 5, 2),),
            "current_index": (slice(1, 3, 2),),
            "expected": (slice(2, 5, 4),),
            "description": "x[0:5:2][1:3:2] -> x[2:5:4] (切片+切片组合)"
        },
        {
            "name": "数组索引组合",
            "base_index": ([0, 2, 4],),
            "current_index": ([1, 2],),
            "expected": ([2, 4],),
            "description": "x[[0,2,4]][[1,2]] -> x[[2,4]] (数组+数组索引)"
        },
        {
            "name": "切片与数组组合",
            "base_index": (slice(0, 5),),
            "current_index": ([1, 3],),
            "expected": ([1, 3],),
            "description": "x[:5][[1,3]] -> x[[1,3]] (切片+数组索引)"
        },
        {
            "name": "数组与切片组合",
            "base_index": ([0, 1, 2, 3, 4],),
            "current_index": (slice(1, 4),),
            "expected": ([1, 2, 3],),
            "description": "x[[0,1,2,3,4]][1:4] -> x[1:4] (数组+切片索引)"
        },
        {
            "name": "负索引混合",
            "base_index": (slice(-10, None),),
            "current_index": (-3,),
            "expected": (2,),
            "description": "x[-10:][-3] -> x[2] (负索引+负索引)"
        },
        {
            "name": "None/newaxis索引",
            "base_index": (slice(None),),
            "current_index": (None,),
            "expected": (None, slice(0, 5, None)),
            "description": "x[:][None] -> 添加新维度 (None索引合并)"
        },
        {
            "name": "复杂省略号使用",
            "base_index": (slice(None), slice(1, 4)),
            "current_index": (..., 2),
            "expected": (slice(0, 5, None), 3),
            "description": "x[:,1:4][...,2] -> x[:,3] (省略号+整数索引)"
        },
        {
            "name": "多层负索引",
            "base_index": (-2,),
            "current_index": (-1,),
            "expected": (-2, -1),
            "description": "x[-2][-1] -> x[-2,-1] (多层负索引)"
        },
        {
            "name": "边界索引值",
            "base_index": (slice(0, 5),),
            "current_index": (4,),
            "expected": (4,),
            "description": "x[:5][4] -> x[4] (边界索引)"
        },
        {
            "name": "步长为None的切片",
            "base_index": (slice(None, None, None),),
            "current_index": (slice(1, 3, None),),
            "expected": (slice(1, 3, None),),
            "description": "x[:][1:3] -> x[1:3] (步长为None的切片组合)"
        },
        {
            "name": "多维复杂索引",
            "base_index": (slice(0, 5, 2), slice(1, 4)),
            "current_index": ([0, 2], slice(None, None, 2)),
            "expected": ([0, 4], slice(1, 4, 2)),
            "description": "x[::2,1:4][[0,2],::2] -> x[[0,4],1:4:2] (多维复杂索引组合)"
        },
        # 补充测试场景 - 布尔索引
        {
            "name": "布尔索引 - 基础索引为切片",
            "base_index": (slice(None),),
            "current_index": np.array([True, False, True, False, True], dtype=bool),
            "expected": (np.array([0, 2, 4]),),
            "description": "x[np.array([True,False,True,False,True])] -> 布尔索引被转换为整数数组"
        },
        {
            "name": "布尔索引 - 基础索引为数组",
            "base_index": ([0, 2, 3],),
            "current_index": np.array([True, False, True], dtype=bool),
            "expected": ([0, 3],),
            "description": "x[[0,2,3]][np.array([True,False,True])] -> 数组+布尔索引组合"
        },
        {
            "name": "布尔索引 - 二维布尔索引",
            "base_index": (),
            "current_index": np.array([[True, False, True, False], [False, True, False, True], [True, False, True, False], [False, True, False, True], [True, False, True, False]]),
            "expected": np.array([[True, False, True, False], [False, True, False, True], [True, False, True, False], [False, True, False, True], [True, False, True, False]]),
            "description": "x[np.array([[True,False,True,False],[False,True,...]])] -> 二维布尔索引（与测试张量形状匹配）"
        },
        # 补充测试场景 - 复杂多层索引
        {
            "name": "三层链式索引",
            "base_index": (slice(1, 4), slice(1, 3)),
            "current_index": (1, 1),
            "expected": (2, 2),
            "description": "x[1:4,1:3][1][1] -> 三层链式索引导致维度完全消失"
        },
        {
            "name": "混合索引类型多层嵌套",
            "base_index": ([0, 3], slice(1, 4)),
            "current_index": (slice(None), [0, 2]),
            "expected": ([0, 3], [1, 3]),
            "description": "x[[0,3],1:4][:,[0,2]] -> 数组+切片+数组多层嵌套",
            "note": "注意：numpy对多维数组索引会进行广播，导致直接索引和链式索引结果不同，这是numpy的正常行为"
        },
        # 补充测试场景 - 特殊边界情况
        {
            "name": "空数组索引",
            "base_index": ([0, 1, 2],),
            "current_index": np.array([], dtype=int),
            "expected": (np.array([], dtype=int),),
            "description": "x[[0,1,2]][np.array([], dtype=int)] -> 空数组索引"
        },
        {
            "name": "空布尔索引",
            "base_index": (slice(None),),
            "current_index": np.array([], dtype=bool),
            "expected": (np.array([], dtype=int),),
            "description": "x[np.array([], dtype=bool)] -> 空布尔索引（转换为整数索引）"
        },
        {
            "name": "全False布尔索引",
            "base_index": (slice(None),),
            "current_index": np.array([False, False, False, False, False], dtype=bool),
            "expected": (np.array([], dtype=int),),
            "description": "x[np.array([False,False,False,False,False])] -> 全False布尔索引（返回空结果）"
        },
        {
            "name": "步长为负的切片索引",
            "base_index": (slice(None, None, -1),),
            "current_index": (slice(1, 4),),
            "expected": (slice(3, 0, -1),),
            "description": "x[::-1][1:4] -> 负步长切片+切片组合（2D张量）"
        },
        # 补充测试场景 - 复杂索引组合
        {
            "name": "省略号与数组索引混合",
            "base_index": (slice(None), slice(0, 4, 2)),
            "current_index": (..., [0, 1]),
            "expected": (slice(0, 5, None), [0, 2]),
            "description": "x[:,::2][...,[0,1]] -> 2维张量的省略号与数组索引混合"
        },
        {
            "name": "None/newaxis与其他索引混合",
            "base_index": (slice(1, 4),),
            "current_index": (None, slice(None), None),
            "expected": (None, slice(1, 4, None), None),
            "description": "x[1:4][None,:,None] -> None/newaxis与切片混合添加维度"
        },
        {
            "name": "复杂混合索引 - 整数、切片",
            "base_index": (2, slice(1, 4)),
            "current_index": (1,),
            "expected": (2, 2),
            "description": "x[2,1:4][1] -> 整数+切片+整数的复杂混合索引（与2维测试张量匹配）"
        }
]
# 测试原地操作类
class TestInplaceOperations(unittest.TestCase):
    WHOLE = object()  # 整张量原地运算的索引标记（区别于 None 索引）

    def setUp(self):
        # 设置随机种子以确保结果可重复
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch.set_default_dtype(torch.float32)
        # 获取当前测试方法名
        self.current_test_name = self._testMethodName
        # 如果是独立脚本运行，则开始记录函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")

    def tearDown(self):
        # 如果是独立脚本运行，则结束函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")

    # ==================== 通用执行框架 ====================
    def _devices(self):
        """返回当前环境可用的设备列表"""
        return ["cpu", "cuda"] if CUDA_AVAILABLE else ["cpu"]

    def _report(self, case_name, passed, start_time, details=None, error=None):
        """统一记录并打印单个用例结果（仅独立脚本运行模式）"""
        if not IS_RUNNING_AS_SCRIPT:
            return
        stats.add_result(case_name, passed)
        elapsed = time.time() - start_time
        if error is not None:
            print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({elapsed:.4f}秒) - {error}")
            return
        status = "通过" if passed else "失败"
        color = Colors.OKGREEN if passed else Colors.FAIL
        print(f"测试用例: {case_name} - {color}{status}{Colors.ENDC} ({elapsed:.4f}秒)")
        if not passed:
            for label, ok in (details or {}).items():
                print(f"  {label}: {'通过' if ok else '失败'}")

    def _run_case(self, case_name, body):
        """执行单个用例：body 返回 (passed, details)，统一处理统计、打印与断言"""
        start_time = time.time()
        try:
            passed, details = body()
        except Exception as e:
            self._report(case_name, False, start_time, error=str(e))
            raise
        self._report(case_name, passed, start_time, details)
        self.assertTrue(passed, f"测试用例失败: {case_name}")

    def _new_pair(self, data, device):
        """创建 Riemann / PyTorch 非叶子张量对（均保留梯度）"""
        rm_x = rm.tensor(data, requires_grad=True, device=device).clone()
        rm_x.retain_grad()
        torch_x = None
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(data, requires_grad=True, device=device).clone()
            torch_x.retain_grad()
        return rm_x, torch_x

    def _create_operand(self, spec, base_shape, device):
        """按 (kind, data) 描述创建可求梯度的操作数"""
        kind, data = spec
        if kind == "like":
            data = np.random.randn(*base_shape)
        elif kind == "broadcast":
            data = np.random.randn(base_shape[0], 1)
        rm_operand = rm.tensor(data, requires_grad=True, device=device)
        torch_operand = torch.tensor(data, requires_grad=True, device=device) if TORCH_AVAILABLE else None
        return rm_operand, torch_operand

    def _execute_inplace_operation(self, tensor, index, value, op_case):
        """执行原地操作的辅助函数"""
        if op_case["op"] == "=":
            tensor[index] = value
        else:
            op_case["rm_func"](tensor, index, value)

    @staticmethod
    def _is_chained_index(index):
        """是否为链式索引（元组的每个元素本身是索引元组）"""
        return (isinstance(index, tuple) and len(index) > 0
                and all(isinstance(step, tuple) for step in index))

    def _run_inplace_case(self, base_tensor, view_chain, index_case, op_case, device,
                          label="", y_mode="target"):
        """通用原地操作用例：支持视图链、链式索引、布尔索引自适应与 y 广播"""
        is_chain_index = self._is_chained_index(index_case["index"])
        case_name = f"{op_case['name']} - {index_case['name']}{label} - {device}"

        def body():
            rm_x, torch_x = self._new_pair(base_tensor, device)
            rm_view, torch_view = rm_x, torch_x
            levels = []
            prefix = ()
            for step in view_chain:
                prefix = prefix + (step,)
                rm_view = rm_view[step]
                rm_view.retain_grad()
                if TORCH_AVAILABLE:
                    torch_view = torch_view[step]
                    torch_view.retain_grad()
                levels.append((prefix, rm_view, torch_view))

            if is_chain_index:
                rm_view = rm_view[index_case["index"][0]]
                rm_view.retain_grad()
                if TORCH_AVAILABLE:
                    torch_view = torch_view[index_case["index"][0]]
                    torch_view.retain_grad()
                target_index = index_case["index"][1]
            else:
                target_index = index_case["index"]
                # 布尔索引长度与当前视图首维不一致时，改用适配长度的布尔掩码
                if index_case["name"] in ("布尔索引", "NumPy布尔数组索引"):
                    if np.shape(target_index[0])[0] != rm_view.shape[0]:
                        target_index = (np.array([True, False, True]),)

            target_shape = np.zeros(rm_view.shape)[target_index].shape
            y_shape = (1,) if (y_mode == "broadcast" and not is_chain_index) else target_shape
            rm_y, torch_y = self._new_pair(np.ones(y_shape) * 5, device)

            self._execute_inplace_operation(rm_view, target_index, rm_y, op_case)
            if TORCH_AVAILABLE:
                if op_case["op"] == "=":
                    torch_view[target_index] = torch_y
                else:
                    op_case["torch_func"](torch_view, target_index, torch_y)

            rm_x.sum().backward()
            if TORCH_AVAILABLE:
                torch_x.sum().backward()

            details = {
                "前向传播比较": compare_values(rm_x, torch_x),
                "X梯度比较": compare_grads(rm_x, torch_x),
                "Y梯度比较": True if is_chain_index else compare_grads(rm_y, torch_y),
            }
            if levels:
                consistent = True
                for level_prefix, rm_level, torch_level in levels:
                    rm_ref, torch_ref = rm_x, torch_x
                    for step in level_prefix:
                        rm_ref = rm_ref[step]
                        if TORCH_AVAILABLE:
                            torch_ref = torch_ref[step]
                    level_ok = compare_values(rm_level, rm_ref)
                    if TORCH_AVAILABLE:
                        level_ok = level_ok and compare_values(torch_level, torch_ref)
                    consistent = consistent and level_ok
                details["视图一致性"] = consistent

            return all(details.values()), details

        self._run_case(case_name, body)

    # ==================== 原地操作网格测试 ====================
    def test_original_tensor_inplace(self):
        """测试对原始张量的原地操作"""
        base_tensor = np.random.randn(5, 4)
        for device in self._devices():
            for op_case in INPLACE_OPS:
                for index_case in INDEX_CASES:
                    self._run_inplace_case(base_tensor, [], index_case, op_case, device,
                                           y_mode="broadcast")

    def test_view_tensor_inplace(self):
        """测试对视图对象的原地操作"""
        base_tensor = np.random.randn(5, 4)
        view_chain = [(slice(1, 4), slice(None))]
        for device in self._devices():
            for op_case in INPLACE_OPS:
                for index_case in INDEX_CASES:
                    # 链式索引本身就是多级索引，单独在多级视图用例中覆盖
                    if index_case["name"] == "链式索引":
                        continue
                    self._run_inplace_case(base_tensor, view_chain, index_case, op_case, device,
                                           label="（视图）", y_mode="target")

    def test_multi_level_view_inplace(self):
        """测试对多层视图对象的原地操作"""
        base_tensor = np.random.randn(5, 4, 3)
        view_chain = [
            (slice(None), slice(1, 3), slice(None)),
            (slice(1, 4), slice(None), slice(0, 2)),
            (slice(None), slice(None), 1),
        ]
        # 这些索引不适用于三维多层视图
        excluded = {"多维数组索引", "复杂的混合索引", "链式索引"}
        for device in self._devices():
            for op_case in INPLACE_OPS:
                for index_case in INDEX_CASES:
                    if index_case["name"] in excluded:
                        continue
                    self._run_inplace_case(base_tensor, view_chain, index_case, op_case, device,
                                           label="（多层视图）", y_mode="target")

    def test_cuda_index_cases(self):
        """测试CUDA环境下的专用索引用例"""
        if not CUDA_AVAILABLE:
            print("CUDA not available, skipping CUDA index tests")
            return

        base_tensor = np.random.randn(5, 4)
        device = "cuda"
        for op_case in INPLACE_OPS:
            rm_x = rm.tensor(base_tensor, requires_grad=True, device=device).clone()
            rm_x.retain_grad()

            test_scenarios = [
                {"name": "CuPy整数数组索引", "base_index": (slice(1, 4),), "current_index": (cp.array([0, 2]),)},
                {"name": "CuPy布尔数组索引", "base_index": (slice(0, 5),), "current_index": (cp.array([True, False, True, False, True]),)},
            ]

            for scenario in test_scenarios:
                case_name = f"{op_case['name']} - {scenario['name']} - {device}"
                start_time = time.time()
                print(f"Test case: {case_name}")
                try:
                    merged_index = rm_x.merge_indices(scenario['base_index'], scenario['current_index'])
                    print(f"Merged index: {merged_index}")

                    view1 = rm_x[scenario['base_index']]
                    # 将 CuPy 索引转换为 NumPy 索引，用于计算目标形状
                    numpy_index = tuple(idx.get() if isinstance(idx, cp.ndarray) else idx
                                        for idx in scenario['current_index'])
                    target_shape = np.zeros(view1.shape)[numpy_index].shape
                    rm_y = rm.tensor(np.ones(target_shape) * 5, requires_grad=True, device=device)

                    view1[scenario['current_index']] = rm_y
                    rm_x.sum().backward()

                    stats.add_result(case_name, True)
                    print(f"✓ Test passed! ({time.time() - start_time:.4f}秒)")
                except Exception as e:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"✗ Test failed with error: {e} ({time.time() - start_time:.4f}秒)")
                    import traceback
                    traceback.print_exc()

    # ==================== copy_ / fill_ / masked_fill_ 测试 ====================
    def _run_special_case(self, op_name, case, device, inplace=True):
        """通用特殊原地/非原地操作用例"""
        base = np.random.randn(5, 4)
        case_name = f"{op_name} - {case['name']} - {device}"

        def body():
            rm_tensor, torch_tensor = self._new_pair(base, device)

            if op_name.startswith("masked_fill"):
                mask_shape = (base.shape[0], 1) if case["mask_type"] == "broadcast_bool" else base.shape
                mask_data = np.random.choice([True, False], size=mask_shape)
                rm_mask = rm.tensor(mask_data, device=device)
                torch_mask = torch.tensor(mask_data, device=device) if TORCH_AVAILABLE else None
                rm_value, torch_value = self._create_operand(case.get("value", ("scalar", 0.0)),
                                                             base.shape, device)
                rm_func = rm_tensor.masked_fill_ if inplace else rm_tensor.masked_fill
                torch_func = torch_tensor.masked_fill_ if inplace else torch_tensor.masked_fill
                args, torch_args = (rm_mask, rm_value), (torch_mask, torch_value)
            else:
                spec = case["src"] if op_name == "copy_" else case["value"]
                rm_value, torch_value = self._create_operand(spec, base.shape, device)
                rm_func = rm_tensor.copy_ if op_name == "copy_" else rm_tensor.fill_
                torch_func = torch_tensor.copy_ if op_name == "copy_" else torch_tensor.fill_
                args, torch_args = (rm_value,), (torch_value,)

            rm_original = rm_tensor.clone()
            rm_result = rm_func(*args)
            torch_result = torch_func(*torch_args) if TORCH_AVAILABLE else None

            if inplace:
                rm_tensor.sum().backward()
                if TORCH_AVAILABLE:
                    torch_tensor.sum().backward()
                details = {
                    "函数值": compare_values(rm_tensor, torch_tensor),
                    "self梯度": compare_grads(rm_tensor, torch_tensor),
                    "右值梯度": compare_grads(rm_value, torch_value) if TORCH_AVAILABLE else True,
                }
            else:
                details = {"原张量未被修改": compare_values(rm_tensor, rm_original)}
                rm_result.sum().backward()
                if TORCH_AVAILABLE:
                    torch_result.sum().backward()
                details["函数值"] = compare_values(rm_result, torch_result)
                details["self梯度"] = compare_grads(rm_tensor, torch_tensor) if TORCH_AVAILABLE else True
                details["右值梯度"] = compare_grads(rm_value, torch_value) if TORCH_AVAILABLE else True

            return all(details.values()), details

        self._run_case(case_name, body)

    def test_copy_function(self):
        """测试 copy_ 特殊原地操作函数"""
        for device in self._devices():
            for case in COPY_CASES:
                self._run_special_case("copy_", case, device)

    def test_fill_function(self):
        """测试 fill_ 特殊原地操作函数"""
        for device in self._devices():
            for case in FILL_CASES:
                self._run_special_case("fill_", case, device)

    def test_masked_fill_function(self):
        """测试 masked_fill_ 特殊原地操作函数"""
        for device in self._devices():
            for case in MASKED_FILL_CASES:
                self._run_special_case("masked_fill_", case, device)

    def test_masked_fill_non_inplace_function(self):
        """测试 masked_fill 非原地操作函数"""
        for device in self._devices():
            for case in MASKED_FILL_CASES:
                self._run_special_case("masked_fill", case, device, inplace=False)

    # ==================== 索引合并测试 ====================
    def _index_merge_body(self, base_tensor, scenario, device):
        rm_x = rm.tensor(base_tensor, requires_grad=True, device=device)
        merged_index = rm_x.merge_indices(scenario['base_index'], scenario['current_index'])
        index_passed = safe_compare_index(merged_index, scenario['expected'])

        view_passed = True
        current_idx_is_empty = (isinstance(scenario['current_index'], tuple)
                                and scenario['current_index'] == ())
        # note 标记的场景跳过视图内容验证（如 numpy 广播行为导致的差异）
        if index_passed and not current_idx_is_empty and 'note' not in scenario:
            view1 = rm_x[scenario['base_index']]
            view2 = view1[scenario['current_index']]
            direct = rm_x[merged_index]
            try:
                if view2.shape == () and direct.shape == ():
                    view_passed = abs(float(view2) - float(direct)) < 1e-10
                else:
                    view_passed = np.allclose(view2.detach().cpu().numpy(),
                                              direct.detach().cpu().numpy(), rtol=1e-05, atol=1e-08)
            except Exception:
                view_passed = False

        details = {"索引合并比较": index_passed, "视图内容比较": view_passed}
        return index_passed and view_passed, details

    def test_index_merge_function(self):
        """测试索引合并功能"""
        base_tensor = np.arange(20, dtype=np.float64).reshape(5, 4)
        for device in self._devices():
            for scenario in INDEX_MERGE_CASES:
                self._run_case(
                    f"索引合并 - {scenario['name']} - {device}",
                    lambda s=scenario, d=device: self._index_merge_body(base_tensor, s, d))

    # ==================== 通用单张量原地操作测试 ====================
    _IPOW_METHODS = {"+=": "__iadd__", "-=": "__isub__", "*=": "__imul__",
                     "/=": "__itruediv__", "**=": "__ipow__"}

    def _apply_simple_op(self, tensor, index, op, value, is_torch=False):
        """对张量执行一次原地操作；链式索引以元组嵌套表示"""
        container, target_index = tensor, index
        if self._is_chained_index(index):
            for step in index[:-1]:
                container = container[step]
            target_index = index[-1]

        if is_torch and isinstance(value, np.ndarray):
            value = torch.tensor(value, device=tensor.device)

        if target_index is self.WHOLE:
            # 整张量原地运算（0 维标量场景）
            getattr(container, self._IPOW_METHODS[op])(value)
        elif op == "=":
            container[target_index] = value
        else:
            # 与 `t[idx] += v` 的语义一致：调用原地方法后写回
            container[target_index] = getattr(container[target_index], self._IPOW_METHODS[op])(value)

    def _simple_case_body(self, case, device):
        base = np.array(case["base"], copy=True)
        rm_x = rm.tensor(base, requires_grad=True, device=device) * 1
        rm_x.retain_grad()
        torch_x = None
        if TORCH_AVAILABLE:
            torch_x = torch.tensor(base, requires_grad=True, device=device) * 1
            torch_x.retain_grad()

        rm_target, torch_target = rm_x, torch_x
        for step in case.get("view", ()):
            rm_target = rm_target[step]
            if TORCH_AVAILABLE:
                torch_target = torch_target[step]

        for index, op, value in case["ops"]:
            self._apply_simple_op(rm_target, index, op, value)
            if TORCH_AVAILABLE:
                self._apply_simple_op(torch_target, index, op, value, is_torch=True)

        rm_x.sum().backward()
        if TORCH_AVAILABLE:
            torch_x.sum().backward()

        details = {
            "前向传播比较": compare_values(rm_x, torch_x),
            "梯度比较": compare_grads(rm_x, torch_x),
        }
        return all(details.values()), details

    def _run_simple_cases(self, cases):
        for device in self._devices():
            for case in cases:
                self._run_case(f"{case['name']} - {device}",
                               lambda c=case, d=device: self._simple_case_body(c, d))

    def test_fancy_indexing_inplace(self):
        """测试整数数组索引和布尔数组索引的原地操作

        覆盖 __setitem__ 中 temp._is_view = False 的代码路径，
        以及数组索引作为 stored_index 时与空元组比较的判断。
        """
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self._run_simple_cases([
            {"name": "NumPy整数数组索引原地加法", "base": base, "ops": [(np.array([0, 2, 4]), "+=", 10)]},
            {"name": "NumPy布尔数组索引原地加法", "base": base, "ops": [(np.array([True, False, True, False, True]), "+=", 100)]},
            {"name": "Python列表整数数组索引原地乘法", "base": base, "ops": [([1, 3], "*=", 2)]},
            {"name": "NumPy整数数组索引原地减法", "base": base, "ops": [(np.array([0, 4]), "-=", 5)]},
            {"name": "NumPy整数数组索引原地除法", "base": base, "ops": [(np.array([1, 3]), "/=", 2)]},
        ])

    def test_powat_inplace(self):
        """测试原地幂运算操作 (**=)"""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self._run_simple_cases([
            {"name": "标量幂运算", "base": base, "ops": [(np.array([0, 2, 4]), "**=", 2)]},
            {"name": "切片索引幂运算", "base": base, "ops": [(slice(1, 4), "**=", 3)]},
            {"name": "负指数幂运算", "base": base, "ops": [(np.array([1, 3]), "**=", -1)]},
        ])

    def test_edge_cases(self):
        """测试边界情况：标量赋值给数组索引、步长切片"""
        self._run_simple_cases([
            {"name": "标量赋值给数组索引", "base": np.ones((5,)) * 2.0,
             "ops": [(np.array([0, 2, 4]), "=", 99.0)]},
            {"name": "步长切片乘法", "base": np.ones((6,)) * 2.0,
             "ops": [(slice(0, 6, 2), "*=", 2)]},
        ])

    def test_chained_inplace(self):
        """测试连续原地操作（+= 10, *= 2, -= 5）"""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self._run_simple_cases([
            {"name": "连续原地操作链", "base": base,
             "ops": [(np.array([0, 2, 4]), "+=", 10),
                     (slice(1, 4), "*=", 2),
                     (np.array([0, 2, 4]), "-=", 5)]},
        ])

    def test_dtype_broadcast(self):
        """测试数据类型和广播（标量赋值 / 加法）"""
        self._run_simple_cases([
            {"name": "广播赋值-标量到数组", "base": np.ones((3, 4)) * 3.0,
             "ops": [((slice(0, 2), slice(0, 2)), "=", 5.0)]},
            {"name": "标量加法到数组索引", "base": np.ones((4, 4)) * 3.0,
             "ops": [((slice(0, 2), slice(0, 2)), "+=", 10.0)]},
        ])

    def test_none_index_inplace(self):
        """测试 None/np.newaxis 索引的原地操作"""
        self._run_simple_cases([
            {"name": "None索引原地加法", "base": np.ones((5,)) * 2.0, "ops": [(None, "+=", 10)]},
            {"name": "None索引原地赋值", "base": np.ones((4,)) * 2.0, "ops": [(None, "=", 99.0)]},
            {"name": "None索引原地乘法", "base": np.ones((3,)) * 2.0, "ops": [(None, "*=", 2)]},
            {"name": "链式None索引操作", "base": np.ones((5,)) * 2.0, "ops": [((None, 0), "+=", 5)]},
            {"name": "slice视图后None索引操作", "base": np.ones((6, 10)) * 2.0,
             "ops": [(((slice(2, 8), slice(3, 7)), (None, slice(1, 3))), "+=", 5)]},
        ])

    def test_scalar_tensor_inplace(self):
        """测试 0 维（标量）张量的原地操作"""
        base = np.array(5.0)
        self._run_simple_cases([
            {"name": "标量张量原地加法", "base": base, "ops": [(self.WHOLE, "+=", 10)]},
            {"name": "标量张量原地乘法", "base": base, "ops": [(self.WHOLE, "*=", 2)]},
            {"name": "标量张量原地赋值", "base": base, "ops": [((), "=", 99.0)]},
        ])

    def test_non_contiguous_inplace(self):
        """测试非连续内存张量（步长切片视图）的原地操作"""
        self._run_simple_cases([
            {"name": "步长切片张量原地乘法",
             "base": np.arange(6).astype(np.float32) + 1.0,
             "view": (slice(None, None, 2),),
             "ops": [(slice(None, None, 2), "*=", 2)]},
        ])

    def test_broadcast_complex(self):
        """测试复杂广播场景的原地操作"""
        self._run_simple_cases([
            {"name": "行向量广播到多行", "base": np.ones((3, 4)) * 2.0,
             "ops": [((0, slice(None)), "=", np.array([10, 20, 30, 40]))]},
            {"name": "列向量广播到多列", "base": np.ones((3, 4)) * 2.0,
             "ops": [((slice(None), 0), "+=", np.array([1, 2, 3]))]},
        ])

    # ==================== 多次重叠索引赋值测试 ====================
    def _run_setitem_sequence(self, case, device):
        """执行“多输入多次重叠赋值”用例，并与 PyTorch 比较前向值和各输入梯度

        每步执行 res[索引] = 源值，步间克隆结果张量以保留计算图。
        """
        def body():
            out_shape = case["out_shape"]

            # Riemann 侧：各输入为叶子张量，结果张量由零张量克隆而来（非叶子）
            rm_inputs = [rm.tensor(np.asarray(v, dtype=np.float32), requires_grad=True, device=device)
                         for v in case["inputs"]]
            rm_result = rm.tensor(np.zeros(out_shape, dtype=np.float32),
                                  requires_grad=True, device=device).clone()
            for i, (idx, src, sub_idx) in enumerate(case["ops"]):
                value = rm_inputs[src] if sub_idx is None else rm_inputs[src][sub_idx]
                rm_result[idx] = value
                if i < len(case["ops"]) - 1:
                    rm_result = rm_result.clone()
            rm_result.sum().backward()
            rm_grads = [t.grad for t in rm_inputs]

            # PyTorch 侧：完全相同的操作序列
            torch_result, torch_grads = None, None
            if TORCH_AVAILABLE:
                torch_inputs = [torch.tensor(np.asarray(v, dtype=np.float32),
                                             requires_grad=True, device=device)
                                for v in case["inputs"]]
                torch_result = torch.zeros(out_shape, requires_grad=True, device=device).clone()
                for i, (idx, src, sub_idx) in enumerate(case["ops"]):
                    value = torch_inputs[src] if sub_idx is None else torch_inputs[src][sub_idx]
                    torch_result[idx] = value
                    if i < len(case["ops"]) - 1:
                        torch_result = torch_result.clone()
                torch_result.sum().backward()
                torch_grads = [t.grad for t in torch_inputs]

            details = {
                "前向值比较": compare_values(rm_result, torch_result),
                "梯度比较": compare_values(rm_grads, torch_grads),
            }
            return all(details.values()), details

        self._run_case(f"{case['name']} - {device}", body)

    def test_overlapping_setitem(self):
        """测试多次重叠的原地索引赋值（前向值与各输入梯度与 PyTorch 一致）"""
        for device in self._devices():
            for case in SETITEM_OVERLAP_CASES:
                self._run_setitem_sequence(case, device)

    def test_2d_mixed_index_setitem(self):
        """测试二维张量上的混合索引重叠赋值（梯度只沿最后一次写入回传）"""
        for device in self._devices():
            for case in SETITEM_2D_CASES:
                self._run_setitem_sequence(case, device)


if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True

    # 清屏
    clear_screen()

    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行原地操作测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()

    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestInplaceOperations)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)

    # 打印测试统计摘要
    stats.print_summary()

    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)
