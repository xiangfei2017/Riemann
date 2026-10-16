"""
测试riemann的einsum函数实现
"""
import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    # 从rm.cuda获取cupy引用和CUDA可用性
    CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
    cp = rm.cuda.cp
except ImportError as e:
    print(f"无法导入riemann模块: {e}")
    print("请确保项目路径设置正确")
    sys.exit(1)


# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True

    # 在模块级别进行PyTorch预热，避免在测试计时中包含初始化开销
    print("预热PyTorch系统...")
    warmup_start = time.time()
    
    # 执行简单的PyTorch操作以触发初始化
    warmup_input = torch.tensor([[0.0]], requires_grad=True)
    warmup_output = warmup_input.sum()
    warmup_output.backward()
    
    # 清理资源
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"PyTorch预热完成，耗时: {time.time() - warmup_start:.4f}秒")
    
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的einsum函数")
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
# 判断是否作为独立脚本运行
IS_RUNNING_AS_SCRIPT = False

# 比较值的函数
def compare_values(rm_result, torch_result, atol=1e-5, rtol=1e-5):
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
    
    # 转换为numpy数组
    try:
        # 处理Riemann结果
        if hasattr(rm_result, 'is_cuda') and rm_result.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            rm_data = rm_result.detach().cpu().numpy()
        else:
            rm_data = rm_result.detach().numpy()
        
        # 处理PyTorch结果
        if hasattr(torch_result, 'is_cuda') and torch_result.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            torch_data = torch_result.detach().cpu().numpy()
        else:
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


# 测试einsum类
class TestEinsum(unittest.TestCase):
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
    
    # ------------------------------------------------------------------
    # 通用用例执行器：集中处理前向 / 反向的 PyTorch 对齐校验
    # ------------------------------------------------------------------

    def _run_cases(self, cases, fail_label):
        """通用用例执行器：逐个设备执行前向 / 反向的 PyTorch 对齐校验。

        每个用例为 dict，字段含义：
          name           用例名（必填）
          eq             einsum 方程（必填）
          shapes         各操作数的形状列表（必填）
          grad           反向模式：'random'（默认，注入非均匀上游梯度）、
                         'ones'（全 1 梯度）或 None（跳过反向）
          implicit       显式方程；提供时额外验证隐式与显式输出一致
          complex        True 时按复数语义构造张量并注入复数上游梯度
          dtype          操作数 dtype（如 np.float32），默认保持 float64
          noncontiguous  True 时在 2 倍尺寸基张量上做 [::2] 切片构造非连续输入
          check_dtype    需校验的输出 dtype（如 np.float32）

        反向默认注入随机上游梯度（0 维输出时即为非单位梯度），以校验 grad_value
        确实参与了梯度计算并落在正确位置。
        """
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")

        for device in devices:
            for case in cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 前向：比较数值，并校验隐式输出、输出 dtype 与设备一致性
                    (forward_passed, device_passed, result, result_torch,
                     ops, ops_torch) = self._forward_check(case, device)

                    # 反向：与 PyTorch 逐操作数对齐梯度
                    grad_mode = case.get('grad', 'random')
                    backward_passed = True
                    if grad_mode is not None:
                        backward_passed = self._backward_check(
                            case, grad_mode, device, result, result_torch, ops, ops_torch)

                    passed = forward_passed and backward_passed and device_passed
                    time_taken = time.time() - start_time

                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")

                    self.assertTrue(passed, f"{fail_label}失败: {case_name}")

                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def _create_tensor_pair(self, np_ops, device, complex_):
        """把同设备、带梯度的 numpy 数据列表分别转成 riemann / torch 张量"""
        rm_kwargs = {"requires_grad": True}
        if device != "cpu":
            rm_kwargs["device"] = device
        ops = [rm.tensor(a, **rm_kwargs) for a in np_ops]

        ops_torch = None
        if TORCH_AVAILABLE:
            torch_kwargs = {"requires_grad": True}
            if device != "cpu":
                torch_kwargs["device"] = device
            if complex_:
                # PyTorch 复数张量统一使用 complex64
                torch_kwargs["dtype"] = torch.complex64
            ops_torch = [torch.tensor(a, **torch_kwargs) for a in np_ops]
        return ops, ops_torch

    def _build_case_operands(self, case, device):
        """按用例配置生成 numpy 数据并构造 riemann / torch 操作数张量对"""
        shapes = case['shapes']
        if case.get('noncontiguous'):
            # 扩大一倍尺寸后按 [::2] 切片，得到非连续视图
            base_np = [np.random.randn(*(d * 2 for d in s)) for s in shapes]
            base_ops, base_torch = self._create_tensor_pair(base_np, device, False)
            step = tuple(slice(None, None, 2) for _ in shapes[0])
            ops = [op[step] for op in base_ops]
            ops_torch = [t[step] for t in base_torch] if base_torch is not None else None
            return ops, ops_torch

        # np.asarray 保证 0-d 标量场景也得到 ndarray（而非 numpy 标量），使 rm.tensor 保留 float64
        if case.get('complex'):
            np_ops = [np.asarray(np.random.randn(*s) + 1j * np.random.randn(*s)) for s in shapes]
        else:
            np_ops = [np.asarray(np.random.randn(*s)) for s in shapes]
            if case.get('dtype') is not None:
                np_ops = [a.astype(case['dtype']) for a in np_ops]
        return self._create_tensor_pair(np_ops, device, case.get('complex', False))

    def _forward_check(self, case, device):
        """执行单个方程的前向对比，并校验隐式输出、输出 dtype 与设备一致性"""
        ops, ops_torch = self._build_case_operands(case, device)
        result = rm.einsum(case['eq'], *ops)
        result_torch = torch.einsum(case['eq'], *ops_torch) if TORCH_AVAILABLE else None

        passed = compare_values(result, result_torch)
        # 隐式输出应与对应的显式输出完全一致
        if case.get('implicit'):
            explicit = rm.einsum(case['implicit'], *ops)
            passed = passed and bool(np.allclose(result.data, explicit.data, atol=1e-6))
        # 输出 dtype 校验
        if case.get('check_dtype') is not None:
            passed = passed and result.dtype == case['check_dtype']
        return passed, result.device == ops[0].device, result, result_torch, ops, ops_torch

    def _backward_check(self, case, grad_mode, device, result, result_torch, ops, ops_torch):
        """注入上游梯度并逐操作数对齐 riemann / PyTorch 的梯度"""
        if not TORCH_AVAILABLE:
            return True
        dev_kwargs = {} if device == "cpu" else {"device": device}
        if case.get('complex'):
            # 复数场景需显式传入复数上游梯度，标量输出取 1+0j
            if result.ndim == 0:
                r_grad = rm.tensor(1 + 0j, dtype=result.dtype)
                t_grad = torch.tensor(1 + 0j, dtype=torch.complex64)
            else:
                r_grad = rm.ones_like(result)
                t_grad = torch.ones_like(result_torch, dtype=torch.complex64)
        elif grad_mode == 'ones':
            r_grad = rm.ones_like(result)
            t_grad = torch.ones_like(result_torch)
        else:
            # 随机非均匀上游梯度（0 维输出时即为非单位梯度）
            w_np = np.asarray(np.random.randn(*result.shape))
            if case.get('dtype') is not None:
                w_np = w_np.astype(case['dtype'])
            r_grad = rm.tensor(w_np, **dev_kwargs)
            t_grad = torch.tensor(w_np, **dev_kwargs)
        result.backward(r_grad)
        result_torch.backward(t_grad)
        return all(compare_values(op.grad, op_t.grad)
                   for op, op_t in zip(ops, ops_torch))

    def test_matrix_and_tensor_contraction(self):
        """测试矩阵/张量缩并：一至四个操作数、批量链式、部分缩并、全缩并及标量参与"""
        self._run_cases([
            {"name": "矩阵乘法 ij,jk->ik", "eq": "ij,jk->ik", "shapes": [(3, 4), (4, 5)]},
            {"name": "批量矩阵乘法 bij,bjk->bik", "eq": "bij,bjk->bik",
             "shapes": [(10, 3, 4), (10, 4, 5)]},
            {"name": "省略号广播批量维矩阵乘法 ...ij,...jk->...ik", "eq": "...ij,...jk->...ik",
             "shapes": [(2, 1, 3, 4), (1, 5, 4, 6)]},
            {"name": "三个操作数 ij,jk,kl->il", "eq": "ij,jk,kl->il",
             "shapes": [(3, 4), (4, 5), (5, 6)]},
            {"name": "四操作数链式 ij,jk,kl,lm->im", "eq": "ij,jk,kl,lm->im",
             "shapes": [(3, 4), (4, 5), (5, 6), (6, 7)]},
            {"name": "批量三操作数 ...ij,...jk,...kl->...il", "eq": "...ij,...jk,...kl->...il",
             "shapes": [(2, 3, 3, 4), (2, 3, 4, 5), (2, 3, 5, 6)]},
            {"name": "多操作数混合 ij,jk,ik->i", "eq": "ij,jk,ik->i",
             "shapes": [(3, 4), (4, 5), (3, 5)]},
            {"name": "三操作数全缩并空输出 ij,jk,ik->", "eq": "ij,jk,ik->",
             "shapes": [(3, 4), (4, 5), (3, 5)]},
            {"name": "张量缩并 ijkl,jklm->ijm", "eq": "ijkl,jklm->ijm",
             "shapes": [(3, 4, 5, 6), (4, 5, 6, 7)]},
            {"name": "部分缩并 ijk,ikl->ijl", "eq": "ijk,ikl->ijl",
             "shapes": [(2, 3, 4), (2, 4, 5)]},
            {"name": "3D张量乘法 ijk,jkl->il", "eq": "ijk,jkl->il",
             "shapes": [(3, 4, 5), (4, 5, 6)]},
            {"name": "双操作数全缩并 ij,ji->", "eq": "ij,ji->", "shapes": [(3, 4), (4, 3)]},
            {"name": "标量乘矩阵 ,ij->ij", "eq": ",ij->ij", "shapes": [(), (3, 4)]},
            {"name": "标量+矩阵+向量 ,i,ij->j", "eq": ",i,ij->j",
             "shapes": [(), (4,), (4, 5)]},
            {"name": "三标量连乘 ,,->", "eq": ",,->", "shapes": [(), (), ()]},
        ], "矩阵与张量缩并测试")

    def test_trace_and_diagonal(self):
        """测试迹（全缩并）与对角线提取，含批量维与三重重复索引"""
        self._run_cases([
            {"name": "矩阵迹 ii->", "eq": "ii->", "shapes": [(5, 5)]},
            {"name": "对角线提取 ii->i", "eq": "ii->i", "shapes": [(5, 5)]},
            {"name": "批量矩阵迹 bii->b", "eq": "bii->b", "shapes": [(10, 5, 5)]},
            {"name": "批量对角线提取 bii->bi", "eq": "bii->bi", "shapes": [(10, 5, 5)]},
            {"name": "三重重复取对角 iii->i", "eq": "iii->i", "shapes": [(4, 4, 4)]},
            {"name": "三重重复求和 iii->", "eq": "iii->", "shapes": [(4, 4, 4)]},
            {"name": "单操作数多重复求和 iijj->", "eq": "iijj->", "shapes": [(3, 3, 4, 4)]},
        ], "迹与对角线测试")

    def test_transpose_and_identity(self):
        """测试单操作数转置（含高维重排）与恒等复制"""
        self._run_cases([
            {"name": "转置 ij->ji", "eq": "ij->ji", "shapes": [(3, 4)]},
            {"name": "3D转置 ijk->kji", "eq": "ijk->kji", "shapes": [(2, 3, 4)]},
            {"name": "3D转置 ijk->jki", "eq": "ijk->jki", "shapes": [(2, 3, 4)]},
            {"name": "3D置换 ijk->kij", "eq": "ijk->kij", "shapes": [(2, 3, 4)]},
            {"name": "4D转置 ijkl->klij", "eq": "ijkl->klij", "shapes": [(2, 3, 4, 5)]},
            {"name": "恒等复制 ij->ij", "eq": "ij->ij", "shapes": [(3, 4)]},
        ], "转置与恒等测试")

    def test_outer_and_elementwise(self):
        """测试外积、点积、Frobenius 内积、自缩并、逐元素乘积与 Kronecker 外积"""
        self._run_cases([
            {"name": "外积 i,j->ij", "eq": "i,j->ij", "shapes": [(5,), (4,)]},
            {"name": "批量外积 bi,bj->bij", "eq": "bi,bj->bij", "shapes": [(10, 3), (10, 4)]},
            {"name": "点积 i,i->", "eq": "i,i->", "shapes": [(5,), (5,)]},
            {"name": "标量点积 ,i,i->", "eq": ",i,i->", "shapes": [(), (5,), (5,)]},
            {"name": "向量与标量逐元素 i,->i", "eq": "i,->i", "shapes": [(5,), ()]},
            {"name": "Frobenius内积 ij,ij->", "eq": "ij,ij->", "shapes": [(3, 4), (3, 4)]},
            {"name": "逐元素乘积 ij,ij->ij", "eq": "ij,ij->ij", "shapes": [(3, 4), (3, 4)]},
            {"name": "省略号逐元素乘积 ...,...->...", "eq": "...,...->...",
             "shapes": [(2, 3, 4), (2, 3, 4)]},
            {"name": "自缩并 ij,ij->i", "eq": "ij,ij->i", "shapes": [(3, 4), (3, 4)]},
            {"name": "Kronecker外积 ij,kl->ijkl", "eq": "ij,kl->ijkl",
             "shapes": [(2, 3), (4, 5)]},
        ], "外积与逐元素测试")

    def test_reduction(self):
        """测试求和归约：全归约、行/列归约、3D 轴归约与归约后置换"""
        self._run_cases([
            {"name": "矩阵求和 ij->", "eq": "ij->", "shapes": [(3, 4)]},
            {"name": "向量求和 i->", "eq": "i->", "shapes": [(5,)]},
            {"name": "3D张量求和 ijk->", "eq": "ijk->", "shapes": [(2, 3, 4)]},
            {"name": "行求和 ij->i", "eq": "ij->i", "shapes": [(3, 4)]},
            {"name": "列求和 ij->j", "eq": "ij->j", "shapes": [(3, 4)]},
            {"name": "3D张量轴求和 ijk->ij", "eq": "ijk->ij", "shapes": [(2, 3, 4)]},
            {"name": "3D张量轴求和 ijk->ik", "eq": "ijk->ik", "shapes": [(2, 3, 4)]},
            {"name": "归约+置换 ijk->ji", "eq": "ijk->ji", "shapes": [(2, 3, 4)]},
            {"name": "归约+置换 ijk->kj", "eq": "ijk->kj", "shapes": [(2, 3, 4)]},
            {"name": "归约+置换 ijkl->li", "eq": "ijkl->li", "shapes": [(2, 3, 4, 5)]},
        ], "求和归约测试")

    def test_duplicate_indices(self):
        """测试重复索引：前/后重复、多重复、非连续重复及与批量维的组合"""
        self._run_cases([
            {"name": "前两索引重复求和 iij->j", "eq": "iij->j", "shapes": [(3, 3, 4)]},
            {"name": "前两索引重复部分输出 iij->ij", "eq": "iij->ij", "shapes": [(3, 3, 4)]},
            {"name": "重复索引+置换 iij->ji", "eq": "iij->ji", "shapes": [(3, 3, 4)]},
            {"name": "多索引重复 iijj->ij", "eq": "iijj->ij", "shapes": [(3, 3, 4, 4)]},
            {"name": "非连续索引重复 ijji->", "eq": "ijji->", "shapes": [(3, 4, 4, 3)]},
            {"name": "重复索引+共享索引 ii,i->i", "eq": "ii,i->i", "shapes": [(4, 4), (4,)]},
            {"name": "批量重复索引+批量共享索引 bii,bi->b", "eq": "bii,bi->b",
             "shapes": [(3, 4, 4), (3, 4)]},
            {"name": "省略号+重复索引+共享索引 ...ii,i->...i", "eq": "...ii,i->...i",
             "shapes": [(2, 4, 4), (4,)]},
            {"name": "多操作数复杂求和归约 iij,ijk->k", "eq": "iij,ijk->k",
             "shapes": [(3, 3, 4), (3, 4, 5)]},
        ], "重复索引测试")

    def test_implicit_output(self):
        """测试隐式输出格式（前向应与对应显式输出的结果一致）"""
        self._run_cases([
            {"name": "隐式输出格式 ij,jk", "eq": "ij,jk", "shapes": [(3, 4), (4, 5)],
             "implicit": "ij,jk->ik"},
            {"name": "省略号隐式输出 ...ij,...jk", "eq": "...ij,...jk",
             "shapes": [(2, 3, 4), (2, 4, 5)], "implicit": "...ij,...jk->...ik"},
            {"name": "单操作数省略号隐式输出 ...ij", "eq": "...ij",
             "shapes": [(2, 3, 4)], "implicit": "...ij->...ij"},
            {"name": "单操作数省略号迹隐式输出 ...ii", "eq": "...ii",
             "shapes": [(2, 3, 4, 4)], "implicit": "...ii->..."},
            {"name": "三操作数隐式 ij,jk,kl", "eq": "ij,jk,kl",
             "shapes": [(3, 4), (4, 5), (5, 6)], "implicit": "ij,jk,kl->il"},
            {"name": "四操作数隐式 ij,jk,kl,lm", "eq": "ij,jk,kl,lm",
             "shapes": [(3, 4), (4, 5), (5, 6), (6, 7)], "implicit": "ij,jk,kl,lm->im"},
            {"name": "隐式全缩并 ij,ij", "eq": "ij,ij", "shapes": [(3, 4), (3, 4)],
             "implicit": "ij,ij->"},
            {"name": "二维迹隐式 ii", "eq": "ii", "shapes": [(4, 4)], "implicit": "ii->"},
            {"name": "重复索引隐式 iij", "eq": "iij", "shapes": [(3, 3, 4)], "implicit": "iij->j"},
            {"name": "标量隐式 ,ij", "eq": ",ij", "shapes": [(), (3, 4)], "implicit": ",ij->ij"},
            {"name": "省略号点积隐式 ...i,...i", "eq": "...i,...i",
             "shapes": [(2, 3, 4), (2, 3, 4)], "implicit": "...i,...i->..."},
        ], "隐式输出测试")

    def test_ellipsis_operations(self):
        """测试省略号（...）批量操作场景，含仅省略号、批量维广播与省略号+显式索引混用"""
        self._run_cases([
            {"name": "批量外积", "eq": "...i,...j->...ij", "shapes": [(2, 3), (2, 4)]},
            {"name": "批量外积（3D批次）", "eq": "...i,...j->...ij", "shapes": [(2, 3, 4), (2, 3, 5)]},
            {"name": "批量矩阵乘法", "eq": "...ij,...jk->...ik", "shapes": [(2, 3, 4), (2, 4, 5)]},
            {"name": "批量矩阵乘法（3D批次）", "eq": "...ij,...jk->...ik",
             "shapes": [(2, 3, 4, 5), (2, 3, 5, 6)]},
            {"name": "批量转置", "eq": "...ij->...ji", "shapes": [(2, 3, 4)]},
            {"name": "批量迹", "eq": "...ii->...", "shapes": [(2, 3, 4, 4)]},
            {"name": "批量对角线提取", "eq": "...ii->...i", "shapes": [(2, 3, 4, 4)]},
            {"name": "批量点积", "eq": "...i,...i->...", "shapes": [(2, 3, 4), (2, 3, 4)]},
            {"name": "省略号批量迹被求和 ...ii->i", "eq": "...ii->i", "shapes": [(2, 4, 4)]},
            {"name": "裸省略号恒等 ...->...", "eq": "...->...", "shapes": [(2, 3, 4)]},
            {"name": "裸省略号全求和 ...->", "eq": "...->", "shapes": [(2, 3, 4)]},
            {"name": "省略号隐式 ...", "eq": "...", "shapes": [(2, 3, 4)], "implicit": "...->..."},
            {"name": "省略号+向量 ...,i->...i", "eq": "...,i->...i",
             "shapes": [(2, 3), (4,)]},
            {"name": "省略号广播逐元素 ...,...->...", "eq": "...,...->...",
             "shapes": [(2, 1, 4), (1, 3, 4)]},
            {"name": "输出省略号位于索引之间 ij...->i...j", "eq": "ij...->i...j",
             "shapes": [(2, 3, 4)]},
            {"name": "输入中间省略号+输出省略号 i...j->...ij", "eq": "i...j->...ij",
             "shapes": [(3, 2, 4, 5)]},
            {"name": "双操作数中间省略号 i...j,...jk->i...k", "eq": "i...j,...jk->i...k",
             "shapes": [(2, 3, 4), (3, 4, 5)]},
        ], "省略号操作测试")

    def test_ellipsis_reduction_and_pairing(self):
        """测试省略号维度被求和，以及与其它操作数省略号配对收缩的方程

        numpy 的 c_einsum 路径不支持「输入含省略号而输出不含」的方程（会抛
        ValueError），前向必须路由到 optimize 路径。此处覆盖单/双操作数的多种
        省略号求和形态、省略号批量维广播，以及目标批量维由其他操作数省略号收缩
        而来、或其在前向被独立求和（梯度对其为常数）两种相反情形。
        """
        self._run_cases([
            {"name": "省略号求和 ...i->i", "eq": "...i->i", "shapes": [(2, 3, 4)]},
            {"name": "省略号求和 ...ij->ij", "eq": "...ij->ij", "shapes": [(2, 3, 4)]},
            {"name": "省略号全求和 ...i->", "eq": "...i->", "shapes": [(2, 3, 4)]},
            {"name": "省略号求和并保留末维 ...ij->j", "eq": "...ij->j", "shapes": [(2, 3, 4)]},
            {"name": "省略号迹+全求和 ...ii->", "eq": "...ii->", "shapes": [(2, 3, 4, 4)]},
            {"name": "双操作数省略号全求和 ...i,...i->", "eq": "...i,...i->",
             "shapes": [(2, 3, 4), (2, 3, 4)]},
            {"name": "省略号广播点积 ...i,...i->...", "eq": "...i,...i->...",
             "shapes": [(2, 3, 4), (1, 3, 4)]},
            {"name": "目标批量维被独立求和 ...cdc->", "eq": "...cdc->", "shapes": [(2, 3, 3, 3)]},
            {"name": "省略号配对收缩 ...aaa,...b->ba", "eq": "...aaa,...b->ba",
             "shapes": [(2, 3, 3, 3), (2, 3)]},
            {"name": "省略号配对收缩且批量维广播 ...aaa,...b->ba", "eq": "...aaa,...b->ba",
             "shapes": [(1, 3, 3, 3), (2, 3)]},
            {"name": "省略号配对收缩 ...ij,...ik->...jk", "eq": "...ij,...ik->...jk",
             "shapes": [(2, 3, 4), (2, 3, 5)]},
            {"name": "中间省略号被求和 i...j->ij", "eq": "i...j->ij", "shapes": [(3, 2, 4, 5)]},
            {"name": "中间省略号被求和 ij...k->ijk", "eq": "ij...k->ijk", "shapes": [(2, 3, 4, 5)]},
            {"name": "中间省略号被求和且转置 i...j->ji", "eq": "i...j->ji", "shapes": [(3, 4, 5)]},
            {"name": "末尾省略号被求和 i...->i", "eq": "i...->i", "shapes": [(3, 2, 4)]},
            {"name": "末尾省略号被求和 ij...->i", "eq": "ij...->i", "shapes": [(2, 3, 4)]},
            {"name": "末尾省略号被求和 ij...->ij", "eq": "ij...->ij", "shapes": [(2, 3, 4)]},
            {"name": "中间省略号+重复索引被求和 i...jj->i", "eq": "i...jj->i",
             "shapes": [(2, 3, 4, 4)]},
            {"name": "中间省略号+多操作数 i...j,...->i", "eq": "i...j,...->i",
             "shapes": [(2, 3, 4), (3,)]},
            {"name": "末尾省略号+迹 ii...->", "eq": "ii...->", "shapes": [(4, 4, 3, 2)]},
            {"name": "末尾省略号+批量迹 ii...->...", "eq": "ii...->...", "shapes": [(4, 4, 3, 2)]},
            {"name": "末尾省略号+对角提取 ii...->i", "eq": "ii...->i", "shapes": [(4, 4, 3, 2)]},
        ], "省略号求和与配对测试")

    def test_nonuniform_upstream_grad(self):
        """测试非均匀/非单位上游梯度下的反向传播

        既有用例多用 backward()（隐式全1梯度）或 .sum()（传全1），可能掩盖
        grad_value 被丢弃或与缩并结果错位的问题。此处为每个输出元素注入随机权重，
        覆盖空输出全缩并、点积、迹、对角线提取等各反向路径。
        """
        self._run_cases([
            {"name": "全缩并（空输出）ij,ji->", "eq": "ij,ji->", "shapes": [(3, 4), (4, 3)]},
            {"name": "点积（空输出）i,i->", "eq": "i,i->", "shapes": [(5,), (5,)]},
            {"name": "迹（空输出）ii->", "eq": "ii->", "shapes": [(4, 4)]},
            {"name": "对角线提取 ii->i", "eq": "ii->i", "shapes": [(5, 5)]},
            {"name": "矩阵乘法 ij,jk->ik", "eq": "ij,jk->ik", "shapes": [(3, 4), (4, 5)]},
            {"name": "三操作数缩并 ij,jk,kl->il", "eq": "ij,jk,kl->il",
             "shapes": [(3, 4), (4, 5), (5, 6)]},
            {"name": "多索引重复 iijj->ij", "eq": "iijj->ij", "shapes": [(3, 3, 4, 4)]},
            {"name": "批量矩阵乘法 bij,bjk->bik", "eq": "bij,bjk->bik",
             "shapes": [(2, 3, 4), (2, 4, 5)]},
        ], "非均匀上游梯度测试")

    def test_complex_einsum(self):
        """测试复数张量einsum（含多操作数、归约、省略号与隐式输出）"""
        self._run_cases([
            {"name": "复数矩阵乘法", "eq": "ij,jk->ik", "shapes": [(3, 4), (4, 5)], "complex": True},
            {"name": "复数批量矩阵乘法", "eq": "bij,bjk->bik", "shapes": [(3, 4, 5), (3, 5, 6)], "complex": True},
            {"name": "复数三操作数缩并", "eq": "ij,jk,kl->il", "shapes": [(2, 3), (3, 4), (4, 5)], "complex": True},
            {"name": "复数迹", "eq": "ii->", "shapes": [(5, 5)], "complex": True},
            {"name": "复数对角线提取", "eq": "ii->i", "shapes": [(5, 5)], "complex": True},
            {"name": "复数转置", "eq": "ij->ji", "shapes": [(3, 4)], "complex": True},
            {"name": "复数点积", "eq": "i,i->", "shapes": [(5,), (5,)], "complex": True},
            {"name": "复数外积", "eq": "i,j->ij", "shapes": [(5,), (4,)], "complex": True},
            {"name": "复数求和归约 ij->", "eq": "ij->", "shapes": [(4, 4)], "complex": True},
            {"name": "复数省略号迹 ...ii->...", "eq": "...ii->...", "shapes": [(2, 3, 4, 4)], "complex": True},
            {"name": "复数隐式输出 ij,jk", "eq": "ij,jk", "shapes": [(2, 3), (3, 4)],
             "implicit": "ij,jk->ik", "complex": True},
        ], "复数einsum测试")

    def test_dtype_and_noncontiguous(self):
        """测试 float32 数据类型与非连续输入（[::2] 切片得到的视图）"""
        self._run_cases([
            {"name": "float32数据类型 ij,jk->ik", "eq": "ij,jk->ik", "shapes": [(3, 4), (4, 5)],
             "dtype": np.float32, "check_dtype": np.float32},
            {"name": "float32求和归约 ij->i", "eq": "ij->i", "shapes": [(3, 4)],
             "dtype": np.float32, "check_dtype": np.float32},
            {"name": "float32批量矩阵乘法 bij,bjk->bik", "eq": "bij,bjk->bik",
             "shapes": [(2, 3, 4), (2, 4, 5)], "dtype": np.float32, "check_dtype": np.float32},
            {"name": "非标准步长", "eq": "ij,jk->ik", "shapes": [(3, 4), (4, 5)],
             "noncontiguous": True, "grad": None},
        ], "数据类型与非连续布局测试")

    def test_1d_vector_operations(self):
        """测试1维向量与矩阵的混合运算（矩阵×向量、向量×矩阵、批量矩阵×向量）

        这三类形态未被矩阵-矩阵缩并用例覆盖；此处只做前向比较。
        """
        self._run_cases([
            {"name": "矩阵乘向量 ij,j->i", "eq": "ij,j->i", "shapes": [(3, 5), (5,)], "grad": None},
            {"name": "向量乘矩阵 i,ij->j", "eq": "i,ij->j", "shapes": [(3,), (3, 5)], "grad": None},
            {"name": "批量矩阵乘向量 bij,bj->bi", "eq": "bij,bj->bi",
             "shapes": [(2, 3, 5), (2, 5)], "grad": None},
        ], "1维向量操作测试")


if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行einsum函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}CUDA 可用: {CUDA_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEinsum)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)
