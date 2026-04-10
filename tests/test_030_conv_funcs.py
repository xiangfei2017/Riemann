import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.nn import functional as rm_func
    # 从rm.cuda获取cupy引用和CUDA可用性
    CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
    cp = rm.cuda.cp
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    import torch.nn.functional as torch_func
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
    print("警告: 无法导入PyTorch，将只测试riemann的卷积相关函数")
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
            torch_data = torch_result.detach().numpy()
    except Exception as e:
        print(f"比较值转换错误: {e}")
        return False
    
    # 处理形状不匹配的情况
    try:
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

# 测试卷积相关函数类
class TestConvFunctions(unittest.TestCase):
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
    
    def test_4D_unfold(self):
        """测试unfold函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本unfold",
                "input_shape": (2, 3, 4, 4),
                "kernel_size": (2, 2),
                "dilation": 1,
                "padding": 0,
                "stride": 1
            },
            {
                "name": "带填充的unfold",
                "input_shape": (2, 3, 4, 4),
                "kernel_size": (2, 2),
                "dilation": 1,
                "padding": 1,
                "stride": 1
            },
            {
                "name": "带步长的unfold",
                "input_shape": (2, 3, 4, 4),
                "kernel_size": (2, 2),
                "dilation": 1,
                "padding": 0,
                "stride": 2
            },
            {
                "name": "带膨胀的unfold",
                "input_shape": (2, 3, 4, 4),
                "kernel_size": (2, 2),
                "dilation": 2,
                "padding": 2,
                "stride": 1
            },
            {
                "name": "正方形卷积核unfold",
                "input_shape": (2, 3, 5, 5),
                "kernel_size": 3,
                "dilation": 1,
                "padding": 1,
                "stride": 1
            },
            {
                "name": "非对称kernel_size unfold",
                "input_shape": (2, 3, 7, 9),
                "kernel_size": (3, 5),
                "dilation": 1,
                "padding": 0,
                "stride": 1
            },
            {
                "name": "非对称stride unfold",
                "input_shape": (2, 3, 7, 9),
                "kernel_size": (3, 3),
                "dilation": 1,
                "padding": 0,
                "stride": (2, 3)
            },
            {
                "name": "非对称padding unfold",
                "input_shape": (2, 3, 7, 9),
                "kernel_size": (3, 3),
                "dilation": 1,
                "padding": (1, 2),
                "stride": 1
            },
            {
                "name": "非对称dilation unfold",
                "input_shape": (2, 3, 7, 9),
                "kernel_size": (3, 3),
                "dilation": (1, 2),
                "padding": (1, 3),
                "stride": 1
            },
            {
                "name": "复杂参数组合unfold",
                "input_shape": (2, 3, 11, 13),
                "kernel_size": (3, 5),
                "dilation": (1, 2),
                "padding": (2, 3),
                "stride": (2, 3)
            },
            {
                "name": "kernel_size等于输入大小unfold",
                "input_shape": (2, 3, 5, 5),
                "kernel_size": (5, 5),
                "dilation": 1,
                "padding": 0,
                "stride": 1
            },
            {
                "name": "更大输入尺寸unfold",
                "input_shape": (2, 3, 16, 16),
                "kernel_size": (3, 3),
                "dilation": 1,
                "padding": 1,
                "stride": 2
            },
            {
                "name": "全非对称参数unfold",
                "input_shape": (2, 3, 13, 17),
                "kernel_size": (2, 4),
                "dilation": (2, 1),
                "padding": (1, 2),
                "stride": (3, 2)
            }

        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"unfold - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    np_input = np.random.randn(*case["input_shape"])
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                        else:
                            torch_input = None
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                        else:
                            torch_input = None
                    
                    # 前向传播测试
                    rm_result = rm_func.unfold(
                        rm_input,
                        kernel_size=case["kernel_size"],
                        dilation=case["dilation"],
                        padding=case["padding"],
                        stride=case["stride"]
                    )
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_result = torch_func.unfold(
                            torch_input,
                            kernel_size=case["kernel_size"],
                            dilation=case["dilation"],
                            padding=case["padding"],
                            stride=case["stride"]
                        )
                    
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result, atol=1e-5, rtol=1e-5)
                    if not forward_passed and TORCH_AVAILABLE:
                        print(f"前向传播失败: {case_name}")
                        print(f"Riemann结果形状: {rm_result.shape}")
                        print(f"PyTorch结果形状: {torch_result.shape}")
                        print(f"Riemann结果前10个值: {rm_result.data.flatten()[:10]}")
                        print(f"PyTorch结果前10个值: {torch_result.data.flatten()[:10]}")
                        print(f"最大差异: {abs(rm_result.data - torch_result.data).max()}")
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        backward_passed = compare_values(rm_input.grad, torch_input.grad, atol=1e-5, rtol=1e-5)
                        if not backward_passed:
                            print(f"反向传播失败: {case_name}")
                            print(f"Riemann梯度前10个值: {rm_input.grad.data.flatten()[:10]}")
                            print(f"PyTorch梯度前10个值: {torch_input.grad.data.flatten()[:10]}")
                            print(f"最大差异: {abs(rm_input.grad.data - torch_input.grad.data).max()}")
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed and TORCH_AVAILABLE:
                            print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                            print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                            print(f"  Riemann形状: {rm_result.shape}, PyTorch形状: {torch_result.shape if torch_result is not None else 'N/A'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"unfold测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_4D_fold(self):
        """测试fold函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本fold",
                "input_shape": (2, 3, 4, 4),
                "kernel_size": (2, 2),
                "dilation": 1,
                "padding": 0,
                "stride": 1
            },
            {
                "name": "带填充的fold",
                "input_shape": (2, 3, 4, 4),
                "kernel_size": (2, 2),
                "dilation": 1,
                "padding": 1,
                "stride": 1
            },
            {
                "name": "带步长的fold",
                "input_shape": (2, 3, 4, 4),
                "kernel_size": (2, 2),
                "dilation": 1,
                "padding": 0,
                "stride": 2
            },
            {
                "name": "带膨胀的fold",
                "input_shape": (2, 3, 4, 4),
                "kernel_size": (2, 2),
                "dilation": 2,
                "padding": 2,
                "stride": 1
            },
            {
                "name": "正方形卷积核fold",
                "input_shape": (2, 3, 5, 5),
                "kernel_size": 3,
                "dilation": 1,
                "padding": 1,
                "stride": 1
            },
            {
                "name": "非对称kernel_size fold",
                "input_shape": (2, 3, 7, 9),
                "kernel_size": (3, 5),
                "dilation": 1,
                "padding": 0,
                "stride": 1
            },
            {
                "name": "非对称stride fold",
                "input_shape": (2, 3, 7, 9),
                "kernel_size": (3, 3),
                "dilation": 1,
                "padding": 0,
                "stride": (2, 3)
            },
            {
                "name": "非对称padding fold",
                "input_shape": (2, 3, 7, 9),
                "kernel_size": (3, 3),
                "dilation": 1,
                "padding": (1, 2),
                "stride": 1
            },
            {
                "name": "非对称dilation fold",
                "input_shape": (2, 3, 7, 9),
                "kernel_size": (3, 3),
                "dilation": (1, 2),
                "padding": (1, 3),
                "stride": 1
            },
            {
                "name": "复杂参数组合fold",
                "input_shape": (2, 3, 11, 13),
                "kernel_size": (3, 5),
                "dilation": (1, 2),
                "padding": (2, 3),
                "stride": (2, 3)
            },
            {
                "name": "kernel_size等于输入大小fold",
                "input_shape": (2, 3, 5, 5),
                "kernel_size": (5, 5),
                "dilation": 1,
                "padding": 0,
                "stride": 1
            },
            {
                "name": "更大输入尺寸fold",
                "input_shape": (2, 3, 16, 16),
                "kernel_size": (3, 3),
                "dilation": 1,
                "padding": 1,
                "stride": 2
            },
            {
                "name": "全非对称参数fold",
                "input_shape": (2, 3, 13, 17),
                "kernel_size": (2, 4),
                "dilation": (2, 1),
                "padding": (1, 2),
                "stride": (3, 2)
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"fold - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    np_input = np.random.randn(*case["input_shape"])
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                        else:
                            torch_input = None
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                        else:
                            torch_input = None
                
                    # 首先使用unfold获取展开后的张量
                    rm_unfolded = rm_func.unfold(
                        rm_input,
                        kernel_size=case["kernel_size"],
                        dilation=case["dilation"],
                        padding=case["padding"],
                        stride=case["stride"]
                    )
                    torch_unfolded = None
                    if TORCH_AVAILABLE:
                        torch_unfolded = torch_func.unfold(
                            torch_input,
                            kernel_size=case["kernel_size"],
                            dilation=case["dilation"],
                            padding=case["padding"],
                            stride=case["stride"]
                        )
                    
                    # 前向传播测试（fold操作）
                    rm_result = rm_func.fold(
                        rm_unfolded,
                        output_size=case["input_shape"][2:],  # (H, W)
                        kernel_size=case["kernel_size"],
                        dilation=case["dilation"],
                        padding=case["padding"],
                        stride=case["stride"]
                    )
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_result = torch_func.fold(
                            torch_unfolded,
                            output_size=case["input_shape"][2:],  # (H, W)
                            kernel_size=case["kernel_size"],
                            dilation=case["dilation"],
                            padding=case["padding"],
                            stride=case["stride"]
                        )
                    
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result, atol=1e-5, rtol=1e-5)
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed and TORCH_AVAILABLE:
                            print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                            print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                            print(f"  Riemann形状: {rm_result.shape}, PyTorch形状: {torch_result.shape if torch_result is not None else 'N/A'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"fold测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_conv2d(self):
        """测试conv2d函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本卷积",
                "input_shape": (1, 2, 4, 4),
                "output_channels": 4,
                "kernel_size": (2, 2),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "带填充的卷积",
                "input_shape": (1, 2, 4, 4),
                "output_channels": 4,
                "kernel_size": (2, 2),
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "带步长的卷积",
                "input_shape": (1, 2, 4, 4),
                "output_channels": 4,
                "kernel_size": (2, 2),
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "分组卷积",
                "input_shape": (1, 2, 4, 4),
                "output_channels": 4,
                "kernel_size": (2, 2),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 2
            },
            {
                "name": "带膨胀的卷积",
                "input_shape": (1, 2, 5, 5),
                "output_channels": 4,
                "kernel_size": (2, 2),
                "stride": 1,
                "padding": 2,
                "dilation": 2,
                "groups": 1
            },
            {
                "name": "正方形卷积核卷积",
                "input_shape": (1, 3, 5, 5),
                "output_channels": 6,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "非对称kernel卷积",
                "input_shape": (1, 3, 7, 9),
                "output_channels": 6,
                "kernel_size": (3, 5),
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "非对称stride卷积",
                "input_shape": (1, 2, 8, 8),
                "output_channels": 4,
                "kernel_size": (2, 2),
                "stride": (2, 3),
                "padding": 1,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "复杂参数组合卷积",
                "input_shape": (1, 4, 10, 10),
                "output_channels": 8,
                "kernel_size": (3, 3),
                "stride": 2,
                "padding": 1,
                "dilation": 2,
                "groups": 2
            },
            {
                "name": "边缘情况卷积（kernel_size=输入大小）",
                "input_shape": (1, 2, 4, 4),
                "output_channels": 3,
                "kernel_size": (4, 4),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "大步长卷积",
                "input_shape": (1, 3, 16, 16),
                "output_channels": 6,
                "kernel_size": (3, 3),
                "stride": 4,
                "padding": 2,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "全分组卷积",
                "input_shape": (1, 4, 8, 8),
                "output_channels": 4,
                "kernel_size": (2, 2),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 4
            },
            {
                "name": "大padding卷积",
                "input_shape": (1, 2, 6, 6),
                "output_channels": 4,
                "kernel_size": (3, 3),
                "stride": 1,
                "padding": 2,
                "dilation": 1,
                "groups": 1
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"conv2d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 解包测试参数
                    input_shape = case["input_shape"]
                    N, C_in, H_in, W_in = input_shape
                    C_out = case["output_channels"]
                    kernel_size = case["kernel_size"]
                    stride = case["stride"]
                    padding = case["padding"]
                    dilation = case["dilation"]
                    groups = case["groups"]
                    
                    # 创建测试数据
                    np_input = np.random.randn(*input_shape)
                    np_weight = np.random.randn(C_out, C_in // groups, kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size, kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size)
                    np_bias = np.random.randn(C_out)
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        rm_weight = rm.tensor(np_weight, requires_grad=True)
                        rm_bias = rm.tensor(np_bias, requires_grad=True)
                        
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                            torch_weight = torch.tensor(np_weight, requires_grad=True)
                            torch_bias = torch.tensor(np_bias, requires_grad=True)
                        else:
                            torch_input = None
                            torch_weight = None
                            torch_bias = None
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        rm_weight = rm.tensor(np_weight, requires_grad=True, device=device)
                        rm_bias = rm.tensor(np_bias, requires_grad=True, device=device)
                        
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                            torch_weight = torch.tensor(np_weight, requires_grad=True, device=device)
                            torch_bias = torch.tensor(np_bias, requires_grad=True, device=device)
                        else:
                            torch_input = None
                            torch_weight = None
                            torch_bias = None
                
                    # 前向传播测试
                    rm_result = rm_func.conv2d(
                        rm_input, rm_weight, rm_bias,
                        stride=stride, padding=padding, dilation=dilation, groups=groups
                    )
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_result = torch_func.conv2d(
                            torch_input, torch_weight, torch_bias,
                            stride=stride, padding=padding, dilation=dilation, groups=groups
                        )
                    
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result, atol=1e-5, rtol=1e-5)
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        grad_input_passed = compare_values(rm_input.grad, torch_input.grad)
                        grad_weight_passed = compare_values(rm_weight.grad, torch_weight.grad)
                        grad_bias_passed = compare_values(rm_bias.grad, torch_bias.grad)
                        backward_passed = grad_input_passed and grad_weight_passed and grad_bias_passed
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed and TORCH_AVAILABLE:
                            print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                            print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                            print(f"  输入梯度: {'通过' if grad_input_passed else '失败'}")
                            print(f"  权重梯度: {'通过' if grad_weight_passed else '失败'}")
                            print(f"  偏置梯度: {'通过' if grad_bias_passed else '失败'}")
                            print(f"  Riemann形状: {rm_result.shape}, PyTorch形状: {torch_result.shape if torch_result is not None else 'N/A'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"conv2d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
            
    def test_conv1d(self):
        """测试conv1d函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本卷积",
                "input_shape": (1, 2, 4),
                "output_channels": 4,
                "kernel_size": (2,),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "带填充的卷积",
                "input_shape": (1, 2, 4),
                "output_channels": 4,
                "kernel_size": (2,),
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "带步长的卷积",
                "input_shape": (1, 2, 4),
                "output_channels": 4,
                "kernel_size": (2,),
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "分组卷积",
                "input_shape": (1, 4, 4),
                "output_channels": 4,
                "kernel_size": (2,),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 2
            },
            {
                "name": "带膨胀的卷积",
                "input_shape": (1, 2, 5),
                "output_channels": 4,
                "kernel_size": (2,),
                "stride": 1,
                "padding": 2,
                "dilation": 2,
                "groups": 1
            },
            {
                "name": "标量卷积核卷积",
                "input_shape": (1, 3, 5),
                "output_channels": 6,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "大padding卷积",
                "input_shape": (1, 2, 6),
                "output_channels": 4,
                "kernel_size": (3,),
                "stride": 1,
                "padding": 2,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "复杂参数组合卷积",
                "input_shape": (1, 4, 10),
                "output_channels": 8,
                "kernel_size": (3,),
                "stride": 2,
                "padding": 1,
                "dilation": 2,
                "groups": 2
            },
            {
                "name": "边缘情况卷积（kernel_size=输入大小）",
                "input_shape": (1, 2, 4),
                "output_channels": 3,
                "kernel_size": (4,),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "大步长卷积",
                "input_shape": (1, 3, 15),
                "output_channels": 6,
                "kernel_size": (3,),
                "stride": 4,
                "padding": 2,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "全分组卷积",
                "input_shape": (1, 4, 8),
                "output_channels": 4,
                "kernel_size": (2,),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 4
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"conv1d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 解包测试参数
                    input_shape = case["input_shape"]
                    N, C_in, L_in = input_shape
                    C_out = case["output_channels"]
                    kernel_size = case["kernel_size"]
                    stride = case["stride"]
                    padding = case["padding"]
                    dilation = case["dilation"]
                    groups = case["groups"]
                    
                    # 处理标量kernel_size
                    if isinstance(kernel_size, int):
                        kernel_size = (kernel_size,)
                    K = kernel_size[0]
                    
                    # 创建测试数据
                    np_input = np.random.randn(*input_shape)
                    np_weight = np.random.randn(C_out, C_in // groups, K)
                    np_bias = np.random.randn(C_out)
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        rm_weight = rm.tensor(np_weight, requires_grad=True)
                        rm_bias = rm.tensor(np_bias, requires_grad=True)
                        
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                            torch_weight = torch.tensor(np_weight, requires_grad=True)
                            torch_bias = torch.tensor(np_bias, requires_grad=True)
                        else:
                            torch_input = None
                            torch_weight = None
                            torch_bias = None
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        rm_weight = rm.tensor(np_weight, requires_grad=True, device=device)
                        rm_bias = rm.tensor(np_bias, requires_grad=True, device=device)
                        
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                            torch_weight = torch.tensor(np_weight, requires_grad=True, device=device)
                            torch_bias = torch.tensor(np_bias, requires_grad=True, device=device)
                        else:
                            torch_input = None
                            torch_weight = None
                            torch_bias = None
                
                    # 前向传播测试
                    rm_result = rm_func.conv1d(
                        rm_input, rm_weight, rm_bias,
                        stride=stride, padding=padding, dilation=dilation, groups=groups
                    )
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_result = torch_func.conv1d(
                            torch_input, torch_weight, torch_bias,
                            stride=stride, padding=padding, dilation=dilation, groups=groups
                        )
                    
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result, atol=1e-5, rtol=1e-5)
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        grad_input_passed = compare_values(rm_input.grad, torch_input.grad)
                        grad_weight_passed = compare_values(rm_weight.grad, torch_weight.grad)
                        grad_bias_passed = compare_values(rm_bias.grad, torch_bias.grad)
                        backward_passed = grad_input_passed and grad_weight_passed and grad_bias_passed
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed and TORCH_AVAILABLE:
                            print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                            print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                            print(f"  输入梯度: {'通过' if grad_input_passed else '失败'}")
                            print(f"  权重梯度: {'通过' if grad_weight_passed else '失败'}")
                            print(f"  偏置梯度: {'通过' if grad_bias_passed else '失败'}")
                            print(f"  Riemann形状: {rm_result.shape}, PyTorch形状: {torch_result.shape if torch_result is not None else 'N/A'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"conv1d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        
    def test_max_pool2d(self):
        """测试max_pool2d函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本max_pool2d",
                "input_shape": (1, 3, 4, 4),
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "步长大于1",
                "input_shape": (1, 3, 6, 6),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "带填充",
                "input_shape": (1, 3, 4, 4),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "带膨胀",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "dilation": 2,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "使用ceil_mode",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "带填充+使用ceil_mode",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 2,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "ceil_mode边缘窗口1",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "ceil_mode边缘窗口2",
                "input_shape": (1, 3, 7, 7),
                "kernel_size": 3,
                "stride": 3,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "ceil_mode带不同padding",
                "input_shape": (1, 3, 8, 8),
                "kernel_size": 4,
                "stride": 3,
                "padding": 2,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "ceil_mode+膨胀",
                "input_shape": (1, 3, 12, 12),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 2,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "大kernel_size+ceil_mode",
                "input_shape": (1, 3, 6, 6),
                "kernel_size": 4,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "非对称kernel_size",
                "input_shape": (1, 3, 6, 7),
                "kernel_size": (2, 3),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "非对称stride",
                "input_shape": (1, 3, 6, 6),
                "kernel_size": 2,
                "stride": (2, 1),
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "非对称padding",
                "input_shape": (1, 3, 4, 5),
                "kernel_size": 3,
                "stride": 1,
                "padding": (1, 1),
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "非对称参数+ceil_mode",
                "input_shape": (1, 3, 7, 8),
                "kernel_size": (2, 3),
                "stride": (2, 1),
                "padding": (1, 0),
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "边缘窗口+膨胀",
                "input_shape": (1, 3, 11, 11),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "dilation": 2,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "大stride+小kernel",
                "input_shape": (1, 3, 9, 9),
                "kernel_size": 2,
                "stride": 3,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "小stride+大kernel",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 5,
                "stride": 2,
                "padding": 2,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "无padding+边缘窗口",
                "input_shape": (1, 3, 13, 13),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "有padding+边缘窗口",
                "input_shape": (1, 3, 12, 12),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "不同输入大小+ceil_mode",
                "input_shape": (1, 3, 15, 15),
                "kernel_size": 4,
                "stride": 3,
                "padding": 1,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "大kernel大stride+ceil_mode",
                "input_shape": (1, 3, 9, 10),
                "kernel_size": 4,
                "stride": 3,
                "padding": 1,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "非对称kernel+padding+ceil_mode",
                "input_shape": (1, 3, 7, 8),
                "kernel_size": (3, 2),
                "stride": (2, 2),
                "padding": (1, 1),
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "不同输入高度宽度+ceil_mode",
                "input_shape": (1, 3, 11, 14),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "dilation>1+ceil_mode边缘窗口",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 2,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "kernel_size=stride+ceil_mode",
                "input_shape": (1, 3, 13, 13),
                "kernel_size": 4,
                "stride": 4,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "极小kernel+大stride+ceil_mode",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 1,
                "stride": 3,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "dilation=2+ceil_mode边缘窗口",
                "input_shape": (1, 3, 12, 12),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 2,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "非对称dilation+ceil_mode",
                "input_shape": (1, 3, 10, 11),
                "kernel_size": (3, 2),
                "stride": (2, 3),
                "padding": (1, 1),
                "dilation": (2, 1),
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "大padding+小kernel+ceil_mode",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 2,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"max_pool2d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 解包测试参数
                    input_shape = case["input_shape"]
                    kernel_size = case["kernel_size"]
                    stride = case["stride"]
                    padding = case["padding"]
                    dilation = case["dilation"]
                    return_indices = case["return_indices"]
                    ceil_mode = case["ceil_mode"]
                    
                    # 创建测试数据
                    np_input = np.random.randn(*input_shape)
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                
                    # 前向传播测试
                    rm_result = None
                    torch_result = None
                    torch_indices = None
                    
                    # 计算前向传播
                    if return_indices:
                        rm_result, rm_indices = rm_func.max_pool2d(
                            rm_input, kernel_size=kernel_size, stride=stride, padding=padding, 
                            dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode
                        )
                    else:
                        rm_result = rm_func.max_pool2d(
                            rm_input, kernel_size=kernel_size, stride=stride, padding=padding, 
                            dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode
                        )
                    
                    if TORCH_AVAILABLE:
                        if return_indices:
                            torch_result, torch_indices = torch_func.max_pool2d(
                                torch_input, kernel_size=kernel_size, stride=stride, padding=padding, 
                                dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode
                            )
                        else:
                            torch_result = torch_func.max_pool2d(
                                torch_input, kernel_size=kernel_size, stride=stride, padding=padding, 
                                dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode
                            )
                    
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result, atol=1e-5, rtol=1e-5)
                    
                    # 比较索引
                    indices_passed = True
                    if TORCH_AVAILABLE and return_indices:
                        indices_passed = compare_values(rm_indices, torch_indices)
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and indices_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"max_pool2d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        
    def test_avg_pool2d(self):
        """测试avg_pool2d函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本avg_pool2d",
                "input_shape": (1, 3, 4, 4),
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "步长大于1",
                "input_shape": (1, 3, 6, 6),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "带填充",
                "input_shape": (1, 3, 4, 4),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "不包含填充区域",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "ceil_mode": False,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "自定义除数",
                "input_shape": (1, 3, 4, 4),
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": 3
            },
            {
                "name": "使用ceil_mode",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode边缘窗口1",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode边缘窗口2",
                "input_shape": (1, 3, 7, 7),
                "kernel_size": 3,
                "stride": 3,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode带不同padding",
                "input_shape": (1, 3, 8, 8),
                "kernel_size": 4,
                "stride": 3,
                "padding": 2,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode+count_include_pad=False",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "大kernel_size+ceil_mode",
                "input_shape": (1, 3, 6, 6),
                "kernel_size": 4,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "非对称kernel_size",
                "input_shape": (1, 3, 6, 7),
                "kernel_size": (2, 3),
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "非对称stride",
                "input_shape": (1, 3, 6, 6),
                "kernel_size": 2,
                "stride": (2, 1),
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "非对称padding",
                "input_shape": (1, 3, 5, 6),
                "kernel_size": 4,
                "stride": 1,
                "padding": (1, 2),
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "非对称参数+ceil_mode",
                "input_shape": (1, 3, 7, 8),
                "kernel_size": (2, 3),
                "stride": (2, 1),
                "padding": (1, 0),
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode+padding+count_include_pad=False",
                "input_shape": (1, 3, 8, 9),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "大stride+小kernel",
                "input_shape": (1, 3, 9, 9),
                "kernel_size": 2,
                "stride": 3,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "小stride+大kernel",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 5,
                "stride": 2,
                "padding": 2,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "无padding+边缘窗口+count_include_pad=False",
                "input_shape": (1, 3, 13, 13),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "有padding+边缘窗口+count_include_pad=False",
                "input_shape": (1, 3, 12, 12),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "自定义除数+ceil_mode",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": 4
            },
            {
                "name": "不同输入大小+ceil_mode",
                "input_shape": (1, 3, 15, 15),
                "kernel_size": 4,
                "stride": 3,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "count_include_pad=False+小kernel",
                "input_shape": (1, 3, 4, 4),
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "count_include_pad=False+非对称",
                "input_shape": (1, 3, 7, 6),
                "kernel_size": (3, 2),
                "stride": 2,
                "padding": (1, 0),
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "大kernel大stride+ceil_mode+无填充",
                "input_shape": (1, 3, 9, 10),
                "kernel_size": 4,
                "stride": 3,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "大kernel大stride+ceil_mode+有填充",
                "input_shape": (1, 3, 9, 10),
                "kernel_size": 4,
                "stride": 3,
                "padding": 2,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "非对称kernel+stride+ceil_mode",
                "input_shape": (1, 3, 7, 8),
                "kernel_size": (3, 2),
                "stride": (2, 3),
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "不同输入高度宽度+ceil_mode",
                "input_shape": (1, 3, 11, 14),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "ceil_mode+count_include_pad=False+自定义除数",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": 4
            },
            {
                "name": "不同padding组合+ceil_mode",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 4,
                "stride": 2,
                "padding": (2, 1),
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "非对称padding+ceil_mode+count_include_pad=False",
                "input_shape": (1, 3, 9, 8),
                "kernel_size": 5,
                "stride": 2,
                "padding": (1, 2),
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "count_include_pad=False+大kernel+ceil_mode",
                "input_shape": (1, 3, 7, 7),
                "kernel_size": 5,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "自定义除数+count_include_pad=False",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": 9
            },
            {
                "name": "对称kernel+非对称stride+ceil_mode",
                "input_shape": (1, 3, 11, 12),
                "kernel_size": 3,
                "stride": (2, 3),
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "极小kernel+大stride+ceil_mode",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 1,
                "stride": 3,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "dilation=2+ceil_mode+count_include_pad=True",
                "input_shape": (1, 3, 12, 12),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "dilation=2+ceil_mode+count_include_pad=False",
                "input_shape": (1, 3, 12, 12),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "非对称dilation+ceil_mode",
                "input_shape": (1, 3, 10, 11),
                "kernel_size": (3, 2),
                "stride": (2, 3),
                "padding": (1, 1),
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "大padding+小kernel+ceil_mode",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 2,
                "stride": 1,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "大padding+小kernel+ceil_mode+count_include_pad=False",
                "input_shape": (1, 3, 5, 5),
                "kernel_size": 2,
                "stride": 1,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "非对称参数组合+ceil_mode+count_include_pad=False",
                "input_shape": (1, 3, 8, 9),
                "kernel_size": (3, 4),
                "stride": (2, 3),
                "padding": (1, 2),
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "边缘窗口+极小kernel",
                "input_shape": (1, 3, 4, 4),
                "kernel_size": 1,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "边缘窗口+极大kernel",
                "input_shape": (1, 3, 6, 6),
                "kernel_size": 5,
                "stride": 1,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "测试原始bug场景1",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "测试原始bug场景2",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "小kernel+大stride+ceil_mode+有填充",
                "input_shape": (1, 3, 13, 13),
                "kernel_size": 2,
                "stride": 3,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "相同kernel_stride+ceil_mode+无填充",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 3,
                "stride": 3,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "相同kernel_stride+ceil_mode+有填充",
                "input_shape": (1, 3, 10, 10),
                "kernel_size": 3,
                "stride": 3,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"avg_pool2d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 解包测试参数
                    input_shape = case["input_shape"]
                    kernel_size = case["kernel_size"]
                    stride = case["stride"]
                    padding = case["padding"]
                    ceil_mode = case["ceil_mode"]
                    count_include_pad = case["count_include_pad"]
                    divisor_override = case.get("divisor_override", None)
                    
                    # 创建测试数据
                    np_input = np.random.randn(*input_shape)
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                
                    # 前向传播测试
                    rm_result = rm_func.avg_pool2d(
                        rm_input, kernel_size=kernel_size, stride=stride, padding=padding,
                        ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                        divisor_override=divisor_override
                    )
                    
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_result = torch_func.avg_pool2d(
                            torch_input, kernel_size=kernel_size, stride=stride, padding=padding,
                            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                            divisor_override=divisor_override
                        )
                    
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result, atol=1e-5, rtol=1e-5)
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        backward_passed = compare_values(rm_input.grad, torch_input.grad, atol=1e-5, rtol=1e-5)
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"avg_pool2d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_conv3d(self):
        """测试conv3d函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本3D卷积",
                "input_shape": (1, 2, 4, 4, 4),
                "output_channels": 4,
                "kernel_size": (2, 2, 2),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "带填充的3D卷积",
                "input_shape": (1, 2, 4, 4, 4),
                "output_channels": 4,
                "kernel_size": (2, 2, 2),
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "带步长的3D卷积",
                "input_shape": (1, 2, 4, 4, 4),
                "output_channels": 4,
                "kernel_size": (2, 2, 2),
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "分组3D卷积",
                "input_shape": (1, 2, 4, 4, 4),
                "output_channels": 4,
                "kernel_size": (2, 2, 2),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 2
            },
            {
                "name": "带膨胀的3D卷积",
                "input_shape": (1, 2, 5, 5, 5),
                "output_channels": 4,
                "kernel_size": (2, 2, 2),
                "stride": 1,
                "padding": 2,
                "dilation": 2,
                "groups": 1
            },
            {
                "name": "立方体卷积核3D卷积",
                "input_shape": (1, 3, 5, 5, 5),
                "output_channels": 6,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "非对称kernel的3D卷积",
                "input_shape": (1, 3, 7, 9, 5),
                "output_channels": 6,
                "kernel_size": (3, 5, 2),
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "非对称stride的3D卷积",
                "input_shape": (1, 2, 8, 8, 8),
                "output_channels": 4,
                "kernel_size": (2, 2, 2),
                "stride": (2, 1, 2),
                "padding": 0,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "非对称padding的3D卷积",
                "input_shape": (1, 3, 6, 6, 6),
                "output_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": (1, 1, 0),
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "大kernel的3D卷积",
                "input_shape": (1, 2, 8, 8, 8),
                "output_channels": 3,
                "kernel_size": (4, 4, 4),
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "大stride的3D卷积",
                "input_shape": (1, 2, 10, 10, 10),
                "output_channels": 4,
                "kernel_size": (2, 2, 2),
                "stride": (3, 3, 3),
                "padding": 0,
                "dilation": 1,
                "groups": 1
            },
            {
                "name": "非对称dilation的3D卷积",
                "input_shape": (1, 2, 7, 8, 9),
                "output_channels": 3,
                "kernel_size": (2, 3, 2),
                "stride": 1,
                "padding": 1,
                "dilation": (1, 2, 1),
                "groups": 1
            },
            {
                "name": "复杂组合参数的3D卷积",
                "input_shape": (1, 4, 9, 11, 7),
                "output_channels": 8,
                "kernel_size": (3, 4, 2),
                "stride": (2, 1, 3),
                "padding": (1, 1, 0),
                "dilation": (1, 1, 2),
                "groups": 1
            },
            {
                "name": "深度方向特殊参数的3D卷积",
                "input_shape": (1, 3, 12, 6, 6),
                "output_channels": 6,
                "kernel_size": (5, 3, 3),
                "stride": (2, 1, 1),
                "padding": (2, 1, 1),
                "dilation": 1,
                "groups": 1
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"conv3d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 解包测试参数
                    input_shape = case["input_shape"]
                    output_channels = case["output_channels"]
                    kernel_size = case["kernel_size"]
                    stride = case["stride"]
                    padding = case["padding"]
                    dilation = case["dilation"]
                    groups = case["groups"]
                    
                    # 创建测试数据
                    np_input = np.random.randn(*input_shape).astype(np.float32)
                    
                    # 计算权重形状
                    input_channels = input_shape[1]
                    if isinstance(kernel_size, int):
                        weight_shape = (output_channels, input_channels // groups, kernel_size, kernel_size, kernel_size)
                    else:
                        weight_shape = (output_channels, input_channels // groups, *kernel_size)
                    
                    np_weight = np.random.randn(*weight_shape).astype(np.float32)
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        rm_weight = rm.tensor(np_weight, requires_grad=True)
                        
                        torch_input = None
                        torch_weight = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                            torch_weight = torch.tensor(np_weight, requires_grad=True)
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        rm_weight = rm.tensor(np_weight, requires_grad=True, device=device)
                        
                        torch_input = None
                        torch_weight = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                            torch_weight = torch.tensor(np_weight, requires_grad=True, device=device)
                
                    # 前向传播测试
                    rm_result = rm_func.conv3d(
                        rm_input, rm_weight, bias=None, stride=stride, 
                        padding=padding, dilation=dilation, groups=groups
                    )
                    
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_result = torch_func.conv3d(
                            torch_input, torch_weight, bias=None, stride=stride,
                            padding=padding, dilation=dilation, groups=groups
                        )
                    
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result, atol=1e-5, rtol=1e-5)
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较输入梯度
                        input_grad_passed = compare_values(rm_input.grad, torch_input.grad, atol=1e-5, rtol=1e-5)
                        # 比较权重梯度
                        weight_grad_passed = compare_values(rm_weight.grad, torch_weight.grad, atol=1e-5, rtol=1e-5)
                        
                        backward_passed = input_grad_passed and weight_grad_passed
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"conv3d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_max_pool3d(self):
        """测试max_pool3d函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本max_pool3d",
                "input_shape": (1, 3, 4, 4, 4),
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "步长大于1",
                "input_shape": (1, 3, 6, 6, 6),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "带填充",
                "input_shape": (1, 3, 4, 4, 4),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "带膨胀",
                "input_shape": (1, 3, 10, 10, 10),
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "dilation": 2,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "使用ceil_mode",
                "input_shape": (1, 3, 5, 5, 5),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "立方体kernel",
                "input_shape": (1, 2, 8, 8, 8),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "非对称kernel",
                "input_shape": (1, 3, 7, 9, 5),
                "kernel_size": (2, 3, 4),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "非对称stride",
                "input_shape": (1, 2, 8, 8, 8),
                "kernel_size": 2,
                "stride": (2, 1, 2),
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "非对称padding",
                "input_shape": (1, 3, 6, 6, 6),
                "kernel_size": 3,
                "stride": 1,
                "padding": (1, 1, 0),
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "大kernel",
                "input_shape": (1, 2, 8, 8, 8),
                "kernel_size": (4, 4, 4),
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "大stride",
                "input_shape": (1, 2, 10, 10, 10),
                "kernel_size": 2,
                "stride": (3, 3, 3),
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "非对称dilation",
                "input_shape": (1, 2, 7, 8, 9),
                "kernel_size": (2, 3, 2),
                "stride": 1,
                "padding": 1,
                "dilation": (1, 2, 1),
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "复杂组合参数",
                "input_shape": (1, 4, 9, 11, 7),
                "kernel_size": (3, 4, 2),
                "stride": (2, 1, 3),
                "padding": (1, 1, 0),
                "dilation": (1, 1, 2),
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "ceil_mode+非对称参数",
                "input_shape": (1, 3, 7, 8, 9),
                "kernel_size": (3, 2, 4),
                "stride": (2, 1, 2),
                "padding": (1, 0, 1),
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "深度方向特殊参数",
                "input_shape": (1, 3, 12, 6, 6),
                "kernel_size": (5, 3, 3),
                "stride": (2, 1, 1),
                "padding": (2, 1, 1),
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "极小kernel",
                "input_shape": (1, 3, 5, 5, 5),
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "极大kernel",
                "input_shape": (1, 2, 6, 6, 6),
                "kernel_size": 5,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"max_pool3d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 解包测试参数
                    input_shape = case["input_shape"]
                    kernel_size = case["kernel_size"]
                    stride = case["stride"]
                    padding = case["padding"]
                    dilation = case["dilation"]
                    return_indices = case["return_indices"]
                    ceil_mode = case["ceil_mode"]
                    
                    # 创建测试数据
                    np_input = np.random.randn(*input_shape).astype(np.float32)
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                    
                    # 前向传播测试
                    rm_result = rm_func.max_pool3d(
                        rm_input, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, return_indices=return_indices,
                        ceil_mode=ceil_mode
                    )
                    
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_result = torch_func.max_pool3d(
                            torch_input, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, return_indices=return_indices,
                            ceil_mode=ceil_mode
                        )
                
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result[0] if return_indices else rm_result, 
                                                torch_result[0] if return_indices else torch_result, atol=1e-5, rtol=1e-5)
                    
                    # 如果返回索引，也比较索引
                    if return_indices and TORCH_AVAILABLE:
                        indices_passed = compare_values(rm_result[1], torch_result[1], atol=1e-5, rtol=1e-5)
                        forward_passed = forward_passed and indices_passed
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_output = rm_result[0] if return_indices else rm_result
                        torch_output = torch_result[0] if return_indices else torch_result
                        
                        rm_loss = rm_output.sum()
                        torch_loss = torch_output.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        backward_passed = compare_values(rm_input.grad, torch_input.grad, atol=1e-5, rtol=1e-5)
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"max_pool3d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_avg_pool3d(self):
        """测试avg_pool3d函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本avg_pool3d",
                "input_shape": (1, 3, 4, 4, 4),
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "步长大于1",
                "input_shape": (1, 3, 6, 6, 6),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "带填充",
                "input_shape": (1, 3, 4, 4, 4),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "不包含填充区域",
                "input_shape": (1, 3, 5, 5, 5),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "ceil_mode": False,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "自定义除数",
                "input_shape": (1, 3, 4, 4, 4),
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": 3
            },
            {
                "name": "使用ceil_mode",
                "input_shape": (1, 3, 5, 5, 5),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "立方体kernel",
                "input_shape": (1, 2, 8, 8, 8),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "非对称kernel",
                "input_shape": (1, 3, 7, 9, 5),
                "kernel_size": (2, 3, 4),
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "非对称stride",
                "input_shape": (1, 2, 8, 8, 8),
                "kernel_size": 2,
                "stride": (2, 1, 2),
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "非对称padding",
                "input_shape": (1, 3, 6, 6, 6),
                "kernel_size": 3,
                "stride": 1,
                "padding": (1, 1, 0),
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "大kernel",
                "input_shape": (1, 2, 8, 8, 8),
                "kernel_size": (4, 4, 4),
                "stride": 1,
                "padding": 1,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "大stride",
                "input_shape": (1, 2, 10, 10, 10),
                "kernel_size": 2,
                "stride": (3, 3, 3),
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "复杂组合参数",
                "input_shape": (1, 4, 9, 11, 7),
                "kernel_size": (3, 4, 2),
                "stride": (2, 1, 3),
                "padding": (1, 1, 0),
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode+非对称参数",
                "input_shape": (1, 3, 7, 8, 9),
                "kernel_size": (3, 2, 4),
                "stride": (2, 1, 2),
                "padding": (1, 0, 1),
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "深度方向特殊参数",
                "input_shape": (1, 3, 12, 6, 6),
                "kernel_size": (5, 3, 3),
                "stride": (2, 1, 1),
                "padding": (2, 1, 1),
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode+count_include_pad=False",
                "input_shape": (1, 3, 5, 5, 5),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "ceil_mode+自定义除数",
                "input_shape": (1, 3, 5, 5, 5),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": 4
            },
            {
                "name": "非对称参数+ceil_mode+count_include_pad=False",
                "input_shape": (1, 3, 8, 9, 7),
                "kernel_size": (3, 4, 2),
                "stride": (2, 3, 1),
                "padding": (1, 0, 1),
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "极小kernel",
                "input_shape": (1, 3, 5, 5, 5),
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "极大kernel",
                "input_shape": (1, 2, 6, 6, 6),
                "kernel_size": 5,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "dilation=2+ceil_mode+count_include_pad=True",
                "input_shape": (1, 3, 12, 12, 12),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "dilation=2+ceil_mode+count_include_pad=False",
                "input_shape": (1, 3, 12, 12, 12),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "非对称dilation+ceil_mode",
                "input_shape": (1, 3, 10, 11, 9),
                "kernel_size": (3, 2, 4),
                "stride": (2, 3, 1),
                "padding": (1, 1, 0),
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "大padding+小kernel+ceil_mode",
                "input_shape": (1, 3, 5, 5, 5),
                "kernel_size": 2,
                "stride": 1,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "大padding+小kernel+ceil_mode+count_include_pad=False",
                "input_shape": (1, 3, 5, 5, 5),
                "kernel_size": 2,
                "stride": 1,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"avg_pool3d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 解包测试参数
                    input_shape = case["input_shape"]
                    kernel_size = case["kernel_size"]
                    stride = case["stride"]
                    padding = case["padding"]
                    ceil_mode = case["ceil_mode"]
                    count_include_pad = case["count_include_pad"]
                    divisor_override = case.get("divisor_override", None)
                    
                    # 创建测试数据
                    np_input = np.random.randn(*input_shape).astype(np.float32)
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                    
                    # 前向传播测试
                    rm_result = rm_func.avg_pool3d(
                        rm_input, kernel_size=kernel_size, stride=stride, padding=padding,
                        ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                        divisor_override=divisor_override
                    )
                    
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_result = torch_func.avg_pool3d(
                        torch_input, kernel_size=kernel_size, stride=stride, padding=padding,
                        ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                        divisor_override=divisor_override
                    )
                
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result)
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"avg_pool3d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_max_pool1d(self):
        """测试max_pool1d函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本max_pool1d",
                "input_shape": (1, 3, 10),
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "步长大于1",
                "input_shape": (1, 3, 10),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "带填充",
                "input_shape": (1, 3, 10),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": False
            },
            {
                "name": "带膨胀",
                "input_shape": (1, 3, 10),
                "kernel_size": 2,
                "stride": 1,
                "padding": 1,
                "dilation": 2,
                "return_indices": False,
                "ceil_mode": False
            },
            {
                "name": "使用ceil_mode",
                "input_shape": (1, 3, 10),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "ceil_mode边缘窗口1",
                "input_shape": (1, 3, 11),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "ceil_mode边缘窗口2",
                "input_shape": (1, 3, 12),
                "kernel_size": 4,
                "stride": 3,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "ceil_mode带padding",
                "input_shape": (1, 3, 10),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "大kernel_size+ceil_mode",
                "input_shape": (1, 3, 10),
                "kernel_size": 5,
                "stride": 2,
                "padding": 2,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "dilation>1+ceil_mode",
                "input_shape": (1, 3, 12),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 2,
                "return_indices": False,
                "ceil_mode": True
            },
            {
                "name": "大stride+小kernel",
                "input_shape": (1, 3, 9),
                "kernel_size": 2,
                "stride": 3,
                "padding": 0,
                "dilation": 1,
                "return_indices": True,
                "ceil_mode": True
            },
            {
                "name": "小stride+大kernel",
                "input_shape": (1, 3, 10),
                "kernel_size": 5,
                "stride": 2,
                "padding": 2,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": True
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"max_pool1d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 解包测试参数
                    input_shape = case["input_shape"]
                    kernel_size = case["kernel_size"]
                    stride = case["stride"]
                    padding = case["padding"]
                    dilation = case["dilation"]
                    return_indices = case["return_indices"]
                    ceil_mode = case["ceil_mode"]
                    
                    # 创建测试数据
                    np_input = np.random.randn(*input_shape)
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                
                    # 前向传播测试
                    rm_result = None
                    rm_indices = None
                    torch_result = None
                    torch_indices = None
                    
                    # 计算前向传播
                    if return_indices:
                        rm_result, rm_indices = rm_func.max_pool1d(
                            rm_input, kernel_size=kernel_size, stride=stride, padding=padding, 
                            dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode
                        )
                    else:
                        rm_result = rm_func.max_pool1d(
                            rm_input, kernel_size=kernel_size, stride=stride, padding=padding, 
                            dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode
                        )
                    
                    if TORCH_AVAILABLE:
                        if return_indices:
                            torch_result, torch_indices = torch_func.max_pool1d(
                                torch_input, kernel_size=kernel_size, stride=stride, padding=padding, 
                                dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode
                            )
                        else:
                            torch_result = torch_func.max_pool1d(
                                torch_input, kernel_size=kernel_size, stride=stride, padding=padding, 
                                dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode
                            )
                    
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result)
                    
                    # 比较索引
                    indices_passed = True
                    if TORCH_AVAILABLE and return_indices:
                        indices_passed = compare_values(rm_indices, torch_indices)
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and indices_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"max_pool1d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_avg_pool1d(self):
        """测试avg_pool1d函数与PyTorch的一致性"""
        test_cases = [
            {
                "name": "基本avg_pool1d",
                "input_shape": (1, 3, 10),
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "步长大于1",
                "input_shape": (1, 3, 10),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "带填充",
                "input_shape": (1, 3, 10),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "不包含填充区域",
                "input_shape": (1, 3, 10),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "ceil_mode": False,
                "count_include_pad": False,
                "divisor_override": None
            },

            {
                "name": "使用ceil_mode",
                "input_shape": (1, 3, 10),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode边缘窗口1",
                "input_shape": (1, 3, 11),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode边缘窗口2",
                "input_shape": (1, 3, 11),
                "kernel_size": 4,
                "stride": 3,
                "padding": 2,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode+count_include_pad=False",
                "input_shape": (1, 3, 10),
                "kernel_size": 3,
                "stride": 3,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            },
            {
                "name": "大kernel_size+ceil_mode",
                "input_shape": (1, 3, 10),
                "kernel_size": 5,
                "stride": 2,
                "padding": 2,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode边缘窗口3",
                "input_shape": (1, 3, 12),
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None
            },
            {
                "name": "ceil_mode边缘窗口4",
                "input_shape": (1, 3, 10),
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "ceil_mode": True,
                "count_include_pad": False,
                "divisor_override": None
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"avg_pool1d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 解包测试参数
                    input_shape = case["input_shape"]
                    kernel_size = case["kernel_size"]
                    stride = case["stride"]
                    padding = case["padding"]
                    ceil_mode = case["ceil_mode"]
                    count_include_pad = case["count_include_pad"]
                    divisor_override = case["divisor_override"]

                    
                    # 创建测试数据
                    np_input = np.random.randn(*input_shape)
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        
                        torch_input = None
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                
                    # 前向传播测试
                    rm_args = {
                        "input": rm_input, 
                        "kernel_size": kernel_size, 
                        "stride": stride, 
                        "padding": padding,
                        "ceil_mode": ceil_mode, 
                        "count_include_pad": count_include_pad
                    }
                    if divisor_override is not None:
                        rm_args["divisor_override"] = divisor_override
                    rm_result = rm_func.avg_pool1d(**rm_args)
                    
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_result = torch_func.avg_pool1d(
                            torch_input, kernel_size=kernel_size, stride=stride, padding=padding,
                            ceil_mode=ceil_mode, count_include_pad=count_include_pad
                        )
                    
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result)
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"avg_pool1d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def test_tensor_unfold(self):
        """测试tensordef中的unfold方法与PyTorch的一致性"""
        test_cases = [
            {
                "name": "1D张量unfold",
                "input_shape": (2, 3, 8),
                "dimension": 2,
                "size": 4,
                "step": 2
            },
            {
                "name": "2D张量unfold",
                "input_shape": (2, 3, 4, 4),
                "dimension": 2,
                "size": 2,
                "step": 1
            },
            {
                "name": "3D张量unfold",
                "input_shape": (2, 3, 4, 4, 4),
                "dimension": 3,
                "size": 2,
                "step": 1
            },
            {
                "name": "非重叠unfold",
                "input_shape": (2, 3, 8),
                "dimension": 2,
                "size": 2,
                "step": 2
            },
            {
                "name": "起始维度unfold",
                "input_shape": (2, 3, 4),
                "dimension": 0,
                "size": 1,
                "step": 1
            },
            {
                "name": "较大步长unfold",
                "input_shape": (2, 3, 9),
                "dimension": 2,
                "size": 3,
                "step": 3
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"tensor_unfold - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    np_input = np.random.randn(*case["input_shape"])
                    
                    # 根据设备创建张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True)
                        else:
                            torch_input = None
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        
                        if TORCH_AVAILABLE:
                            torch_input = torch.tensor(np_input, requires_grad=True, device=device)
                        else:
                            torch_input = None
                
                    # 前向传播测试
                    rm_result = rm_input.unfold(
                        dimension=case["dimension"],
                        size=case["size"],
                        step=case["step"]
                    )
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_result = torch_input.unfold(
                            dimension=case["dimension"],
                            size=case["size"],
                            step=case["step"]
                        )
                    
                    # 比较前向传播结果
                    forward_passed = compare_values(rm_result, torch_result)
                    
                    # 反向传播测试
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        # 计算损失
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        torch_loss.backward()
                        
                        # 比较梯度
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed and TORCH_AVAILABLE:
                            print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                            print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                            print(f"  Riemann形状: {rm_result.shape}, PyTorch形状: {torch_result.shape if torch_result is not None else 'N/A'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"tensor_unfold测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def test_adaptive_avg_pool1d(self):
        """测试adaptive_avg_pool1d函数与PyTorch的一致性"""
        test_cases = [
            {"name": "基本池化", "input_shape": (2, 3, 10), "output_size": 5},
            {"name": "全局池化", "input_shape": (2, 3, 10), "output_size": 1},
            {"name": "相同尺寸", "input_shape": (2, 3, 10), "output_size": 10},
            {"name": "大尺寸输入", "input_shape": (4, 8, 100), "output_size": 7},
            {"name": "小尺寸输出", "input_shape": (2, 3, 8), "output_size": 3},
            {"name": "单通道", "input_shape": (1, 1, 16), "output_size": 4},
            {"name": "多批次", "input_shape": (8, 16, 32), "output_size": 8},
        ]
        
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"adaptive_avg_pool1d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    input_shape = case["input_shape"]
                    output_size = case["output_size"]
                    
                    np_input = np.random.randn(*input_shape)
                    
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        torch_input = torch.tensor(np_input, requires_grad=True) if TORCH_AVAILABLE else None
                    else:
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        torch_input = torch.tensor(np_input, requires_grad=True, device=device) if TORCH_AVAILABLE else None
                    
                    rm_result = rm_func.adaptive_avg_pool1d(rm_input, output_size)
                    torch_result = torch_func.adaptive_avg_pool1d(torch_input, output_size) if TORCH_AVAILABLE else None
                    
                    forward_passed = compare_values(rm_result, torch_result)
                    shape_passed = rm_result.shape == (input_shape[0], input_shape[1], output_size)
                    
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        rm_loss.backward()
                        torch_loss.backward()
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and shape_passed and backward_passed
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"adaptive_avg_pool1d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def test_adaptive_avg_pool2d(self):
        """测试adaptive_avg_pool2d函数与PyTorch的一致性"""
        test_cases = [
            {"name": "基本池化", "input_shape": (2, 3, 8, 8), "output_size": (4, 4)},
            {"name": "全局池化", "input_shape": (2, 3, 8, 8), "output_size": 1},
            {"name": "正方形输出", "input_shape": (2, 3, 8, 8), "output_size": 7},
            {"name": "矩形输出", "input_shape": (2, 3, 16, 32), "output_size": (4, 8)},
            {"name": "相同尺寸", "input_shape": (2, 3, 8, 8), "output_size": (8, 8)},
            {"name": "大尺寸输入", "input_shape": (4, 8, 64, 64), "output_size": (7, 7)},
            {"name": "小尺寸输出", "input_shape": (2, 3, 4, 4), "output_size": (2, 2)},
        ]
        
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"adaptive_avg_pool2d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    input_shape = case["input_shape"]
                    output_size = case["output_size"]
                    
                    np_input = np.random.randn(*input_shape)
                    
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        torch_input = torch.tensor(np_input, requires_grad=True) if TORCH_AVAILABLE else None
                    else:
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        torch_input = torch.tensor(np_input, requires_grad=True, device=device) if TORCH_AVAILABLE else None
                    
                    rm_result = rm_func.adaptive_avg_pool2d(rm_input, output_size)
                    torch_result = torch_func.adaptive_avg_pool2d(torch_input, output_size) if TORCH_AVAILABLE else None
                    
                    forward_passed = compare_values(rm_result, torch_result)
                    
                    if isinstance(output_size, int):
                        expected_shape = (input_shape[0], input_shape[1], output_size, output_size)
                    else:
                        expected_shape = (input_shape[0], input_shape[1], output_size[0], output_size[1])
                    shape_passed = rm_result.shape == expected_shape
                    
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        rm_loss.backward()
                        torch_loss.backward()
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and shape_passed and backward_passed
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"adaptive_avg_pool2d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def test_adaptive_avg_pool3d(self):
        """测试adaptive_avg_pool3d函数与PyTorch的一致性"""
        test_cases = [
            {"name": "基本池化", "input_shape": (2, 3, 8, 8, 8), "output_size": (4, 4, 4)},
            {"name": "全局池化", "input_shape": (2, 3, 8, 8, 8), "output_size": 1},
            {"name": "立方体输出", "input_shape": (2, 3, 8, 8, 8), "output_size": 5},
            {"name": "长方体输出", "input_shape": (2, 3, 16, 32, 8), "output_size": (4, 8, 2)},
            {"name": "相同尺寸", "input_shape": (2, 3, 4, 4, 4), "output_size": (4, 4, 4)},
            {"name": "小尺寸输出", "input_shape": (2, 3, 4, 4, 4), "output_size": (2, 2, 2)},
        ]
        
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"adaptive_avg_pool3d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    input_shape = case["input_shape"]
                    output_size = case["output_size"]
                    
                    np_input = np.random.randn(*input_shape)
                    
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        torch_input = torch.tensor(np_input, requires_grad=True) if TORCH_AVAILABLE else None
                    else:
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        torch_input = torch.tensor(np_input, requires_grad=True, device=device) if TORCH_AVAILABLE else None
                    
                    rm_result = rm_func.adaptive_avg_pool3d(rm_input, output_size)
                    torch_result = torch_func.adaptive_avg_pool3d(torch_input, output_size) if TORCH_AVAILABLE else None
                    
                    forward_passed = compare_values(rm_result, torch_result)
                    
                    if isinstance(output_size, int):
                        expected_shape = (input_shape[0], input_shape[1], output_size, output_size, output_size)
                    else:
                        expected_shape = (input_shape[0], input_shape[1], output_size[0], output_size[1], output_size[2])
                    shape_passed = rm_result.shape == expected_shape
                    
                    backward_passed = True
                    if TORCH_AVAILABLE:
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        rm_loss.backward()
                        torch_loss.backward()
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and shape_passed and backward_passed
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"adaptive_avg_pool3d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def test_adaptive_max_pool1d(self):
        """测试adaptive_max_pool1d函数与PyTorch的一致性"""
        test_cases = [
            {"name": "基本池化", "input_shape": (2, 3, 10), "output_size": 5, "return_indices": False},
            {"name": "全局池化", "input_shape": (2, 3, 10), "output_size": 1, "return_indices": False},
            {"name": "返回索引", "input_shape": (2, 3, 10), "output_size": 5, "return_indices": True},
            {"name": "大尺寸输入", "input_shape": (4, 8, 100), "output_size": 7, "return_indices": True},
            {"name": "相同尺寸", "input_shape": (2, 3, 10), "output_size": 10, "return_indices": False},
        ]
        
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"adaptive_max_pool1d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    input_shape = case["input_shape"]
                    output_size = case["output_size"]
                    return_indices = case["return_indices"]
                    
                    np_input = np.random.randn(*input_shape)
                    
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        torch_input = torch.tensor(np_input, requires_grad=True) if TORCH_AVAILABLE else None
                    else:
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        torch_input = torch.tensor(np_input, requires_grad=True, device=device) if TORCH_AVAILABLE else None
                    
                    if return_indices:
                        rm_result, rm_indices = rm_func.adaptive_max_pool1d(rm_input, output_size, return_indices=True)
                        torch_result, torch_indices = torch_func.adaptive_max_pool1d(torch_input, output_size, return_indices=True) if TORCH_AVAILABLE else (None, None)
                    else:
                        rm_result = rm_func.adaptive_max_pool1d(rm_input, output_size, return_indices=False)
                        torch_result = torch_func.adaptive_max_pool1d(torch_input, output_size, return_indices=False) if TORCH_AVAILABLE else None
                    
                    forward_passed = compare_values(rm_result, torch_result)
                    shape_passed = rm_result.shape == (input_shape[0], input_shape[1], output_size)
                    
                    indices_passed = True
                    if TORCH_AVAILABLE and return_indices:
                        indices_passed = compare_values(rm_indices, torch_indices)
                    
                    backward_passed = True
                    if TORCH_AVAILABLE and not return_indices:
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        rm_loss.backward()
                        torch_loss.backward()
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and shape_passed and indices_passed and backward_passed
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"adaptive_max_pool1d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def test_adaptive_max_pool2d(self):
        """测试adaptive_max_pool2d函数与PyTorch的一致性"""
        test_cases = [
            {"name": "基本池化", "input_shape": (2, 3, 8, 8), "output_size": (4, 4), "return_indices": False},
            {"name": "全局池化", "input_shape": (2, 3, 8, 8), "output_size": 1, "return_indices": False},
            {"name": "返回索引", "input_shape": (2, 3, 8, 8), "output_size": (4, 4), "return_indices": True},
            {"name": "正方形输出", "input_shape": (2, 3, 16, 16), "output_size": 7, "return_indices": True},
            {"name": "矩形输出", "input_shape": (2, 3, 16, 32), "output_size": (4, 8), "return_indices": False},
        ]
        
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"adaptive_max_pool2d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    input_shape = case["input_shape"]
                    output_size = case["output_size"]
                    return_indices = case["return_indices"]
                    
                    np_input = np.random.randn(*input_shape)
                    
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        torch_input = torch.tensor(np_input, requires_grad=True) if TORCH_AVAILABLE else None
                    else:
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        torch_input = torch.tensor(np_input, requires_grad=True, device=device) if TORCH_AVAILABLE else None
                    
                    if return_indices:
                        rm_result, rm_indices = rm_func.adaptive_max_pool2d(rm_input, output_size, return_indices=True)
                        torch_result, torch_indices = torch_func.adaptive_max_pool2d(torch_input, output_size, return_indices=True) if TORCH_AVAILABLE else (None, None)
                    else:
                        rm_result = rm_func.adaptive_max_pool2d(rm_input, output_size, return_indices=False)
                        torch_result = torch_func.adaptive_max_pool2d(torch_input, output_size, return_indices=False) if TORCH_AVAILABLE else None
                    
                    forward_passed = compare_values(rm_result, torch_result)
                    
                    if isinstance(output_size, int):
                        expected_shape = (input_shape[0], input_shape[1], output_size, output_size)
                    else:
                        expected_shape = (input_shape[0], input_shape[1], output_size[0], output_size[1])
                    shape_passed = rm_result.shape == expected_shape
                    
                    indices_passed = True
                    if TORCH_AVAILABLE and return_indices:
                        indices_passed = compare_values(rm_indices, torch_indices)
                    
                    backward_passed = True
                    if TORCH_AVAILABLE and not return_indices:
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        rm_loss.backward()
                        torch_loss.backward()
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and shape_passed and indices_passed and backward_passed
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"adaptive_max_pool2d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def test_adaptive_max_pool2d_indices_boundary(self):
        """测试adaptive_max_pool2d返回索引的边界情况
        
        验证在各种边界情况下（单行、单列、单元素区域），返回的索引是否正确
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用，跳过索引边界测试")
        
        test_cases = [
            # 测试单行区域（高度方向只有一个元素）
            {"name": "单行区域", "input_shape": (1, 1, 3, 8), "output_size": (3, 4)},
            # 测试单列区域（宽度方向只有一个元素）
            {"name": "单列区域", "input_shape": (1, 1, 8, 3), "output_size": (4, 3)},
            # 测试单元素区域（高度和宽度都只有一个元素）
            {"name": "单元素区域", "input_shape": (1, 1, 4, 4), "output_size": (4, 4)},
            # 测试非均匀分割
            {"name": "非均匀分割", "input_shape": (1, 1, 7, 7), "output_size": (3, 3)},
            # 测试大尺寸输入小尺寸输出
            {"name": "大输入小输出", "input_shape": (1, 1, 100, 100), "output_size": (3, 3)},
        ]
        
        for case in test_cases:
            case_name = f"adaptive_max_pool2d_indices_boundary - {case['name']}"
            start_time = time.time()
            try:
                input_shape = case["input_shape"]
                output_size = case["output_size"]
                H_in, W_in = input_shape[2], input_shape[3]
                
                # 创建确定性输入，使用递增序列便于验证索引
                np_input = np.arange(input_shape[2] * input_shape[3]).reshape(input_shape).astype(np.float32)
                
                rm_input = rm.tensor(np_input)
                torch_input = torch.tensor(np_input)
                
                rm_result, rm_indices = rm_func.adaptive_max_pool2d(rm_input, output_size, return_indices=True)
                torch_result, torch_indices = torch_func.adaptive_max_pool2d(torch_input, output_size, return_indices=True)
                
                # 验证结果值一致
                forward_passed = compare_values(rm_result, torch_result)
                
                # 验证索引一致
                indices_passed = compare_values(rm_indices, torch_indices)
                
                # 额外验证：通过索引能否正确获取最大值
                indices_valid = True
                if indices_passed:
                    rm_indices_flat = rm_indices.data.flatten()
                    rm_result_flat = rm_result.data.flatten()
                    for idx in range(len(rm_indices_flat)):
                        flat_idx = int(rm_indices_flat[idx])
                        h_idx = flat_idx // W_in
                        w_idx = flat_idx % W_in
                        expected_val = np_input[0, 0, h_idx, w_idx]
                        actual_val = rm_result_flat[idx]
                        if not np.isclose(expected_val, actual_val, rtol=1e-5, atol=1e-6):
                            indices_valid = False
                            break
                
                passed = forward_passed and indices_passed and indices_valid
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"adaptive_max_pool2d索引边界测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_adaptive_max_pool3d(self):
        """测试adaptive_max_pool3d函数与PyTorch的一致性"""
        test_cases = [
            {"name": "基本池化", "input_shape": (2, 3, 8, 8, 8), "output_size": (4, 4, 4), "return_indices": False},
            {"name": "全局池化", "input_shape": (2, 3, 8, 8, 8), "output_size": 1, "return_indices": False},
            {"name": "返回索引", "input_shape": (2, 3, 8, 8, 8), "output_size": (4, 4, 4), "return_indices": True},
            {"name": "立方体输出", "input_shape": (2, 3, 16, 16, 16), "output_size": 5, "return_indices": True},
            {"name": "长方体输出", "input_shape": (2, 3, 16, 32, 8), "output_size": (4, 8, 2), "return_indices": False},
        ]
        
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"adaptive_max_pool3d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    input_shape = case["input_shape"]
                    output_size = case["output_size"]
                    return_indices = case["return_indices"]
                    
                    np_input = np.random.randn(*input_shape)
                    
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                        torch_input = torch.tensor(np_input, requires_grad=True) if TORCH_AVAILABLE else None
                    else:
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                        torch_input = torch.tensor(np_input, requires_grad=True, device=device) if TORCH_AVAILABLE else None
                    
                    if return_indices:
                        rm_result, rm_indices = rm_func.adaptive_max_pool3d(rm_input, output_size, return_indices=True)
                        torch_result, torch_indices = torch_func.adaptive_max_pool3d(torch_input, output_size, return_indices=True) if TORCH_AVAILABLE else (None, None)
                    else:
                        rm_result = rm_func.adaptive_max_pool3d(rm_input, output_size, return_indices=False)
                        torch_result = torch_func.adaptive_max_pool3d(torch_input, output_size, return_indices=False) if TORCH_AVAILABLE else None
                    
                    forward_passed = compare_values(rm_result, torch_result)
                    
                    if isinstance(output_size, int):
                        expected_shape = (input_shape[0], input_shape[1], output_size, output_size, output_size)
                    else:
                        expected_shape = (input_shape[0], input_shape[1], output_size[0], output_size[1], output_size[2])
                    shape_passed = rm_result.shape == expected_shape
                    
                    indices_passed = True
                    if TORCH_AVAILABLE and return_indices:
                        indices_passed = compare_values(rm_indices, torch_indices)
                    
                    backward_passed = True
                    if TORCH_AVAILABLE and not return_indices:
                        rm_loss = rm_result.sum()
                        torch_loss = torch_result.sum()
                        rm_loss.backward()
                        torch_loss.backward()
                        backward_passed = compare_values(rm_input.grad, torch_input.grad)
                    
                    passed = forward_passed and shape_passed and indices_passed and backward_passed
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                    self.assertTrue(passed, f"adaptive_max_pool3d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

    def test_adaptive_max_pool3d_indices_boundary(self):
        """测试adaptive_max_pool3d返回索引的边界情况
        
        验证在各种边界情况下（单层、单行、单列、单元素区域），返回的索引是否正确
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用，跳过索引边界测试")
        
        test_cases = [
            # 测试单层区域（深度方向只有一个元素）
            {"name": "单层区域", "input_shape": (1, 1, 3, 8, 8), "output_size": (3, 4, 4)},
            # 测试单行区域（高度方向只有一个元素）
            {"name": "单行区域", "input_shape": (1, 1, 8, 3, 8), "output_size": (4, 3, 4)},
            # 测试单列区域（宽度方向只有一个元素）
            {"name": "单列区域", "input_shape": (1, 1, 8, 8, 3), "output_size": (4, 4, 3)},
            # 测试单元素区域（深度、高度、宽度都只有一个元素）
            {"name": "单元素区域", "input_shape": (1, 1, 4, 4, 4), "output_size": (4, 4, 4)},
            # 测试非均匀分割
            {"name": "非均匀分割", "input_shape": (1, 1, 7, 7, 7), "output_size": (3, 3, 3)},
            # 测试大尺寸输入小尺寸输出
            {"name": "大输入小输出", "input_shape": (1, 1, 50, 50, 50), "output_size": (3, 3, 3)},
        ]
        
        for case in test_cases:
            case_name = f"adaptive_max_pool3d_indices_boundary - {case['name']}"
            start_time = time.time()
            try:
                input_shape = case["input_shape"]
                output_size = case["output_size"]
                D_in, H_in, W_in = input_shape[2], input_shape[3], input_shape[4]
                
                # 创建确定性输入，使用递增序列便于验证索引
                np_input = np.arange(D_in * H_in * W_in).reshape(input_shape).astype(np.float32)
                
                rm_input = rm.tensor(np_input)
                torch_input = torch.tensor(np_input)
                
                rm_result, rm_indices = rm_func.adaptive_max_pool3d(rm_input, output_size, return_indices=True)
                torch_result, torch_indices = torch_func.adaptive_max_pool3d(torch_input, output_size, return_indices=True)
                
                # 验证结果值一致
                forward_passed = compare_values(rm_result, torch_result)
                
                # 验证索引一致
                indices_passed = compare_values(rm_indices, torch_indices)
                
                # 额外验证：通过索引能否正确获取最大值
                indices_valid = True
                if indices_passed:
                    rm_indices_flat = rm_indices.data.flatten()
                    rm_result_flat = rm_result.data.flatten()
                    for idx in range(len(rm_indices_flat)):
                        flat_idx = int(rm_indices_flat[idx])
                        d_idx = flat_idx // (H_in * W_in)
                        rem = flat_idx % (H_in * W_in)
                        h_idx = rem // W_in
                        w_idx = rem % W_in
                        expected_val = np_input[0, 0, d_idx, h_idx, w_idx]
                        actual_val = rm_result_flat[idx]
                        if not np.isclose(expected_val, actual_val, rtol=1e-5, atol=1e-6):
                            indices_valid = False
                            break
                
                passed = forward_passed and indices_passed and indices_valid
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                self.assertTrue(passed, f"adaptive_max_pool3d索引边界测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_tensor_fold(self):
        """测试tensordef中的fold方法"""
        # 由于PyTorch没有对应的Tensor.fold方法，我们需要自己验证结果的正确性
        
        def calculate_expected_fold(input_tensor, dimension, size, step, output_size=None):   
            """计算fold的预期结果"""
            input_shape = input_tensor.shape
            num_windows = input_shape[dimension]
            window_size = input_shape[-1]

            if output_size is None:
                output_size = (num_windows - 1) * step + size

            output_shape = list(input_shape[:-1])  # 移除最后一维（窗口元素）
            output_shape[dimension] = output_size

            # 检测输入张量的类型，使用对应的库创建输出张量
            # 检查是否是CuPy数组
            if hasattr(input_tensor, '__module__') and input_tensor.__module__.startswith('cupy'):
                # 对于CuPy数组（CUDA），使用CuPy
                output = cp.zeros(output_shape)
            else:
                # 对于NumPy数组（CPU），使用NumPy
                output = np.zeros(output_shape)

            # 对于每个窗口，将其内容添加到输出张量的对应位置
            for i in range(num_windows):
                start = i * step
                end = start + size

                # 获取当前窗口内容
                window_idx = [slice(None)] * input_tensor.ndim
                window_idx[dimension] = i
                window_idx[-1] = slice(None)
                window = input_tensor[tuple(window_idx)]

                # 确保窗口形状与输出切片形状匹配
                window_shape = list(output_shape)
                window_shape[dimension] = size
                window = window.reshape(tuple(window_shape))

                # 将窗口内容添加到输出张量的对应位置
                output_idx = [slice(None)] * output.ndim
                output_idx[dimension] = slice(start, end)
                output[tuple(output_idx)] += window

            return output
        
        test_cases = [
            {
                "name": "1D张量fold",
                "input_shape": (2, 3, 8),
                "dimension": 2,
                "size": 4,
                "step": 2
            },
            {
                "name": "2D张量fold",
                "input_shape": (2, 3, 4, 4),
                "dimension": 2,
                "size": 2,
                "step": 1
            },
            {
                "name": "非重叠fold",
                "input_shape": (2, 3, 8),
                "dimension": 2,
                "size": 2,
                "step": 2
            },
            {
                "name": "较大步长fold",
                "input_shape": (2, 3, 9),
                "dimension": 2,
                "size": 3,
                "step": 3
            },
            {
                "name": "指定输出大小fold",
                "input_shape": (2, 3, 8),
                "dimension": 2,
                "size": 4,
                "step": 2,
                "output_size": 8
            }
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"tensor_fold - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    np_input = np.random.randn(*case["input_shape"])
                    
                    # 首先对输入进行unfold操作，得到unfolded张量
                    if device == "cpu":
                        rm_input = rm.tensor(np_input, requires_grad=True)
                    else:  # cuda
                        rm_input = rm.tensor(np_input, requires_grad=True, device=device)
                    
                    rm_unfolded = rm_input.unfold(
                        dimension=case["dimension"],
                        size=case["size"],
                        step=case["step"]
                    )
                
                    # 计算预期的fold结果
                    np_unfolded = rm_unfolded.data  # 转换为numpy数组
                    output_size = case.get("output_size", None)
                    expected_output = calculate_expected_fold(np_unfolded, case["dimension"], case["size"], case["step"], output_size)
                    
                    # 执行fold操作
                    rm_folded = rm_unfolded.fold(
                        dimension=case["dimension"],
                        size=case["size"],
                        step=case["step"],
                        output_size=output_size
                    )
                    
                    # 比较前向传播结果
                    rm_output = rm_folded.data
                    forward_passed = np.allclose(rm_output, expected_output, atol=1e-6, rtol=1e-6)
                    
                    # 反向传播测试
                    backward_passed = True
                    try:
                        # 计算损失
                        rm_loss = rm_folded.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        
                        # 检查梯度是否存在且形状正确
                        if rm_input.grad is None:
                            backward_passed = False
                        elif rm_input.grad.shape != rm_input.shape:
                            backward_passed = False
                    except Exception as e:
                        backward_passed = False
                    
                    passed = forward_passed and backward_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                            print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                            print(f"  Riemann输出形状: {rm_folded.shape}, 预期输出形状: {expected_output.shape}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"tensor_fold测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行卷积相关函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestConvFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)