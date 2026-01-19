import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.autograd.functional import vhp, hvp
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
    print("警告: 无法导入PyTorch，将只测试riemann的VHP/HVP函数")
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
        for i, (r, t) in enumerate(zip(rm_result, torch_result)):
            if not compare_values(r, t, atol, rtol):
                all_passed = False
                break
        
        return all_passed
    
    # 处理Riemann结果
    if hasattr(rm_result, 'is_cuda') and rm_result.is_cuda:
        rm_data = rm_result.detach().cpu().numpy()
    else:
        rm_data = rm_result.detach().numpy()
    # 处理PyTorch结果
    if hasattr(torch_result, 'is_cuda') and torch_result.is_cuda:
        torch_data = torch_result.detach().cpu().numpy()
    else:
        torch_data = torch_result.detach().numpy()
    
    # 处理形状不匹配的情况
    try:
        # 增加容差参数
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

class TestVhpHvpFunctions(unittest.TestCase):
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
    
    def test_single_input_single_output(self):
        """测试场景1: 单输入单输出函数"""
        test_cases = [
            {"name": "单输入单输出函数 VHP/HVP"}
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 定义测试函数 - 返回标量
                    def f(x):
                        return (x ** 2).sum()
                    
                    # 定义对应的PyTorch函数
                    def pt_f(x):
                        return (x ** 2).sum()
                    
                    # 创建测试数据
                    np_x = np.random.randn(3, 4)
                    rm_x = rm.tensor(np_x, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        torch_x = torch.tensor(np_x, requires_grad=True, device=device)
                    else:
                        torch_x = None
                    
                    np_v = np.random.randn(3, 4)
                    rm_v = rm.tensor(np_v, device=device)
                    if TORCH_AVAILABLE:
                        torch_v = torch.tensor(np_v, device=device)
                    else:
                        torch_v = None
                    
                    # 计算Riemann的HVP
                    _,rm_hvp = hvp(f, rm_x, rm_v)
                    
                    # 计算PyTorch的HVP
                    if TORCH_AVAILABLE:
                        torch_output, torch_hvp_result = torch.autograd.functional.hvp(pt_f, torch_x, torch_v)
                    
                    # 计算Riemann的VHP
                    _,rm_vhp = vhp(f, rm_x, rm_v)
                    
                    # 计算PyTorch的VHP
                    if TORCH_AVAILABLE:
                        torch_output, torch_vhp_result = torch.autograd.functional.vhp(pt_f, torch_x, torch_v)
                    
                    # 比较结果
                    passed = True
                    if TORCH_AVAILABLE:
                        # 比较HVP结果
                        hvp_passed = compare_values(rm_hvp, torch_hvp_result)
                        # 比较VHP结果
                        vhp_passed = compare_values(rm_vhp, torch_vhp_result)
                        # 验证HVP和VHP的结果是否相等（由于Hessian对称性）
                        hvp_vhp_passed = compare_values(rm_hvp, rm_vhp)
                        passed = hvp_passed and vhp_passed and hvp_vhp_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed and TORCH_AVAILABLE:
                            print(f"  HVP比较: {'通过' if hvp_passed else '失败'}")
                            print(f"  VHP比较: {'通过' if vhp_passed else '失败'}")
                            print(f"  HVP与VHP对称性: {'通过' if hvp_vhp_passed else '失败'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"VHP/HVP计算结果不匹配: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_multiple_inputs_single_output(self):
        """测试场景2: 多输入单输出函数"""
        test_cases = [
            {"name": "多输入单输出函数 VHP/HVP"}
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 定义测试函数 - 返回标量
                    def f(x, y):
                        return (x @ y).sum()
                    
                    # 定义对应的PyTorch函数
                    def pt_f(x, y):
                        return (x @ y).sum()
                    
                    # 创建测试数据
                    np_x = np.random.randn(2, 3)
                    np_y = np.random.randn(3, 4)
                    rm_x = rm.tensor(np_x, requires_grad=True, device=device)
                    rm_y = rm.tensor(np_y, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        torch_x = torch.tensor(np_x, requires_grad=True, device=device)
                        torch_y = torch.tensor(np_y, requires_grad=True, device=device)
                    else:
                        torch_x, torch_y = None, None
                    
                    np_vx = np.random.randn(2, 3)
                    np_vy = np.random.randn(3, 4)
                    rm_vx = rm.tensor(np_vx, device=device)
                    rm_vy = rm.tensor(np_vy, device=device)
                    if TORCH_AVAILABLE:
                        torch_vx = torch.tensor(np_vx, device=device)
                        torch_vy = torch.tensor(np_vy, device=device)
                    else:
                        torch_vx, torch_vy = None, None
                    
                    # 不需要初始化这些变量，因为它们会在条件语句中被赋值
                    
                    # 计算Riemann的HVP
                    _,rm_hvp = hvp(f, (rm_x, rm_y), (rm_vx, rm_vy))
                    
                    # 计算PyTorch的HVP
                    if TORCH_AVAILABLE:
                        torch_output, torch_hvp_result = torch.autograd.functional.hvp(pt_f, (torch_x, torch_y), (torch_vx, torch_vy))
                    
                    # 计算Riemann的VHP
                    _,rm_vhp = vhp(f, (rm_x, rm_y), (rm_vx, rm_vy))
                    
                    # 计算PyTorch的VHP
                    if TORCH_AVAILABLE:
                        torch_output, torch_vhp_result = torch.autograd.functional.vhp(pt_f, (torch_x, torch_y), (torch_vx, torch_vy))
                    
                    # 比较结果
                    passed = True
                    if TORCH_AVAILABLE:
                        # 比较HVP结果
                        hvp_passed = compare_values(rm_hvp, torch_hvp_result)
                        # 比较VHP结果
                        vhp_passed = compare_values(rm_vhp, torch_vhp_result)
                        passed = hvp_passed and vhp_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed and TORCH_AVAILABLE:
                            print(f"  HVP比较: {'通过' if hvp_passed else '失败'}")
                            print(f"  VHP比较: {'通过' if vhp_passed else '失败'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"VHP/HVP计算结果不匹配: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_complex_function(self):
        """测试场景3: 复杂函数"""
        test_cases = [
            {"name": "复杂函数 VHP/HVP"}
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 定义复杂测试函数 - 返回标量
                    def f(x):
                        y = rm.sin(x)
                        z = y ** 2
                        w = z.sum(dim=1, keepdim=True)
                        return (w * x).sum()
                    
                    # 定义对应的PyTorch函数
                    def pt_f(x):
                        y = torch.sin(x)
                        z = y ** 2
                        w = z.sum(dim=1, keepdim=True)
                        return (w * x).sum()
                    
                    # 创建测试数据
                    np_x = np.random.randn(5, 3)
                    rm_x = rm.tensor(np_x, requires_grad=True, device=device)
                    if TORCH_AVAILABLE:
                        torch_x = torch.tensor(np_x, requires_grad=True, device=device)
                    else:
                        torch_x = None
                    
                    np_v = np.random.randn(5, 3)
                    rm_v = rm.tensor(np_v, device=device)
                    if TORCH_AVAILABLE:
                        torch_v = torch.tensor(np_v, device=device)
                    else:
                        torch_v = None
                    
                    # 不需要初始化这些变量，因为它们会在条件语句中被赋值
                    
                    # 计算Riemann的HVP
                    _,rm_hvp = hvp(f, rm_x, rm_v)
                    
                    # 计算PyTorch的HVP
                    if TORCH_AVAILABLE:
                        torch_output, torch_hvp_result = torch.autograd.functional.hvp(pt_f, torch_x, torch_v)
                    
                    # 计算Riemann的VHP
                    _,rm_vhp = vhp(f, rm_x, rm_v)
                    
                    # 计算PyTorch的VHP
                    if TORCH_AVAILABLE:
                        torch_output, torch_vhp_result = torch.autograd.functional.vhp(pt_f, torch_x, torch_v)
                    
                    # 比较结果
                    passed = True
                    if TORCH_AVAILABLE:
                        # 比较HVP结果
                        hvp_passed = compare_values(rm_hvp, torch_hvp_result)
                        # 比较VHP结果
                        vhp_passed = compare_values(rm_vhp, torch_vhp_result)
                        passed = hvp_passed and vhp_passed
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed and TORCH_AVAILABLE:
                            print(f"  HVP比较: {'通过' if hvp_passed else '失败'}")
                            print(f"  VHP比较: {'通过' if vhp_passed else '失败'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"VHP/HVP计算结果不匹配: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_create_graph(self):
        """测试场景4: create_graph参数"""
        test_cases = [
            {"name": "create_graph参数测试"}
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"{case['name']} - {device}"
                start_time = time.time()
                try:
                    # 定义测试函数 - 返回标量
                    def f(x):
                        return (x ** 2).sum()
                    
                    # 创建测试数据
                    np_x = np.random.randn(2, 2)
                    rm_x = rm.tensor(np_x, requires_grad=True, device=device)
                    
                    np_v = np.random.randn(2, 2)
                    rm_v = rm.tensor(np_v, device=device)
                    
                    # 测试create_graph=True的HVP
                    _,rm_hvp = hvp(f, rm_x, rm_v, create_graph=True)
                    
                    # 测试create_graph=True的VHP
                    _,rm_vhp = vhp(f, rm_x, rm_v, create_graph=True)
                    
                    # 检查结果是否可求导
                    hvp_requires_grad = hasattr(rm_hvp, 'requires_grad') and rm_hvp.requires_grad
                    vhp_requires_grad = hasattr(rm_vhp, 'requires_grad') and rm_vhp.requires_grad
                    
                    passed = hvp_requires_grad and vhp_requires_grad
                    
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        print(f"  HVP output requires_grad: {hvp_requires_grad}")
                        print(f"  VHP output requires_grad: {vhp_requires_grad}")
                    
                    # 断言确保测试通过
                    self.assertTrue(passed, f"create_graph参数测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_strict_parameter(self):
        """测试场景5: strict参数"""
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            start_time = time.time()
            try:
                # 定义一个函数，其中第二个输入与输出无关
                def f_independent(x, y):
                    return (x ** 2).sum()
                
                # 创建测试数据
                np_x = np.random.randn(2, 2)
                np_y = np.random.randn(2, 2)
                np_vx = np.random.randn(2, 2)
                np_vy = np.random.randn(2, 2)
                rm_x = rm.tensor(np_x, requires_grad=True, device=device)
                rm_y = rm.tensor(np_y, requires_grad=True, device=device)
                rm_vx = rm.tensor(np_vx, device=device)
                rm_vy = rm.tensor(np_vy, device=device)
                
                # 测试HVP的strict=False
                hvp_strict_false_passed = False
                try:
                    # 应该不会抛出异常
                    _,rm_hvp_indep = hvp(f_independent, (rm_x, rm_y), (rm_vx, rm_vy), strict=False)
                    hvp_strict_false_passed = True
                except Exception as e:
                    pass
                
                # 测试HVP的strict=True
                hvp_strict_true_passed = False
                try:
                    # 应该会抛出异常，因为y是独立变量
                    _,rm_hvp_indep = hvp(f_independent, (rm_x, rm_y), (rm_vx, rm_vy), strict=True)
                except Exception:
                    hvp_strict_true_passed = True
                
                # 测试VHP的strict=False
                vhp_strict_false_passed = False
                try:
                    # 应该不会抛出异常
                    _,rm_vhp_indep = vhp(f_independent, (rm_x, rm_y), (rm_vx, rm_vy), strict=False)
                    vhp_strict_false_passed = True
                except Exception as e:
                    pass
                
                # 测试VHP的strict=True
                vhp_strict_true_passed = False
                try:
                    # 应该会抛出异常，因为y是独立变量
                    _,rm_vhp_indep = vhp(f_independent, (rm_x, rm_y), (rm_vx, rm_vy), strict=True)
                except Exception:
                    vhp_strict_true_passed = True
                
                # 所有测试都应该通过
                passed = hvp_strict_false_passed and hvp_strict_true_passed and vhp_strict_false_passed and vhp_strict_true_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(f"HVP strict=False - {device}", hvp_strict_false_passed)
                    stats.add_result(f"HVP strict=True - {device}", hvp_strict_true_passed)
                    stats.add_result(f"VHP strict=False - {device}", vhp_strict_false_passed)
                    stats.add_result(f"VHP strict=True - {device}", vhp_strict_true_passed)
                    
                    print(f"测试用例: HVP strict=False - {device} - {Colors.OKGREEN if hvp_strict_false_passed else Colors.FAIL}{'通过' if hvp_strict_false_passed else '失败'}{Colors.ENDC}")
                    print(f"测试用例: HVP strict=True - {device} - {Colors.OKGREEN if hvp_strict_true_passed else Colors.FAIL}{'通过' if hvp_strict_true_passed else '失败'}{Colors.ENDC}")
                    print(f"测试用例: VHP strict=False - {device} - {Colors.OKGREEN if vhp_strict_false_passed else Colors.FAIL}{'通过' if vhp_strict_false_passed else '失败'}{Colors.ENDC}")
                    print(f"测试用例: VHP strict=True - {device} - {Colors.OKGREEN if vhp_strict_true_passed else Colors.FAIL}{'通过' if vhp_strict_true_passed else '失败'}{Colors.ENDC}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"strict参数测试失败 - {device}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(f"strict参数测试 - {device}", False, [str(e)])
                    print(f"测试用例: strict参数测试 - {device} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行VHP和HVP函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestVhpHvpFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)