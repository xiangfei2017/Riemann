import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入riemann模块
try:
    import riemann as rm
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
    
    # 执行简单的PyTorch操作以触发初始化
    warmup_input = torch.tensor([[0.0]], requires_grad=True)
    warmup_output = warmup_input.sum()
    warmup_output.backward()
    
    # 清理资源
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"PyTorch预热完成，耗时: {time.time() - warmup_start:.4f}秒")
    
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的embedding函数")
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
    
    # 转换为numpy数组
    try:
        rm_data = rm_result.data if hasattr(rm_result, 'data') else rm_result.numpy()
        torch_data = torch_result.detach().numpy()
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

# 测试embedding类
class TestEmbedding(unittest.TestCase):
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
    
    def test_basic_embedding(self):
        """测试基本embedding功能"""
        test_cases = [
            {"name": "基本嵌入功能", "input_shape": (4,), "weight_shape": (10, 3)},
            {"name": "多维输入形状", "input_shape": (3, 3), "weight_shape": (10, 3)},
        ]
        
        for case in test_cases:
            case_name = f"embedding - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                weight_data = np.random.randn(*case["weight_shape"])
                input_data = np.random.randint(0, case["weight_shape"][0], size=case["input_shape"])
                
                rm_weight = rm.tensor(weight_data, requires_grad=True)
                rm_input = rm.tensor(input_data, dtype='int32')
                
                torch_weight = None
                torch_input = None
                if TORCH_AVAILABLE:
                    torch_weight = torch.tensor(weight_data, requires_grad=True)
                    torch_input = torch.tensor(input_data, dtype=torch.long)
                
                # 前向传播测试
                rm_output = rm.nn.functional.embedding(rm_input, rm_weight)
                torch_output = None
                if TORCH_AVAILABLE:
                    torch_output = torch.nn.functional.embedding(torch_input, torch_weight)
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_output, torch_output)
                
                # 反向传播测试
                backward_passed = True
                if TORCH_AVAILABLE:
                    # 计算损失
                    rm_loss = rm_output.sum()
                    torch_loss = torch_output.sum()
                    
                    # 反向传播
                    rm_loss.backward()
                    torch_loss.backward()
                    
                    # 比较梯度
                    backward_passed = compare_values(rm_weight.grad, torch_weight.grad)
                
                passed = forward_passed and backward_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                        print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"embedding测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_padding_idx(self):
        """测试padding_idx参数"""
        test_cases = [
            {"name": "padding_idx=0的处理", "padding_idx": 0},
            {"name": "padding_idx为负数的情况", "padding_idx": -1},
            {"name": "padding_idx=-1与scale_grad_by_freq结合", "padding_idx": -1, "scale_grad_by_freq": True},
        ]
        
        for case in test_cases:
            case_name = f"embedding - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                weight_shape = (10, 3)
                weight_data = np.random.randn(*weight_shape)
                input_data = np.array([0, 2, 0, 4]) if case["padding_idx"] == 0 else np.array([0, 2, 8, 9])
                
                rm_weight = rm.tensor(weight_data, requires_grad=True)
                rm_input = rm.tensor(input_data, dtype='int32')
                
                torch_weight = None
                torch_input = None
                if TORCH_AVAILABLE:
                    torch_weight = torch.tensor(weight_data, requires_grad=True)
                    torch_input = torch.tensor(input_data, dtype=torch.long)
                
                # 前向传播测试
                scale_grad_by_freq = case.get("scale_grad_by_freq", False)
                rm_output = rm.nn.functional.embedding(rm_input, rm_weight, padding_idx=case["padding_idx"], scale_grad_by_freq=scale_grad_by_freq)
                torch_output = None
                if TORCH_AVAILABLE:
                    torch_output = torch.nn.functional.embedding(torch_input, torch_weight, padding_idx=case["padding_idx"], scale_grad_by_freq=scale_grad_by_freq)
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_output, torch_output)
                
                # 反向传播测试
                backward_passed = True
                if TORCH_AVAILABLE:
                    # 计算损失
                    rm_loss = rm_output.sum()
                    torch_loss = torch_output.sum()
                    
                    # 反向传播
                    rm_loss.backward()
                    torch_loss.backward()
                    
                    # 比较梯度
                    backward_passed = compare_values(rm_weight.grad, torch_weight.grad)
                
                # 检查padding_idx的梯度是否为0
                padding_idx_grad_passed = True
                actual_padding_idx = case["padding_idx"]
                if actual_padding_idx < 0:
                    actual_padding_idx = weight_shape[0] + actual_padding_idx
                if not all(rm_weight.grad[actual_padding_idx].data == 0):
                    padding_idx_grad_passed = False
                    print(f"  padding_idx={case['padding_idx']}的梯度不为0")
                
                passed = forward_passed and backward_passed and padding_idx_grad_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                        print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                        print(f"  padding_idx梯度检查: {'通过' if padding_idx_grad_passed else '失败'}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"embedding测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_max_norm(self):
        """测试max_norm参数"""
        test_cases = [
            {"name": "max_norm参数", "max_norm": 2.0, "padding_idx": None},
            {"name": "max_norm与padding_idx结合的情况", "max_norm": 2.0, "padding_idx": 0},
            {"name": "max_norm与scale_grad_by_freq结合无padding_idx", "max_norm": 2.0, "padding_idx": None, "scale_grad_by_freq": True},
            {"name": "max_norm与padding_idx=-1结合", "max_norm": 2.0, "padding_idx": -1},
            {"name": "不同max_norm值 - max_norm=1.0", "max_norm": 1.0, "padding_idx": None},
            {"name": "不同max_norm值 - max_norm=3.0", "max_norm": 3.0, "padding_idx": None},
            {"name": "不同norm_type值 - norm_type=1.0(L1范数)", "max_norm": 2.0, "padding_idx": None, "norm_type": 1.0},
            {"name": "不同norm_type值 - norm_type=3.0(L3范数)", "max_norm": 2.0, "padding_idx": None, "norm_type": 3.0},
        ]
        
        for case in test_cases:
            case_name = f"embedding - {case['name']}"
            start_time = time.time()
            try:
                # 创建嵌入矩阵，其中有些向量的范数较大
                weight_data = np.array([
                    [3.0, 0.0, 0.0],  # 范数3
                    [0.0, 2.0, 0.0],  # 范数2
                    [1.0, 1.0, 1.0],  # 范数≈1.732
                    [0.0, 0.0, 4.0],  # 范数4
                    [2.0, 2.0, 2.0],  # 范数≈3.464
                ])
                
                # 对于scale_grad_by_freq=True的情况，使用包含重复索引的输入数据
                if case.get("scale_grad_by_freq", False):
                    input_data = np.array([0, 1, 0, 2, 1, 3, 4, 1])  # 包含重复索引
                else:
                    if case["padding_idx"] == 0:
                        input_data = np.array([0, 1, 0, 3, 4])
                    elif case["padding_idx"] == -1:
                        input_data = np.array([0, 1, 4, 3, 4])  # 包含padding_idx=-1（对应索引4）
                    else:
                        input_data = np.array([0, 1, 2, 3, 4])
                
                rm_weight = rm.tensor(weight_data, requires_grad=True)
                rm_input = rm.tensor(input_data, dtype='int32')
                
                torch_weight = None
                torch_input = None
                if TORCH_AVAILABLE:
                    torch_weight = torch.tensor(weight_data, requires_grad=True)
                    torch_input = torch.tensor(input_data, dtype=torch.long)
                
                # 前向传播测试
                scale_grad_by_freq = case.get("scale_grad_by_freq", False)
                norm_type = case.get("norm_type", 2.0)
                rm_output = rm.nn.functional.embedding(rm_input, rm_weight, max_norm=case["max_norm"], padding_idx=case["padding_idx"], scale_grad_by_freq=scale_grad_by_freq, norm_type=norm_type)
                torch_output = None
                if TORCH_AVAILABLE:
                    torch_output = torch.nn.functional.embedding(torch_input, torch_weight, max_norm=case["max_norm"], padding_idx=case["padding_idx"], scale_grad_by_freq=scale_grad_by_freq, norm_type=norm_type)
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_output, torch_output)
                
                # 反向传播测试
                backward_passed = True
                if TORCH_AVAILABLE:
                    # 计算损失
                    rm_loss = rm_output.sum()
                    torch_loss = torch_output.sum()
                    
                    # 反向传播
                    rm_loss.backward()
                    torch_loss.backward()
                    
                    # 比较梯度
                    backward_passed = compare_values(rm_weight.grad, torch_weight.grad)
                
                passed = forward_passed and backward_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                        print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"embedding测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_scale_grad_by_freq(self):
        """测试scale_grad_by_freq参数"""
        test_cases = [
            {"name": "scale_grad_by_freq参数", "padding_idx": None, "scale_grad_by_freq": True},
            {"name": "scale_grad_by_freq与padding_idx=0结合", "padding_idx": 0, "scale_grad_by_freq": True},
            {"name": "scale_grad_by_freq与padding_idx=-1结合", "padding_idx": -1, "scale_grad_by_freq": True},
        ]
        
        for case in test_cases:
            case_name = f"embedding - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                weight_shape = (5, 3)
                weight_data = np.random.randn(*weight_shape)
                
                # 根据padding_idx调整输入数据
                if case["padding_idx"] == 0:
                    input_data = np.array([0, 1, 0, 2, 1, 1])  # 0出现2次（作为padding），1出现3次，2出现1次
                elif case["padding_idx"] == -1:
                    input_data = np.array([0, 1, 4, 2, 1, 4])  # 4出现2次（作为padding_idx=-1），0出现1次，1出现2次，2出现1次
                else:
                    input_data = np.array([0, 1, 0, 2, 1, 1])  # 0出现2次，1出现3次，2出现1次
                
                rm_weight = rm.tensor(weight_data, requires_grad=True)
                rm_input = rm.tensor(input_data, dtype='int32')
                
                torch_weight = None
                torch_input = None
                if TORCH_AVAILABLE:
                    torch_weight = torch.tensor(weight_data, requires_grad=True)
                    torch_input = torch.tensor(input_data, dtype=torch.long)
                
                # 前向传播测试
                rm_output = rm.nn.functional.embedding(rm_input, rm_weight, padding_idx=case["padding_idx"], scale_grad_by_freq=case["scale_grad_by_freq"])
                torch_output = None
                if TORCH_AVAILABLE:
                    torch_output = torch.nn.functional.embedding(torch_input, torch_weight, padding_idx=case["padding_idx"], scale_grad_by_freq=case["scale_grad_by_freq"])
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_output, torch_output)
                
                # 反向传播测试
                backward_passed = True
                if TORCH_AVAILABLE:
                    # 计算损失
                    rm_loss = rm_output.sum()
                    torch_loss = torch_output.sum()
                    
                    # 反向传播
                    rm_loss.backward()
                    torch_loss.backward()
                    
                    # 比较梯度
                    backward_passed = compare_values(rm_weight.grad, torch_weight.grad)
                
                passed = forward_passed and backward_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                        print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"embedding测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_comprehensive_comparison(self):
        """与PyTorch进行全面比较"""
        if not TORCH_AVAILABLE:
            return
        
        test_cases = [
            {"name": "无padding_idx", "input_data": [0, 1, 2, 3, 4], "padding_idx": None, "max_norm": None, "scale_grad_by_freq": False},
            {"name": "padding_idx=0", "input_data": [0, 2, 0, 4], "padding_idx": 0, "max_norm": None, "scale_grad_by_freq": False},
            {"name": "scale_grad_by_freq=True", "input_data": [0, 1, 0, 2, 1, 1], "padding_idx": None, "max_norm": None, "scale_grad_by_freq": True},
            {"name": "max_norm=2.0", "input_data": [0, 1, 2, 3, 4], "padding_idx": None, "max_norm": 2.0, "scale_grad_by_freq": False},
            {"name": "所有参数结合", "input_data": [0, 1, 0, 3, 4], "padding_idx": 0, "max_norm": 2.0, "scale_grad_by_freq": True},
            {"name": "负数padding_idx与所有参数结合", "input_data": [0, 1, 4, 3, 4], "padding_idx": -1, "max_norm": 2.0, "scale_grad_by_freq": True},
            {"name": "max_norm=2.0与norm_type=1.0(L1范数)", "input_data": [0, 1, 2, 3, 4], "padding_idx": None, "max_norm": 2.0, "norm_type": 1.0, "scale_grad_by_freq": False},
        ]
        
        for case in test_cases:
            case_name = f"embedding - 与PyTorch比较 - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                weight_shape = (5, 3)
                weight_data = np.random.randn(*weight_shape)
                input_data = np.array(case["input_data"])
                
                rm_weight = rm.tensor(weight_data, requires_grad=True)
                rm_input = rm.tensor(input_data, dtype='int32')
                
                torch_weight = torch.tensor(weight_data, requires_grad=True)
                torch_input = torch.tensor(input_data, dtype=torch.long)
                
                # 前向传播测试
                norm_type = case.get("norm_type", 2.0)
                rm_output = rm.nn.functional.embedding(
                    rm_input, rm_weight, 
                    padding_idx=case["padding_idx"], 
                    max_norm=case["max_norm"], 
                    norm_type=norm_type,
                    scale_grad_by_freq=case["scale_grad_by_freq"]
                )
                torch_output = torch.nn.functional.embedding(
                    torch_input, torch_weight, 
                    padding_idx=case["padding_idx"], 
                    max_norm=case["max_norm"], 
                    norm_type=norm_type,
                    scale_grad_by_freq=case["scale_grad_by_freq"]
                )
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_output, torch_output)
                
                # 反向传播测试
                # 计算损失
                rm_loss = rm_output.sum()
                torch_loss = torch_output.sum()
                
                # 反向传播
                rm_loss.backward()
                torch_loss.backward()
                
                # 比较梯度
                backward_passed = compare_values(rm_weight.grad, torch_weight.grad)
                
                passed = forward_passed and backward_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                        print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"embedding测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_embedding_class(self):
        """测试Embedding类的特有功能"""
        start_time = time.time()
        try:
            # 创建Embedding类实例
            num_embeddings = 10
            embedding_dim = 3
            padding_idx = 0
            
            # 测试1: 类的基本使用
            rm_embedding = rm.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
            self.assertIsInstance(rm_embedding, rm.nn.Module)
            self.assertEqual(rm_embedding.num_embeddings, num_embeddings)
            self.assertEqual(rm_embedding.embedding_dim, embedding_dim)
            self.assertEqual(rm_embedding.padding_idx, padding_idx)
            
            # 测试2: 参数管理（权重注册）
            self.assertIn('weight', rm_embedding._parameters)
            self.assertIsInstance(rm_embedding.weight, rm.nn.Parameter)
            self.assertEqual(rm_embedding.weight.shape, (num_embeddings, embedding_dim))
            
            # 测试3: padding_idx的权重初始化
            self.assertTrue(np.allclose(rm_embedding.weight.data[padding_idx], 0.0))
            
            # 测试4: 前向传播
            input_data = np.array([0, 1, 2, 0, 3])
            rm_input = rm.tensor(input_data, dtype='int32')
            rm_output = rm_embedding(rm_input)
            self.assertEqual(rm_output.shape, (len(input_data), embedding_dim))
            
            # 测试5: 与PyTorch的兼容性
            if TORCH_AVAILABLE:
                torch_embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
                torch_embedding.weight.data = torch.tensor(rm_embedding.weight.data)
                
                torch_input = torch.tensor(input_data, dtype=torch.long)
                torch_output = torch_embedding(torch_input)
                
                self.assertTrue(compare_values(rm_output, torch_output))
                
                # 反向传播测试
                rm_loss = rm_output.sum()
                torch_loss = torch_output.sum()
                
                rm_loss.backward()
                torch_loss.backward()
                
                self.assertTrue(compare_values(rm_embedding.weight.grad, torch_embedding.weight.grad))
            
            # 测试6: 类的额外表示信息
            repr_str = rm_embedding.extra_repr()
            self.assertIn(f"{num_embeddings}, {embedding_dim}", repr_str)
            self.assertIn(f"padding_idx={padding_idx}", repr_str)
            
            case_name = "Embedding类测试"
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, True)
                status = "通过"
                time_taken = time.time() - start_time
                print(f"测试用例: {case_name} - {Colors.OKGREEN}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
        except Exception as e:
            case_name = "Embedding类测试"
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
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行embedding函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmbedding)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(suite)
    
    # 打印统计信息
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)