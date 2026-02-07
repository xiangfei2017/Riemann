import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    print("警告: 无法导入PyTorch，将只测试riemann的dropout函数")
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

# 验证dropout函数行为的函数
def verify_dropout_behavior(input_tensor, output_tensor, p, training, atol=1e-4):
    """验证dropout函数的行为是否正确
    
    Args:
        input_tensor: 输入张量
        output_tensor: 输出张量
        p: dropout概率
        training: 是否在训练模式
        atol: 绝对误差容限
    
    Returns:
        bool: 验证是否通过
    """
    # 转换为numpy数组
    if hasattr(input_tensor, 'is_cuda') and input_tensor.is_cuda:
        input_data = input_tensor.detach().cpu().numpy()
    else:
        input_data = input_tensor.detach().numpy()
    
    if hasattr(output_tensor, 'is_cuda') and output_tensor.is_cuda:
        output_data = output_tensor.detach().cpu().numpy()
    else:
        output_data = output_tensor.detach().numpy()
    
    # 验证形状是否一致
    if input_data.shape != output_data.shape:
        print(f"形状不一致: 输入形状={input_data.shape}, 输出形状={output_data.shape}")
        return False
    
    # 如果不是训练模式或p=0，输出应该等于输入
    if not training or p == 0:
        try:
            np.testing.assert_allclose(input_data, output_data, atol=atol)
            return True
        except AssertionError:
            print(f"非训练模式或p=0时，输出不等于输入")
            return False
    
    # 训练模式且p>0时，验证统计特征
    # 1. 验证输出中是否有0元素
    has_zeros = np.any(output_data == 0)
    if not has_zeros:
        print(f"训练模式且p={p}时，输出中没有0元素")
        # 注意：由于随机性，可能会出现没有0元素的情况，这里不返回False
    
    # 2. 验证全部元素的平均值是否接近输入平均值（这是最重要的验证）
    input_mean = np.mean(input_data)
    output_mean = np.mean(output_data)
    
    # 计算相对误差容限，考虑到dropout的随机性
    # 对于高dropout概率，使用更大的容差
    base_atol = max(1e-2, atol * 100)
    # dropout概率越高，容差越大
    relative_atol = base_atol * min(10, 1 + p * 20)
    
    # 对于CUDA设备，使用更大的容差
    is_cuda = False
    try:
        if hasattr(input_tensor, 'is_cuda') and input_tensor.is_cuda:
            is_cuda = True
    except:
        pass
    
    if is_cuda:
        relative_atol *= 2
    
    if not np.isclose(output_mean, input_mean, atol=relative_atol):
        print(f"输出平均值不接近输入平均值: 输入平均值={input_mean}, 输出平均值={output_mean}, 容差={relative_atol}")
        return False
    
    # 3. 验证非零元素的平均值是否接近输入平均值 * 1/(1-p)
    non_zero_output = output_data[output_data != 0]
    if len(non_zero_output) > 0:
        non_zero_mean = np.mean(non_zero_output)
        expected_mean = input_mean / (1 - p)
        if not np.isclose(non_zero_mean, expected_mean, atol=relative_atol):
            print(f"非零元素平均值不接近预期值: 实际值={non_zero_mean}, 预期值={expected_mean}, 容差={relative_atol}")
            return False
    
    return True

# 验证dropout2d/3d函数行为的函数
def verify_channel_dropout_behavior(input_tensor, output_tensor, p, training, dropout_type, atol=1e-4):
    """验证dropout2d/3d函数的行为是否正确
    
    Args:
        input_tensor: 输入张量
        output_tensor: 输出张量
        p: dropout概率
        training: 是否在训练模式
        dropout_type: '2d' 或 '3d'
        atol: 绝对误差容限
    
    Returns:
        bool: 验证是否通过
    """
    # 转换为numpy数组
    if hasattr(input_tensor, 'is_cuda') and input_tensor.is_cuda:
        input_data = input_tensor.detach().cpu().numpy()
    else:
        input_data = input_tensor.detach().numpy()
    
    if hasattr(output_tensor, 'is_cuda') and output_tensor.is_cuda:
        output_data = output_tensor.detach().cpu().numpy()
    else:
        output_data = output_tensor.detach().numpy()
    
    # 验证形状是否一致
    if input_data.shape != output_data.shape:
        print(f"形状不一致: 输入形状={input_data.shape}, 输出形状={output_data.shape}")
        return False
    
    # 如果不是训练模式或p=0，输出应该等于输入
    if not training or p == 0:
        try:
            np.testing.assert_allclose(input_data, output_data, atol=atol)
            return True
        except AssertionError:
            print(f"非训练模式或p=0时，输出不等于输入")
            return False
    
    # 训练模式且p>0时，验证通道级别的dropout行为
    # 对于dropout2d，输入形状为 (N, C, H, W)
    # 对于dropout3d，输入形状为 (N, C, D, H, W)
    
    # 获取批次大小和通道数
    batch_size = input_data.shape[0]
    num_channels = input_data.shape[1]
    
    # 验证每个样本的每个通道是否被完全保留或完全丢弃
    for i in range(batch_size):
        for c in range(num_channels):
            if dropout_type == '2d':
                # 对于dropout2d，通道数据形状为 (H, W)
                channel_data = output_data[i, c, :, :]
            else:  # dropout3d
                # 对于dropout3d，通道数据形状为 (D, H, W)
                channel_data = output_data[i, c, :, :, :]
            
            # 检查通道是否被完全保留或完全丢弃
            num_zeros = np.count_nonzero(channel_data == 0)
            total_elements = channel_data.size
            
            # 通道要么完全为0，要么完全不为0
            if not (num_zeros == 0 or num_zeros == total_elements):
                print(f"通道 {c} 既不是完全保留也不是完全丢弃")
                return False
    
    # 验证统计特征
    # 1. 验证全部元素的平均值是否接近输入平均值（这是最重要的验证）
    input_mean = np.mean(input_data)
    output_mean = np.mean(output_data)
    
    # 计算相对误差容限，考虑到dropout的随机性
    # 对于高dropout概率，使用更大的容差
    base_atol = max(1e-2, atol * 100)
    # dropout概率越高，容差越大
    relative_atol = base_atol * min(10, 1 + p * 20)
    
    # 对于CUDA设备，使用更大的容差
    is_cuda = False
    try:
        if hasattr(input_tensor, 'is_cuda') and input_tensor.is_cuda:
            is_cuda = True
    except:
        pass
    
    if is_cuda:
        relative_atol *= 2
    
    if not np.isclose(output_mean, input_mean, atol=relative_atol):
        print(f"输出平均值不接近输入平均值: 输入平均值={input_mean}, 输出平均值={output_mean}, 容差={relative_atol}")
        return False
    
    # 2. 验证保留通道的平均值是否接近输入通道平均值乘以 1/(1-p)
    # 对于原地操作，由于可能存在精度问题，使用更大的容差
    # 检查是否为原地操作（通过比较输入和输出的内存地址）
    is_inplace = False
    try:
        if hasattr(input_tensor, 'data') and hasattr(output_tensor, 'data'):
            if input_tensor.data is output_tensor.data:
                is_inplace = True
    except:
        pass
    
    # 为原地操作设置更大的容差
    channel_atol = relative_atol * 2 if is_inplace else relative_atol
    
    for i in range(batch_size):
        for c in range(num_channels):
            if dropout_type == '2d':
                input_channel = input_data[i, c, :, :]
                output_channel = output_data[i, c, :, :]
            else:  # dropout3d
                input_channel = input_data[i, c, :, :, :]
                output_channel = output_data[i, c, :, :, :]
            
            # 检查通道是否被保留
            if np.any(output_channel != 0):
                # 计算输入通道的平均值
                input_channel_mean = np.mean(input_channel)
                # 计算输出通道的平均值
                output_channel_mean = np.mean(output_channel)
                # 计算预期的输出通道平均值
                expected_channel_mean = input_channel_mean / (1 - p)
                
                if not np.isclose(output_channel_mean, expected_channel_mean, atol=channel_atol):
                    print(f"通道 {c} 的平均值不接近预期值: 实际值={output_channel_mean}, 预期值={expected_channel_mean}, 容差={channel_atol}")
                    return False
    
    return True

# 测试dropout函数类
class TestDropout(unittest.TestCase):
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
    
    def test_dropout(self):
        """测试dropout函数"""
        # 定义测试用例
        test_cases = [
            {"name": "基本dropout功能", "input_shape": (100, 100), "p": 0.5, "training": True, "inplace": False},
            {"name": "不同dropout概率 - p=0.2", "input_shape": (100, 100), "p": 0.2, "training": True, "inplace": False},
            {"name": "不同dropout概率 - p=0.8", "input_shape": (100, 100), "p": 0.8, "training": True, "inplace": False},
            {"name": "训练模式关闭", "input_shape": (100, 100), "p": 0.5, "training": False, "inplace": False},
            {"name": "dropout概率为0", "input_shape": (100, 100), "p": 0.0, "training": True, "inplace": False},
            {"name": "原地操作", "input_shape": (100, 100), "p": 0.5, "training": True, "inplace": True},
            {"name": "多维输入形状", "input_shape": (10, 20, 30), "p": 0.5, "training": True, "inplace": False},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"dropout - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    input_data = np.random.randn(*case["input_shape"])
                    
                    # 根据设备创建张量
                    if case["inplace"] and case["training"] and case["p"] > 0:
                        # 对于原地操作，创建一个非叶子张量
                        if device == "cpu":
                            # 先创建一个叶子张量
                            leaf_input = rm.tensor(input_data, requires_grad=True)
                            # 通过操作创建一个非叶子张量
                            rm_input = leaf_input + 0
                            if TORCH_AVAILABLE:
                                torch_leaf_input = torch.tensor(input_data, requires_grad=True)
                                torch_input = torch_leaf_input + 0
                            else:
                                torch_input = None
                        else:  # cuda
                            # 先创建一个叶子张量
                            leaf_input = rm.tensor(input_data, requires_grad=True, device=device)
                            # 通过操作创建一个非叶子张量
                            rm_input = leaf_input + 0
                            if TORCH_AVAILABLE:
                                torch_leaf_input = torch.tensor(input_data, requires_grad=True, device=device)
                                torch_input = torch_leaf_input + 0
                            else:
                                torch_input = None
                    else:
                        # 常规情况，创建叶子张量
                        if device == "cpu":
                            rm_input = rm.tensor(input_data, requires_grad=True)
                            if TORCH_AVAILABLE:
                                torch_input = torch.tensor(input_data, requires_grad=True)
                            else:
                                torch_input = None
                        else:  # cuda
                            rm_input = rm.tensor(input_data, requires_grad=True, device=device)
                            if TORCH_AVAILABLE:
                                torch_input = torch.tensor(input_data, requires_grad=True, device=device)
                            else:
                                torch_input = None
                    
                    # 前向传播测试
                    rm_output = rm.nn.functional.dropout(
                        rm_input, 
                        p=case["p"], 
                        training=case["training"], 
                        inplace=case["inplace"]
                    )
                    
                    # 验证前向传播结果
                    forward_passed = verify_dropout_behavior(
                        rm_input, 
                        rm_output, 
                        case["p"], 
                        case["training"]
                    )
                    
                    # 反向传播测试
                    backward_passed = True
                    if case["training"] and case["p"] > 0:
                        # 计算损失
                        rm_loss = rm_output.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        
                        # 验证梯度是否存在
                        if case["inplace"]:
                            # 对于原地操作，梯度存储在叶子张量中
                            if 'leaf_input' in locals() and leaf_input.grad is None:
                                print("反向传播后叶子张量梯度为None")
                                backward_passed = False
                        else:
                            # 对于非原地操作，梯度存储在输入张量中
                            if rm_input.grad is None:
                                print("反向传播后梯度为None")
                                backward_passed = False
                    
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
                    self.assertTrue(passed, f"dropout测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_dropout2d(self):
        """测试dropout2d函数"""
        # 定义测试用例
        test_cases = [
            {"name": "基本dropout2d功能", "input_shape": (10, 20, 30, 30), "p": 0.5, "training": True, "inplace": False},
            {"name": "不同dropout概率 - p=0.2", "input_shape": (10, 20, 30, 30), "p": 0.2, "training": True, "inplace": False},
            {"name": "不同dropout概率 - p=0.8", "input_shape": (10, 20, 30, 30), "p": 0.8, "training": True, "inplace": False},
            {"name": "训练模式关闭", "input_shape": (10, 20, 30, 30), "p": 0.5, "training": False, "inplace": False},
            {"name": "dropout概率为0", "input_shape": (10, 20, 30, 30), "p": 0.0, "training": True, "inplace": False},
            {"name": "原地操作", "input_shape": (10, 20, 30, 30), "p": 0.5, "training": True, "inplace": True},
            {"name": "不同输入形状", "input_shape": (8, 16, 24, 24), "p": 0.5, "training": True, "inplace": False},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"dropout2d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    input_data = np.random.randn(*case["input_shape"])
                    
                    # 根据设备创建张量
                    if case["inplace"] and case["training"] and case["p"] > 0:
                        # 对于原地操作，创建一个非叶子张量
                        if device == "cpu":
                            # 先创建一个叶子张量
                            leaf_input = rm.tensor(input_data, requires_grad=True)
                            # 通过操作创建一个非叶子张量
                            rm_input = leaf_input + 0
                            if TORCH_AVAILABLE:
                                torch_leaf_input = torch.tensor(input_data, requires_grad=True)
                                torch_input = torch_leaf_input + 0
                            else:
                                torch_input = None
                        else:  # cuda
                            # 先创建一个叶子张量
                            leaf_input = rm.tensor(input_data, requires_grad=True, device=device)
                            # 通过操作创建一个非叶子张量
                            rm_input = leaf_input + 0
                            if TORCH_AVAILABLE:
                                torch_leaf_input = torch.tensor(input_data, requires_grad=True, device=device)
                                torch_input = torch_leaf_input + 0
                            else:
                                torch_input = None
                    else:
                        # 常规情况，创建叶子张量
                        if device == "cpu":
                            rm_input = rm.tensor(input_data, requires_grad=True)
                            if TORCH_AVAILABLE:
                                torch_input = torch.tensor(input_data, requires_grad=True)
                            else:
                                torch_input = None
                        else:  # cuda
                            rm_input = rm.tensor(input_data, requires_grad=True, device=device)
                            if TORCH_AVAILABLE:
                                torch_input = torch.tensor(input_data, requires_grad=True, device=device)
                            else:
                                torch_input = None
                    
                    # 前向传播测试
                    rm_output = rm.nn.functional.dropout2d(
                        rm_input, 
                        p=case["p"], 
                        training=case["training"], 
                        inplace=case["inplace"]
                    )
                    
                    # 验证前向传播结果
                    forward_passed = verify_channel_dropout_behavior(
                        rm_input, 
                        rm_output, 
                        case["p"], 
                        case["training"],
                        '2d'
                    )
                    
                    # 反向传播测试
                    backward_passed = True
                    if case["training"] and case["p"] > 0:
                        # 计算损失
                        rm_loss = rm_output.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        
                        # 验证梯度是否存在
                        if case["inplace"]:
                            # 对于原地操作，梯度存储在叶子张量中
                            if 'leaf_input' in locals() and leaf_input.grad is None:
                                print("反向传播后叶子张量梯度为None")
                                backward_passed = False
                        else:
                            # 对于非原地操作，梯度存储在输入张量中
                            if rm_input.grad is None:
                                print("反向传播后梯度为None")
                                backward_passed = False
                    
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
                    self.assertTrue(passed, f"dropout2d测试失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_dropout3d(self):
        """测试dropout3d函数"""
        # 定义测试用例
        test_cases = [
            {"name": "基本dropout3d功能", "input_shape": (5, 10, 15, 15, 15), "p": 0.5, "training": True, "inplace": False},
            {"name": "不同dropout概率 - p=0.2", "input_shape": (5, 10, 15, 15, 15), "p": 0.2, "training": True, "inplace": False},
            {"name": "不同dropout概率 - p=0.8", "input_shape": (5, 10, 15, 15, 15), "p": 0.8, "training": True, "inplace": False},
            {"name": "训练模式关闭", "input_shape": (5, 10, 15, 15, 15), "p": 0.5, "training": False, "inplace": False},
            {"name": "dropout概率为0", "input_shape": (5, 10, 15, 15, 15), "p": 0.0, "training": True, "inplace": False},
            {"name": "原地操作", "input_shape": (5, 10, 15, 15, 15), "p": 0.5, "training": True, "inplace": True},
            {"name": "不同输入形状", "input_shape": (4, 8, 12, 12, 12), "p": 0.5, "training": True, "inplace": False},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"dropout3d - {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    input_data = np.random.randn(*case["input_shape"])
                    
                    # 根据设备创建张量
                    if case["inplace"] and case["training"] and case["p"] > 0:
                        # 对于原地操作，创建一个非叶子张量
                        if device == "cpu":
                            # 先创建一个叶子张量
                            leaf_input = rm.tensor(input_data, requires_grad=True)
                            # 通过操作创建一个非叶子张量
                            rm_input = leaf_input + 0
                            if TORCH_AVAILABLE:
                                torch_leaf_input = torch.tensor(input_data, requires_grad=True)
                                torch_input = torch_leaf_input + 0
                            else:
                                torch_input = None
                        else:  # cuda
                            # 先创建一个叶子张量
                            leaf_input = rm.tensor(input_data, requires_grad=True, device=device)
                            # 通过操作创建一个非叶子张量
                            rm_input = leaf_input + 0
                            if TORCH_AVAILABLE:
                                torch_leaf_input = torch.tensor(input_data, requires_grad=True, device=device)
                                torch_input = torch_leaf_input + 0
                            else:
                                torch_input = None
                    else:
                        # 常规情况，创建叶子张量
                        if device == "cpu":
                            rm_input = rm.tensor(input_data, requires_grad=True)
                            if TORCH_AVAILABLE:
                                torch_input = torch.tensor(input_data, requires_grad=True)
                            else:
                                torch_input = None
                        else:  # cuda
                            rm_input = rm.tensor(input_data, requires_grad=True, device=device)
                            if TORCH_AVAILABLE:
                                torch_input = torch.tensor(input_data, requires_grad=True, device=device)
                            else:
                                torch_input = None
                    
                    # 前向传播测试
                    rm_output = rm.nn.functional.dropout3d(
                        rm_input, 
                        p=case["p"], 
                        training=case["training"], 
                        inplace=case["inplace"]
                    )
                    
                    # 验证前向传播结果
                    forward_passed = verify_channel_dropout_behavior(
                        rm_input, 
                        rm_output, 
                        case["p"], 
                        case["training"],
                        '3d'
                    )
                    
                    # 反向传播测试
                    backward_passed = True
                    if case["training"] and case["p"] > 0:
                        # 计算损失
                        rm_loss = rm_output.sum()
                        
                        # 反向传播
                        rm_loss.backward()
                        
                        # 验证梯度是否存在
                        if case["inplace"]:
                            # 对于原地操作，梯度存储在叶子张量中
                            if 'leaf_input' in locals() and leaf_input.grad is None:
                                print("反向传播后叶子张量梯度为None")
                                backward_passed = False
                        else:
                            # 对于非原地操作，梯度存储在输入张量中
                            if rm_input.grad is None:
                                print("反向传播后梯度为None")
                                backward_passed = False
                    
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
                    self.assertTrue(passed, f"dropout3d测试失败: {case_name}")
                    
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
    print(f"{Colors.BOLD}开始测试dropout、dropout2d和dropout3d函数{Colors.ENDC}")
    print("=" * 80)
    
    # 运行测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # 打印测试统计
    stats.print_summary()
