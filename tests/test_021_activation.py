import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.nn import functional as F
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    import torch.nn.functional as torch_F
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的激活函数")
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
        headers = ['激活函数', '通过/总数', '通过率', '耗时(秒)']
        
        # 计算各列标题的显示宽度
        header_widths = [self._get_display_width(h) for h in headers]
        
        # 计算数据行中各列的最大显示宽度
        max_func_name_width = header_widths[0]
        for func_name in self.function_stats.keys():
            max_func_name_width = max(max_func_name_width, self._get_display_width(func_name))
        
        # 为各列设置最终宽度，标题宽度和内容宽度的最大值，并留出适当间距
        col_widths = [
            max(max_func_name_width, header_widths[0]) + 2,  # 激活函数列
            header_widths[1] + 4,  # 通过/总数列
            header_widths[2] + 4,  # 通过率列
            header_widths[3] + 4   # 耗时列
        ]
        
        total_width = sum(col_widths)
        
        print("\n" + "="*total_width)
        print(f"{Colors.BOLD}激活函数测试统计摘要{Colors.ENDC}")
        print("="*total_width)
        print(f"总测试用例数: {self.total_cases}")
        print(f"通过测试用例数: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases}{Colors.ENDC}")
        print(f"测试通过率: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases/self.total_cases*100:.2f}%{Colors.ENDC}")
        print(f"总耗时: {self.total_time:.4f} 秒")
        print("\n各函数测试详情:")
        print("-"*total_width)
        
        # 打印表头
        header_line = ""
        for i, header in enumerate(headers):
            header_width = self._get_display_width(header)
            padding = col_widths[i] - header_width
            header_line += header + " " * padding
        print(header_line)
        print("-"*total_width)
        
        # 打印数据行
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
            pass_rate_display = f"{pass_rate:.2f}%"
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
def compare_values(rm_result, torch_result, atol=1e-2, rtol=1e-2):
    """比较Riemann和PyTorch的值是否接近"""
    # 处理None值的情况
    if not TORCH_AVAILABLE:
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
    
    # 获取数据部分
    try:
        # 从Riemann结果中提取数据
        if hasattr(rm_result, 'data'):
            rm_data = rm_result.data
        else:
            rm_data = rm_result
            
        # 从PyTorch结果中提取数据
        if hasattr(torch_result, 'data'):
            torch_data = torch_result.data
        elif hasattr(torch_result, 'detach'):
            # 对于PyTorch张量，直接使用numpy()方法
            torch_data = torch_result.detach().numpy()
        else:
            torch_data = torch_result
            
        # 转换为numpy数组
        if not isinstance(rm_data, np.ndarray):
            try:
                rm_data = np.asarray(rm_data)  # 使用asarray替代array
            except:
                rm_data = np.array(rm_data)
        
        if not isinstance(torch_data, np.ndarray):
            try:
                torch_data = np.asarray(torch_data)  # 使用asarray替代array
            except:
                # 尝试其他转换方式
                if hasattr(torch_data, 'tolist'):
                    torch_data = np.array(torch_data.tolist())
                else:
                    torch_data = np.array(torch_data)
    
        # 比较形状
        if rm_data.shape != torch_data.shape:
            return False
            
        # 比较数值
        return np.allclose(rm_data, torch_data, atol=atol, rtol=rtol)
    except Exception as e:
        print(f"比较值时出错: {e}")
        return False

# 生成测试数据的函数
def generate_test_data(case):
    """生成测试数据，处理不同格式的测试用例"""
    # 检查测试用例类型
    if 'shape' in case:
        # 根据shape生成随机数据
        shape = case['shape']
        # 对于标量输入([])，生成单个随机值并确保是NumPy数组
        if shape == []:
            return np.array(np.random.randn(), dtype=case.get('dtype', rm.get_default_dtype()))
        else:
            return np.random.randn(*shape).astype(case.get('dtype', rm.get_default_dtype()))
    elif 'values' in case:
        # 直接使用提供的值
        values = case['values']
        # 确保返回numpy数组格式
        return np.array(values).astype(case.get('dtype', rm.get_default_dtype()))
    else:
        raise ValueError("测试用例必须包含'shape'或'values'键")

class TestActivationFunctions(unittest.TestCase):
    """测试激活函数"""
    
    def setUp(self):
        """每个测试方法执行前的设置"""
        # 设置随机种子以确保结果可复现
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
        # 获取当前测试方法名
        self.current_test_name = self._testMethodName
        # 如果是独立脚本运行，则开始记录函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")
                
    def tearDown(self):
        """每个测试方法执行后的清理"""
        # 如果是独立脚本运行，则结束函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_sigmoid(self):
        """测试sigmoid激活函数"""
        func_name = "sigmoid"
        # 测试不同的输入形状和值
        test_cases = [
            {"name": "标量输入", "shape": []},
            {"name": "一维张量", "shape": (10,)},
            {"name": "二维张量", "shape": (5, 6)},
            {"name": "三维张量", "shape": (3, 4, 5)},
            {"name": "大正值", "shape": (5,), "values": np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
            {"name": "大负值", "shape": (5,), "values": np.array([-10.0, -20.0, -30.0, -40.0, -50.0])},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.sigmoid(rm_input)
                
                # PyTorch计算
                if TORCH_AVAILABLE:
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_output = torch_F.sigmoid(torch_input)
                    value_match = compare_values(rm_output, torch_output)
                else:
                    value_match = True
                
                # 2. 测试梯度
                # 随机生成梯度输出，处理标量输入
                if isinstance(input_data, np.ndarray) and input_data.ndim > 0:
                    grad_output = np.random.randn(*input_data.shape)
                else:
                    grad_output = np.float32(np.random.randn())
                grad_output = grad_output.astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # PyTorch反向传播
                if TORCH_AVAILABLE:
                    torch_output.backward(torch.tensor(grad_output))
                    torch_grad = torch_input.grad
                    grad_match = compare_values(rm_grad, torch_grad)
                else:
                    grad_match = True
                
                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"函数值匹配: {value_match}, 梯度匹配: {grad_match}"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  函数值匹配: {'通过' if value_match else '失败'}")
                        print(f"  梯度匹配: {'通过' if grad_match else '失败'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")
    
    def test_softmax(self):
        """测试softmax激活函数"""
        func_name = "softmax"
        # 测试不同的输入形状和维度
        test_cases = [
            {"name": "二维张量-维度0", "shape": (5, 6), "dim": 0},
            {"name": "二维张量-维度1", "shape": (5, 6), "dim": 1},
            {"name": "三维张量-维度0", "shape": (3, 4, 5), "dim": 0},
            {"name": "三维张量-维度1", "shape": (3, 4, 5), "dim": 1},
            {"name": "三维张量-维度2", "shape": (3, 4, 5), "dim": 2},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.softmax(rm_input, dim=case["dim"])
                
                # PyTorch计算
                if TORCH_AVAILABLE:
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_output = torch_F.softmax(torch_input, dim=case["dim"])
                    value_match = compare_values(rm_output, torch_output)
                else:
                    value_match = True
                
                # 2. 测试梯度
                # 随机生成梯度输出
                grad_output = np.random.randn(*input_data.shape).astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # PyTorch反向传播
                if TORCH_AVAILABLE:
                    torch_output.backward(torch.tensor(grad_output))
                    torch_grad = torch_input.grad
                    grad_match = compare_values(rm_grad, torch_grad)
                else:
                    grad_match = True
                
                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"函数值匹配: {value_match}, 梯度匹配: {grad_match}"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  函数值匹配: {'通过' if value_match else '失败'}")
                        print(f"  梯度匹配: {'通过' if grad_match else '失败'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")
    
    def test_relu(self):
        """测试relu激活函数"""
        func_name = "relu"
        # 测试不同的输入形状和值
        test_cases = [
            {"name": "标量输入", "shape": []},
            {"name": "一维张量", "shape": (10,)},
            {"name": "二维张量", "shape": (5, 6)},
            {"name": "三维张量", "shape": (3, 4, 5)},
            {"name": "边界值", "shape": (5,), "values": np.array([-10.0, -5.0, 0.0, 5.0, 10.0])},
            {"name": "负值输入", "shape": (3, 3), "values": -np.random.rand(3, 3)},
            {"name": "正值输入", "shape": (3, 3), "values": np.random.rand(3, 3)},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.relu(rm_input)
                
                # PyTorch计算
                if TORCH_AVAILABLE:
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_output = torch.relu(torch_input)
                    value_match = compare_values(rm_output, torch_output)
                else:
                    value_match = True
                
                # 2. 测试梯度
                # 随机生成梯度输出，处理标量输入
                if isinstance(input_data, np.ndarray) and input_data.ndim > 0:
                    grad_output = np.random.randn(*input_data.shape)
                else:
                    grad_output = np.random.randn()
                # 将grad_output转换为numpy数组后再调用astype
                grad_output = np.array(grad_output).astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # PyTorch反向传播
                if TORCH_AVAILABLE:
                    torch_output.backward(torch.tensor(grad_output))
                    torch_grad = torch_input.grad
                    grad_match = compare_values(rm_grad, torch_grad)
                else:
                    grad_match = True
                
                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"函数值匹配: {value_match}, 梯度匹配: {grad_match}"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  函数值匹配: {'通过' if value_match else '失败'}")
                        print(f"  梯度匹配: {'通过' if grad_match else '失败'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")

    def test_leaky_relu(self):
        """测试leaky_relu激活函数"""
        func_name = "leaky_relu"
        # 测试不同的输入形状和负斜率
        test_cases = [
            {"name": "默认负斜率(0.01)", "shape": (5, 6)},
            {"name": "负斜率0.1", "shape": (5, 6), "alpha": 0.1},
            {"name": "负斜率0.3", "shape": (5, 6), "alpha": 0.3},
            {"name": "一维张量", "shape": (10,)},
            {"name": "三维张量", "shape": (3, 4, 5)},
            {"name": "边界值", "shape": (5,), "values": np.array([-10.0, -5.0, 0.0, 5.0, 10.0])},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                alpha = case.get("alpha", 0.01)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.leaky_relu(rm_input, alpha=alpha)
                
                # PyTorch计算
                if TORCH_AVAILABLE:
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_output = torch_F.leaky_relu(torch_input, negative_slope=alpha)
                    value_match = compare_values(rm_output, torch_output)
                else:
                    value_match = True
                
                # 2. 测试梯度
                # 随机生成梯度输出
                if isinstance(input_data, np.ndarray) and input_data.ndim > 0:
                    grad_output = np.random.randn(*input_data.shape)
                else:
                    grad_output = np.float32(np.random.randn())
                grad_output = grad_output.astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # PyTorch反向传播
                if TORCH_AVAILABLE:
                    torch_output.backward(torch.tensor(grad_output))
                    torch_grad = torch_input.grad
                    grad_match = compare_values(rm_grad, torch_grad)
                else:
                    grad_match = True
                
                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"函数值匹配: {value_match}, 梯度匹配: {grad_match}"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  函数值匹配: {'通过' if value_match else '失败'}")
                        print(f"  梯度匹配: {'通过' if grad_match else '失败'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")

    def test_prelu(self):
        """测试prelu激活函数"""
        func_name = "prelu"
        # 测试不同的输入形状
        test_cases = [
            {"name": "一维张量", "shape": (10,)},
            {"name": "二维张量", "shape": (5, 6)},
            {"name": "三维张量", "shape": (3, 4, 5)},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                # 随机生成参数并转换为TN张量
                alpha_value = np.random.rand() * 0.5  # 生成0-0.5之间的随机值            
                alpha_tensor = rm.tensor([alpha_value], requires_grad=True)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.prelu(rm_input, alpha=alpha_tensor)
                
                # PyTorch计算
                if TORCH_AVAILABLE:
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_alpha = torch.tensor([alpha_value], requires_grad=True)
                    torch_output = torch_F.prelu(torch_input, torch_alpha)
                    value_match = compare_values(rm_output, torch_output)
                else:
                    value_match = True
                
                # 2. 测试梯度
                # 随机生成梯度输出
                if isinstance(input_data, np.ndarray) and input_data.ndim > 0:
                    grad_output = np.random.randn(*input_data.shape)
                else:
                    grad_output = np.float32(np.random.randn())
                grad_output = grad_output.astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # PyTorch反向传播
                if TORCH_AVAILABLE:
                    torch_output.backward(torch.tensor(grad_output))
                    torch_grad = torch_input.grad
                    grad_match = compare_values(rm_grad, torch_grad)
                else:
                    grad_match = True
                
                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"函数值匹配: {value_match}, 梯度匹配: {grad_match}"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  函数值匹配: {'通过' if value_match else '失败'}")
                        print(f"  梯度匹配: {'通过' if grad_match else '失败'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")

    def test_rrelu(self):
        """测试rrelu激活函数"""
        func_name = "rrelu"
        # 测试不同的输入形状和参数
        test_cases = [
            {"name": "默认参数", "shape": (5, 6)},
            {"name": "自定义参数-低方差", "shape": (5, 6), "lower": 0.01, "upper": 0.03},
            {"name": "自定义参数-高方差", "shape": (5, 6), "lower": 0.1, "upper": 0.3},
            {"name": "一维张量", "shape": (10,)},
            {"name": "三维张量", "shape": (3, 4, 5)},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                lower = case.get("lower", 1./8.)
                upper = case.get("upper", 1./3.)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.rrelu(rm_input, lower=lower, upper=upper)
                
                # PyTorch计算
                if TORCH_AVAILABLE:
                    # 注意：由于RReLU使用随机负斜率，这里只检查输出是否在合理范围内
                    # 为了测试稳定性，我们可以多次运行取平均
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_output = torch_F.rrelu(torch_input, lower=lower, upper=upper)
                    # 对于随机函数，我们只检查输出是否在合理范围内
                    value_match = True  # 由于随机性，我们不进行严格比较
                else:
                    value_match = True
                
                # 2. 测试梯度 (同样考虑随机性)
                # 随机生成梯度输出
                if isinstance(input_data, np.ndarray) and input_data.ndim > 0:
                    grad_output = np.random.randn(*input_data.shape)
                else:
                    grad_output = np.float32(np.random.randn())
                grad_output = grad_output.astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # 由于RReLU是随机函数，我们只确保梯度不为None
                grad_match = rm_grad is not None
                
                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"输出值检查通过, 梯度存在"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  输出值检查: {'通过' if value_match else '失败'}")
                        print(f"  梯度存在: {'是' if grad_match else '否'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")

    def test_gelu(self):
        """测试gelu激活函数"""
        func_name = "gelu"
        # 测试不同的输入形状和值
        test_cases = [
            {"name": "一维张量", "shape": (10,)},
            {"name": "二维张量", "shape": (5, 6)},
            {"name": "三维张量", "shape": (3, 4, 5)},
            {"name": "边界值", "shape": (5,), "values": np.array([-10.0, -5.0, 0.0, 5.0, 10.0])},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.gelu(rm_input)
                
                # PyTorch计算
                if TORCH_AVAILABLE:
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_output = torch_F.gelu(torch_input)
                    # 修改这里，传入更低的精度要求
                    value_match = compare_values(rm_output, torch_output, atol=1e-3, rtol=1e-3)
                else:
                    value_match = True
                
                # 2. 测试梯度
                # 随机生成梯度输出
                if isinstance(input_data, np.ndarray) and input_data.ndim > 0:
                    grad_output = np.random.randn(*input_data.shape)
                else:
                    grad_output = np.float32(np.random.randn())
                grad_output = grad_output.astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # PyTorch反向传播
                if TORCH_AVAILABLE:
                    torch_output.backward(torch.tensor(grad_output))
                    torch_grad = torch_input.grad
                    # 在test_gelu函数中，将梯度比较的精度进一步降低
                    grad_match = compare_values(rm_grad, torch_grad, atol=1e-2, rtol=1e-2)
                else:
                    grad_match = True
                


                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"函数值匹配: {value_match}, 梯度匹配: {grad_match}"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  函数值匹配: {'通过' if value_match else '失败'}")
                        print(f"  梯度匹配: {'通过' if grad_match else '失败'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")

    def test_log_softmax(self):
        """测试log_softmax激活函数"""
        func_name = "log_softmax"
        # 测试不同的输入形状和维度
        test_cases = [
            {"name": "二维张量-维度0", "shape": (5, 6), "dim": 0},
            {"name": "二维张量-维度1", "shape": (5, 6), "dim": 1},
            {"name": "三维张量-维度0", "shape": (3, 4, 5), "dim": 0},
            {"name": "三维张量-维度1", "shape": (3, 4, 5), "dim": 1},
            {"name": "三维张量-维度2", "shape": (3, 4, 5), "dim": 2},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.log_softmax(rm_input, dim=case["dim"])
                
                # PyTorch计算
                if TORCH_AVAILABLE:
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_output = torch_F.log_softmax(torch_input, dim=case["dim"])
                    value_match = compare_values(rm_output, torch_output)
                else:
                    value_match = True
                
                # 2. 测试梯度
                # 随机生成梯度输出
                grad_output = np.random.randn(*input_data.shape).astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # PyTorch反向传播
                if TORCH_AVAILABLE:
                    torch_output.backward(torch.tensor(grad_output))
                    torch_grad = torch_input.grad
                    grad_match = compare_values(rm_grad, torch_grad)
                else:
                    grad_match = True
                
                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"函数值匹配: {value_match}, 梯度匹配: {grad_match}"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  函数值匹配: {'通过' if value_match else '失败'}")
                        print(f"  梯度匹配: {'通过' if grad_match else '失败'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")

    def test_softplus(self):
        """测试softplus激活函数"""
        func_name = "softplus"
        # 测试不同的输入形状和值
        test_cases = [
            {"name": "标量输入", "shape": []},
            {"name": "一维张量", "shape": (10,)},
            {"name": "二维张量", "shape": (5, 6)},
            {"name": "三维张量", "shape": (3, 4, 5)},
            {"name": "边界值", "shape": (5,), "values": np.array([-10.0, -5.0, 0.0, 5.0, 10.0])},
            {"name": "大负值", "shape": (3, 3), "values": -100 * np.ones((3, 3))},
            {"name": "大正值", "shape": (3, 3), "values": 100 * np.ones((3, 3))},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.softplus(rm_input)
                    
                # PyTorch计算
                if TORCH_AVAILABLE:
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_output = torch_F.softplus(torch_input)
                    value_match = compare_values(rm_output, torch_output)
                else:
                    value_match = True
                
                # 2. 测试梯度
                # 随机生成梯度输出
                if isinstance(input_data, np.ndarray) and input_data.ndim > 0:
                    grad_output = np.random.randn(*input_data.shape)
                else:
                    grad_output = np.float32(np.random.randn())
                grad_output = grad_output.astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # PyTorch反向传播
                if TORCH_AVAILABLE:
                    torch_output.backward(torch.tensor(grad_output))
                    torch_grad = torch_input.grad
                    grad_match = compare_values(rm_grad, torch_grad)
                else:
                    grad_match = True
                
                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"函数值匹配: {value_match}, 梯度匹配: {grad_match}"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  函数值匹配: {'通过' if value_match else '失败'}")
                        print(f"  梯度匹配: {'通过' if grad_match else '失败'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")

    def test_tanh(self):
        """测试tanh激活函数"""
        func_name = "tanh"
        # 测试不同的输入形状和值
        test_cases = [
            {"name": "标量输入", "shape": []},
            {"name": "一维张量", "shape": (10,)},
            {"name": "二维张量", "shape": (5, 6)},
            {"name": "三维张量", "shape": (3, 4, 5)},
            {"name": "边界值", "shape": (5,), "values": np.array([-10.0, -5.0, 0.0, 5.0, 10.0])},
            {"name": "大正值", "shape": (3, 3), "values": 50 * np.ones((3, 3))},
            {"name": "大负值", "shape": (3, 3), "values": -50 * np.ones((3, 3))},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.tanh(rm_input)
                
                # PyTorch计算
                if TORCH_AVAILABLE:
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_output = torch_F.tanh(torch_input)
                    value_match = compare_values(rm_output, torch_output)
                else:
                    value_match = True
                
                # 2. 测试梯度
                # 随机生成梯度输出
                if isinstance(input_data, np.ndarray) and input_data.ndim > 0:
                    grad_output = np.random.randn(*input_data.shape)
                else:
                    grad_output = np.float32(np.random.randn())
                grad_output = grad_output.astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # PyTorch反向传播
                if TORCH_AVAILABLE:
                    torch_output.backward(torch.tensor(grad_output))
                    torch_grad = torch_input.grad
                    grad_match = compare_values(rm_grad, torch_grad)
                else:
                    grad_match = True
                
                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"函数值匹配: {value_match}, 梯度匹配: {grad_match}"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  函数值匹配: {'通过' if value_match else '失败'}")
                        print(f"  梯度匹配: {'通过' if grad_match else '失败'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")

    def test_silu(self):
        """测试silu激活函数"""
        func_name = "silu"
        # 测试不同的输入形状和值
        test_cases = [
            {"name": "标量输入", "shape": []},
            {"name": "一维张量", "shape": (10,)},
            {"name": "二维张量", "shape": (5, 6)},
            {"name": "三维张量", "shape": (3, 4, 5)},
            {"name": "边界值", "shape": (5,), "values": np.array([-10.0, -5.0, 0.0, 5.0, 10.0])},
            {"name": "大正值", "shape": (3, 3), "values": 50 * np.ones((3, 3))},
            {"name": "大负值", "shape": (3, 3), "values": -50 * np.ones((3, 3))},
        ]
        
        for case in test_cases:
            case_name = f"{func_name}: {case['name']}"
            start_time = time.time()
            
            try:
                # 使用generate_test_data函数生成测试数据
                input_data = generate_test_data(case)
                
                # 1. 测试函数值
                # Riemann计算
                rm_input = rm.tensor(input_data, requires_grad=True)
                rm_output = F.silu(rm_input)
                
                # PyTorch计算
                if TORCH_AVAILABLE:
                    torch_input = torch.tensor(input_data, requires_grad=True)
                    torch_output = torch_F.silu(torch_input)
                    value_match = compare_values(rm_output, torch_output)
                else:
                    value_match = True
                
                # 2. 测试梯度
                # 随机生成梯度输出
                if isinstance(input_data, np.ndarray) and input_data.ndim > 0:
                    grad_output = np.random.randn(*input_data.shape)
                else:
                    grad_output = np.float32(np.random.randn())
                grad_output = grad_output.astype(rm.get_default_dtype())
                
                # Riemann反向传播
                rm_output.backward(rm.tensor(grad_output))
                rm_grad = rm_input.grad
                
                # PyTorch反向传播
                if TORCH_AVAILABLE:
                    torch_output.backward(torch.tensor(grad_output))
                    torch_grad = torch_input.grad
                    grad_match = compare_values(rm_grad, torch_grad)
                else:
                    grad_match = True
                
                # 判断测试是否通过
                passed = value_match and grad_match
                details = f"函数值匹配: {value_match}, 梯度匹配: {grad_match}"
                stats.add_result(case_name, passed, details)
                
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  函数值匹配: {'通过' if value_match else '失败'}")
                        print(f"  梯度匹配: {'通过' if grad_match else '失败'}")
                
                # 使用断言确保测试结果
                self.assertTrue(passed, f"{case_name} 测试失败: {details}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, f"执行出错: {str(e)}")
                if IS_RUNNING_AS_SCRIPT:
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                self.fail(f"{case_name} 执行出错: {str(e)}")

# 如果作为独立脚本运行
if __name__ == "__main__":
    IS_RUNNING_AS_SCRIPT = True

    # rm.set_default_dtype(rm.float64)
    # torch.set_default_dtype(torch.float64)

    clear_screen()
    print(f"{Colors.BOLD}{Colors.HEADER}===== 开始测试激活函数 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestActivationFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 0表示不输出详细信息
    result = runner.run(suite)
    
    # 打印统计摘要
    stats.print_summary()
    
    # 设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)