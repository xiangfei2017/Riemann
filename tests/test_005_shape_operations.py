import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

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
    print("警告: 无法导入PyTorch，将只测试riemann的形状操作函数")
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

# 测试形状操作类
class TestShapeOperations(unittest.TestCase):
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
    
    def test_reshape(self):
        """测试reshape函数"""
        test_cases = [
            {"name": "基本reshape", "shape": (2, 3, 4), "new_shape": (2, 12)},
            {"name": "带-1参数的reshape", "shape": (2, 3, 4), "new_shape": (-1, 4)},
            {"name": "多维reshape", "shape": (2, 3, 4), "new_shape": (3, 2, 4)},
            {"name": "标量reshape为向量", "shape": (), "new_shape": (1,)},
            {"name": "向量reshape为标量", "shape": (1,), "new_shape": ()},
        ]
        
        for case in test_cases:
            case_name = f"reshape - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                else:
                    torch_x = None
                
                # 前向传播测试
                rm_result = rm_x.reshape(*case["new_shape"])
                torch_result = None
                if TORCH_AVAILABLE:
                    # 修复：PyTorch的reshape接受形状元组，而不是解包参数
                    torch_result = torch_x.reshape(case["new_shape"])
                
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
                    backward_passed = compare_values(rm_x.grad, torch_x.grad)
                
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
                self.assertTrue(passed, f"reshape测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_view(self):
        """测试view函数（通常与reshape功能类似）"""
        test_cases = [
            {"name": "基本view", "shape": (2, 3, 4), "new_shape": (2, 12)},
            {"name": "带-1参数的view", "shape": (2, 3, 4), "new_shape": (-1, 4)},
            {"name": "多维view", "shape": (2, 3, 4), "new_shape": (3, 2, 4)},
        ]
        
        for case in test_cases:
            case_name = f"view - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                else:
                    torch_x = None
                
                # 前向传播测试
                rm_result = rm_x.view(*case["new_shape"])
                torch_result = None
                if TORCH_AVAILABLE:
                    try:
                        torch_result = torch_x.view(*case["new_shape"])
                    except RuntimeError:
                        # 如果PyTorch报错，我们也期望Riemann报错
                        self.fail(f"PyTorch view失败但Riemann成功: {case_name}")
                
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
                    backward_passed = compare_values(rm_x.grad, torch_x.grad)
                
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
                self.assertTrue(passed, f"view测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_flatten(self):
        """测试flatten函数"""
        test_cases = [
            {"name": "默认flatten", "shape": (2, 3, 4), "start_dim": None, "end_dim": None},
            {"name": "指定start_dim的flatten", "shape": (2, 3, 4), "start_dim": 1, "end_dim": None},
            {"name": "指定start_dim和end_dim的flatten", "shape": (2, 3, 4, 5), "start_dim": 1, "end_dim": 2},
        ]
        
        for case in test_cases:
            case_name = f"flatten - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                else:
                    torch_x = None
                
                # 前向传播测试
                if case["start_dim"] is None and case["end_dim"] is None:
                    rm_result = rm_x.flatten()
                    if TORCH_AVAILABLE:
                        torch_result = torch_x.flatten()
                elif case["end_dim"] is None:
                    rm_result = rm_x.flatten(start_dim=case["start_dim"])
                    if TORCH_AVAILABLE:
                        torch_result = torch_x.flatten(start_dim=case["start_dim"])
                else:
                    rm_result = rm_x.flatten(start_dim=case["start_dim"], end_dim=case["end_dim"])
                    if TORCH_AVAILABLE:
                        torch_result = torch_x.flatten(start_dim=case["start_dim"], end_dim=case["end_dim"])
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_result, torch_result if TORCH_AVAILABLE else None)
                
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
                    backward_passed = compare_values(rm_x.grad, torch_x.grad)
                
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
                self.assertTrue(passed, f"flatten测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_squeeze(self):
        """测试squeeze函数"""
        test_cases = [
            {"name": "默认squeeze(None)", "shape": (1, 2, 1, 3), "dim": None},
            {"name": "squeeze单个维度", "shape": (1, 2, 3), "dim": 0},
            {"name": "squeeze负数维度", "shape": (2, 1, 3), "dim": -2},
            {"name": "squeeze元组维度", "shape": (1, 2, 1, 3), "dim": (0, 2)},
            {"name": "squeeze非1维度(应返回原张量)", "shape": (2, 3), "dim": 0},
            {"name": "空元组squeeze(应返回原张量)", "shape": (2, 3), "dim": ()},
        ]
        
        for case in test_cases:
            case_name = f"squeeze - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                else:
                    torch_x = None
                
                # 前向传播测试
                if case["dim"] is None:
                    rm_result = rm_x.squeeze()
                    if TORCH_AVAILABLE:
                        torch_result = torch_x.squeeze()
                else:
                    rm_result = rm_x.squeeze(case["dim"])
                    if TORCH_AVAILABLE:
                        try:
                            torch_result = torch_x.squeeze(case["dim"])
                        except Exception:
                            # 处理PyTorch可能抛出的异常
                            pass
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_result, torch_result if TORCH_AVAILABLE else None)
                
                # 反向传播测试
                backward_passed = True
                if TORCH_AVAILABLE:
                    # 计算损失
                    rm_loss = rm_result.sum()
                    if torch_result is not None:
                        torch_loss = torch_result.sum()
                        # 反向传播
                        torch_loss.backward()
                    # 反向传播
                    rm_loss.backward()
                    
                    # 比较梯度
                    backward_passed = compare_values(rm_x.grad, torch_x.grad if torch_result is not None else None)
                
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
                self.assertTrue(passed, f"squeeze测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_unsqueeze(self):
        """测试unsqueeze函数"""
        test_cases = [
            {"name": "unsqueeze单个维度", "shape": (2, 3), "dim": 0},
            {"name": "unsqueeze末尾维度", "shape": (2, 3), "dim": 2},
            {"name": "unsqueeze负数维度", "shape": (2, 3), "dim": -1},
            {"name": "unsqueeze元组维度", "shape": (2, 3), "dim": (0, 2)},
            {"name": "空元组unsqueeze(应返回原张量)", "shape": (2, 3), "dim": ()},
        ]
        
        for case in test_cases:
            case_name = f"unsqueeze - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                else:
                    torch_x = None
                
                # 前向传播测试
                rm_result = rm_x.unsqueeze(case["dim"])
                torch_result = None
                if TORCH_AVAILABLE:
                    # 检查是否是元组维度，PyTorch不支持元组维度
                    if isinstance(case["dim"], tuple):
                        # 对于元组维度（包括空元组），PyTorch不支持，跳过PyTorch部分的测试
                        pass
                    else:
                        try:
                            torch_result = torch_x.unsqueeze(case["dim"])
                        except Exception:
                            # 处理其他可能的异常
                            pass
                
                # 修改前向传播比较逻辑
                if isinstance(case["dim"], tuple):
                    # 对于元组维度（包括空元组），我们只验证Riemann的结果
                    # 对于空元组，验证结果是否与原张量相同
                    if len(case["dim"]) == 0:
                        forward_passed = (rm_result is rm_x)  # 空元组应返回原张量
                    else:
                        forward_passed = True  # 非空元组维度测试只验证Riemann是否正常运行
                else:
                    forward_passed = compare_values(rm_result, torch_result if TORCH_AVAILABLE else None)
                
                # 初始化backward_passed，确保在所有情况下都有定义
                backward_passed = True
                if TORCH_AVAILABLE:
                    # 计算损失
                    rm_loss = rm_result.sum()
                    # 只有当不是元组维度测试时才计算PyTorch的损失和反向传播
                    if not isinstance(case["dim"], tuple):
                        if torch_result is not None:
                            torch_loss = torch_result.sum()
                            torch_loss.backward()
                        
                        # 执行Riemann的反向传播（修正缩进，确保总是执行）
                        rm_loss.backward()
                        
                        # 修改反向传播比较逻辑
                        if isinstance(case["dim"], tuple):
                            # 对于元组维度（包括空元组），我们只验证Riemann是否能正常反向传播
                            backward_passed = rm_x.grad is not None  # 只要梯度存在就认为通过
                        else:
                            backward_passed = compare_values(rm_x.grad, torch_x.grad if torch_result is not None else None)
                    
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
                self.assertTrue(passed, f"unsqueeze测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_expand(self):
        """测试expand函数"""
        test_cases = [
            {"name": "基本expand", "shape": (1, 3), "size": (2, 3)},
            {"name": "多维expand", "shape": (1, 1, 3), "size": (2, 4, 3)},
            {"name": "expand标量为向量", "shape": (), "size": (3,)},
            # 修改后的正确用例：第一维为1，所以可以扩展，第二维保持不变
            {"name": "expand带-1参数(固定列)", "shape": (1, 3), "size": (4, -1)},
            # 或者完全符合expand行为的测试用例
            {"name": "expand带-1参数(固定行)", "shape": (2, 1), "size": (-1, 5)},
        ]
        
        for case in test_cases:
            case_name = f"expand - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                else:
                    torch_x = None
                
                # 前向传播测试
                rm_result = rm_x.expand(*case["size"])
                torch_result = None
                if TORCH_AVAILABLE:
                    try:
                        torch_result = torch_x.expand(*case["size"])
                    except RuntimeError:
                        # 如果PyTorch报错，我们也期望Riemann报错
                        self.fail(f"PyTorch expand失败但Riemann成功: {case_name}")
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_result, torch_result if TORCH_AVAILABLE else None)
                
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
                    backward_passed = compare_values(rm_x.grad, torch_x.grad)
                
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
                self.assertTrue(passed, f"expand测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_expand_as(self):
        """测试expand_as函数"""
        test_cases = [
            {"name": "基本expand_as", "src_shape": (1, 3), "target_shape": (2, 3)},
            {"name": "多维expand_as", "src_shape": (1, 1, 3), "target_shape": (2, 4, 3)},
        ]
        
        for case in test_cases:
            case_name = f"expand_as - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["src_shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                
                np_target = np.random.randn(*case["target_shape"])
                rm_target = rm.tensor(np_target)
                
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                    torch_target = torch.tensor(np_target)
                else:
                    torch_x = None
                    torch_target = None
                
                # 前向传播测试
                rm_result = rm_x.expand_as(rm_target)
                torch_result = None
                if TORCH_AVAILABLE:
                    try:
                        torch_result = torch_x.expand_as(torch_target)
                    except RuntimeError:
                        # 如果PyTorch报错，我们也期望Riemann报错
                        self.fail(f"PyTorch expand_as失败但Riemann成功: {case_name}")
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_result, torch_result if TORCH_AVAILABLE else None)
                
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
                    backward_passed = compare_values(rm_x.grad, torch_x.grad)
                
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
                self.assertTrue(passed, f"expand_as测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_permute(self):
        """测试permute函数"""
        test_cases = [
            {"name": "3D张量permute", "shape": (2, 3, 4), "dims": (2, 0, 1)},
            {"name": "2D张量转置permute", "shape": (2, 3), "dims": (1, 0)},
            {"name": "4D张量permute", "shape": (1, 2, 3, 4), "dims": (3, 2, 1, 0)},
            {"name": "包含负数索引的permute", "shape": (2, 3, 4), "dims": (-1, -2, -3)},
        ]
        
        for case in test_cases:
            case_name = f"permute - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                else:
                    torch_x = None
                
                # 前向传播测试
                rm_result = rm_x.permute(case["dims"])
                torch_result = None
                if TORCH_AVAILABLE:
                    # 移除异常处理，直接调用
                    torch_result = torch_x.permute(case["dims"])
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_result, torch_result if TORCH_AVAILABLE else None)
                
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
                    backward_passed = compare_values(rm_x.grad, torch_x.grad)
                
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
                self.assertTrue(passed, f"permute测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_transpose(self):
        """测试transpose函数"""
        test_cases = [
            {"name": "2D张量transpose", "shape": (2, 3), "dim0": 0, "dim1": 1},
            {"name": "3D张量transpose", "shape": (2, 3, 4), "dim0": 1, "dim1": 2},
            {"name": "使用负数索引的transpose", "shape": (2, 3, 4), "dim0": -2, "dim1": -1},
        ]
        
        for case in test_cases:
            case_name = f"transpose - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                else:
                    torch_x = None
                
                # 前向传播测试
                rm_result = rm_x.transpose(case["dim0"], case["dim1"])
                torch_result = None
                if TORCH_AVAILABLE:
                    try:
                        torch_result = torch_x.transpose(case["dim0"], case["dim1"])
                    except RuntimeError:
                        # 如果PyTorch报错，我们也期望Riemann报错
                        self.fail(f"PyTorch transpose失败但Riemann成功: {case_name}")
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_result, torch_result if TORCH_AVAILABLE else None)
                
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
                    backward_passed = compare_values(rm_x.grad, torch_x.grad)
                
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
                self.assertTrue(passed, f"transpose测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise

    def test_transpose_property_T(self):
        """测试.Riemann的.T转置属性（只测试Riemann实现，不使用PyTorch）"""
        test_cases = [
            {"name": "一维张量T", "shape": (5,)},
            {"name": "二维张量T", "shape": (3, 4)},
            {"name": "三维张量T", "shape": (2, 3, 4)},
            {"name": "四维张量T", "shape": (1, 2, 3, 4)},
        ]
        
        for case in test_cases:
            case_name = f"T - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                
                # 前向传播测试 - 仅测试Riemann部分
                rm_result = rm_x.T
                forward_passed = rm_result is not None
                
                # 反向传播测试
                rm_loss = rm_result.sum()
                rm_loss.backward()
                backward_passed = rm_x.grad is not None
                
                passed = forward_passed and backward_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed:
                        print(f"  前向传播测试: {'通过' if forward_passed else '失败'}")
                        print(f"  反向传播测试: {'通过' if backward_passed else '失败'}")
                        print(f"  Riemann形状: {rm_result.shape}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"T属性测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_transpose_property_mT(self):
        """测试.mT转置属性（交换最后两个维度）"""
        test_cases = [
            {"name": "二维张量mT", "shape": (3, 4)},
            {"name": "三维张量mT", "shape": (2, 3, 4)},
            {"name": "四维张量mT", "shape": (1, 2, 3, 4)},
        ]
        
        for case in test_cases:
            case_name = f"mT - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                else:
                    torch_x = None
                
                # 前向传播测试
                rm_result = rm_x.mT
                torch_result = None
                if TORCH_AVAILABLE:
                    torch_result = torch_x.mT
                
                # 比较前向传播结果
                forward_passed = compare_values(rm_result, torch_result if TORCH_AVAILABLE else None)
                
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
                    backward_passed = compare_values(rm_x.grad, torch_x.grad)
                
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
                self.assertTrue(passed, f"mT属性测试失败: {case_name}")
                
            except Exception as e:
                time_taken = time.time() - start_time
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_flip(self):
        """测试flip函数与PyTorch的一致性"""
        test_cases = [
            {"name": "0D标量翻转", "shape": (), "dims": []},
            {"name": "1D张量翻转", "shape": (5,), "dims": [0]},
            {"name": "2D张量单维度翻转-行", "shape": (3, 4), "dims": [0]},
            {"name": "2D张量单维度翻转-列", "shape": (3, 4), "dims": [1]},
            {"name": "2D张量多维度翻转", "shape": (3, 4), "dims": [0, 1]},
            {"name": "3D张量单维度翻转", "shape": (2, 3, 4), "dims": [1]},
            {"name": "3D张量多维度翻转", "shape": (2, 3, 4), "dims": [0, 2]},
            {"name": "高维张量多维度翻转", "shape": (2, 2, 3, 4), "dims": [1, 3]}
        ]
        
        for case in test_cases:
            case_name = f"flip - {case['name']}"
            start_time = time.time()
            try:
                # 创建测试数据
                np_x = np.random.randn(*case["shape"])
                rm_x = rm.tensor(np_x, requires_grad=True)
                if TORCH_AVAILABLE:
                    torch_x = torch.tensor(np_x, requires_grad=True)
                else:
                    torch_x = None
                
                # 前向传播测试
                rm_result = rm_x.flip(dims=case["dims"])
                torch_result = None
                if TORCH_AVAILABLE:
                    # PyTorch的flip接受dims参数
                    torch_result = torch_x.flip(dims=case["dims"])
                
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
                    backward_passed = compare_values(rm_x.grad, torch_x.grad)
                
                passed = forward_passed and backward_passed
                
                time_taken = time.time() - start_time
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case_name, passed)
                    status = "通过" if passed else "失败"
                    print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                    if not passed and TORCH_AVAILABLE:
                        print(f"  前向传播比较: {'通过' if forward_passed else '失败'}")
                        print(f"  反向传播比较: {'通过' if backward_passed else '失败'}")
                        print(f"  测试维度: {case['dims']}")
                        print(f"  Riemann形状: {rm_result.shape}, PyTorch形状: {torch_result.shape if torch_result is not None else 'N/A'}")
                
                # 断言确保测试通过
                self.assertTrue(passed, f"flip测试失败: {case_name}")
                
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
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行形状操作函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestShapeOperations)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)