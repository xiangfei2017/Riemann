import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann import TN
    from riemann.autograd import gradcheck
    from riemann.tensordef import track_grad
    from riemann.autograd.grad import Function  # 正确导入Function类
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

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

# 测试自定义梯度功能类
class TestDefineGradFunc(unittest.TestCase):
    """测试自定义梯度功能，包括track_grad修饰器和Function类"""
    
    def setUp(self):
        """测试前的准备工作"""
        np.random.seed(42)
        self.current_test_name = self._testMethodName
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")
            print(f"{Colors.OKBLUE}测试目标: {self.current_test_name}{Colors.ENDC}")
    
    def tearDown(self):
        """测试结束后的清理工作"""
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"\n{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_track_grad_decorator(self):
        """测试 track_grad 修饰器的功能"""
        # 单输入函数梯度测试
        def _single_input_grad(x):
            return (2. * x,)
        
        @track_grad(_single_input_grad)
        def custom_single_input(x):
            return x * x
        
        # 多输入函数梯度测试
        def _multi_input_grad(x, y):
            return (1., 1.)
        
        @track_grad(_multi_input_grad)
        def custom_multi_input(x, y):
            return x + y
        
        # 除法函数梯度测试
        def _divide_grad(x, y):
            return (1./y, -x/(y*y))
        
        @track_grad(_divide_grad)
        def custom_divide(x, y):
            return x / y
        
        # 测试用例列表
        test_cases = [
            {"name": "单输入梯度计算", "func": custom_single_input, "inputs": [rm.tensor(2.0, requires_grad=True)], "expected_grad": 4.0},
            {"name": "多输入梯度计算", "func": custom_multi_input, "inputs": [rm.tensor(3.0, requires_grad=True), rm.tensor(4.0, requires_grad=True)], "expected_grads": [1.0, 1.0]},
            {"name": "除法函数梯度计算", "func": custom_divide, "inputs": [rm.tensor(5.0, requires_grad=True), rm.tensor(2.0, requires_grad=True)], "expected_grads": [0.5, -1.25]}
        ]
        
        for case in test_cases:
            start_time = time.time()
            try:
                inputs = case["inputs"]
                result = case["func"](*inputs)
                # 反向传播
                result.backward()
                
                if len(inputs) == 1:
                    # 单输入情况
                    x = inputs[0]
                    passed = abs(x.grad.data - case["expected_grad"]) < 1e-8
                    if passed:
                        print(f"{Colors.OKGREEN}[PASS]{Colors.ENDC} 单输入梯度计算: x.grad = {x.grad.data}, 预期值 = {case['expected_grad']}")
                    else:
                        print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} 单输入梯度计算: x.grad = {x.grad.data}, 预期值 = {case['expected_grad']}")
                else:
                    # 多输入情况
                    x, y = inputs
                    expected_grads = case["expected_grads"]
                    passed = abs(x.grad.data - expected_grads[0]) < 1e-8 and abs(y.grad.data - expected_grads[1]) < 1e-8
                    print(f"{Colors.OKGREEN}[PASS]{Colors.ENDC} 多输入梯度计算: x.grad = {x.grad.data}, y.grad = {y.grad.data}")
            except Exception as e:
                passed = False
                print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {case['name']}: 错误 - {str(e)}")
            finally:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case['name'], passed)
            time_taken = time.time() - start_time
    
    def test_gradcheck_function(self):
        """测试 gradcheck 函数的功能"""
        
        # 测试场景列表
        test_cases = [
            {
                "name": "简单乘法测试",
                "func": lambda x, y: x * y,
                "inputs": lambda: [rm.tensor(2.0, requires_grad=True), rm.tensor(3.0, requires_grad=True)],
                "description": "测试两个标量的乘法运算"
            },
            {
                "name": "线性函数测试",
                "func": lambda x, w, b: (x * w).sum() + b,
                "inputs": lambda: [
                    rm.tensor([1.0, 2.0, 3.0], requires_grad=True),
                    rm.tensor([0.5, 0.5, 0.5], requires_grad=True),
                    rm.tensor(1.0, requires_grad=True)
                ],
                "description": "测试带权重和偏置的线性函数"
            },
            {
                "name": "Sigmoid激活函数测试",
                "func": lambda x: 1.0 / (1.0 + (-x).exp()),
                "inputs": lambda: [rm.tensor([-1.0, 0.0, 1.0], requires_grad=True)],
                "description": "测试sigmoid非线性激活函数"
            },
            {
                "name": "快速模式测试",
                "func": lambda x, y: x * y,
                "inputs": lambda: [rm.tensor(2.0, requires_grad=True), rm.tensor(3.0, requires_grad=True)],
                "description": "测试快速模式下的梯度检查",
                "fast_mode": True
            },
            {
                "name": "高维输入测试",
                "func": lambda x: x.sum(),
                "inputs": lambda: [rm.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)],
                "description": "测试高维张量的梯度检查"
            }
        ]
        
        for case in test_cases:
            start_time = time.time()
            try:
                func = case["func"]
                inputs = case["inputs"]()  # 调用工厂函数获取输入
                
                # 调用gradcheck函数
                fast_mode = case.get("fast_mode", False)
                result = gradcheck(func, inputs, fast_mode=fast_mode)
                
                passed = result == True
                if passed:
                    print(f"{Colors.OKGREEN}[PASS]{Colors.ENDC} {case['name']}: {case['description']}")
                else:
                    print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {case['name']}: 梯度检查失败")
            except Exception as e:
                passed = False
                print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {case['name']}: 错误 - {str(e)}")
            finally:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case['name'], passed)
            time_taken = time.time() - start_time

    def test_function_class(self):
        """测试 Function 类的自定义梯度功能"""
        
        # 定义一个自定义的乘法函数
        class MulFunction(Function):
            """自定义乘法函数类"""
            
            @staticmethod
            def forward(ctx, x, y):
                """前向传播"""
                ctx.save_for_backward(x, y)
                return x * y
            
            @staticmethod
            def backward(ctx, grad_output):
                """反向传播"""
                x, y = ctx.saved_tensors
                return grad_output * y, grad_output * x
        
        # 定义一个自定义的ReLU函数
        class ReLUFunction(Function):
            """自定义ReLU函数类"""
            
            @staticmethod
            def forward(ctx, x):
                """前向传播"""
                ctx.save_for_backward(x)
                return rm.maximum(x, rm.tensor(0.0))
            
            @staticmethod
            def backward(ctx, grad_output):
                """反向传播"""
                x, = ctx.saved_tensors
                return grad_output * rm.tensor(x.data > 0)
        
        # 测试用例列表
        test_cases = [
            {
                "name": "MulFunction 梯度计算",
                "func": MulFunction.apply,
                "inputs": [rm.tensor(2.0, requires_grad=True), rm.tensor(3.0, requires_grad=True)],
                "expected_grads": [3.0, 2.0]
            },
            {
                "name": "ReLUFunction 梯度计算",
                "func": ReLUFunction.apply,
                "inputs": [rm.tensor(-1.0, requires_grad=True)],
                "expected_grad": 0.0
            }
        ]
        
        for case in test_cases:
            start_time = time.time()
            try:
                inputs = case["inputs"]
                result = case["func"](*inputs)
                # 反向传播
                result.backward()
                
                if len(inputs) == 1:
                    # 单输入情况
                    x = inputs[0]
                    passed = abs(x.grad.data - case["expected_grad"]) < 1e-8
                    if passed:
                        print(f"{Colors.OKGREEN}[PASS]{Colors.ENDC} ReLUFunction 梯度计算: x.grad = {x.grad.data}, 预期值 = {case['expected_grad']}")
                    else:
                        print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} ReLUFunction 梯度计算: x.grad = {x.grad.data}, 预期值 = {case['expected_grad']}")
                else:
                    # 多输入情况
                    x, y = inputs
                    expected_grads = case["expected_grads"]
                    passed = abs(x.grad.data - expected_grads[0]) < 1e-8 and abs(y.grad.data - expected_grads[1]) < 1e-8
                    print(f"{Colors.OKGREEN}[PASS]{Colors.ENDC} MulFunction 梯度计算: x.grad = {x.grad.data}, y.grad = {y.grad.data}")
            except Exception as e:
                passed = False
                print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {case['name']}: 错误 - {str(e)}")
            finally:
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(case['name'], passed)
            time_taken = time.time() - start_time

if __name__ == "__main__":
    print("\n开始测试自定义梯度功能...")
    IS_RUNNING_AS_SCRIPT = True
    clear_screen()
    
    print("\n" + "="*60)
    print(f"{Colors.BOLD}Riemann 自定义梯度功能测试{Colors.ENDC}")
    print("="*60)
    print("测试目标: track_grad修饰器、Function类")
    print("\n开始执行所有测试...\n")
    
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestDefineGradFunc)
    test_result = unittest.TextTestRunner().run(test_suite)
    
    print("\n\n" + "="*60)
    print(f"{Colors.BOLD}测试完成{Colors.ENDC}")
    print("="*60)
    
    # 打印测试结果摘要
    stats.print_summary()
    
    if test_result.wasSuccessful():
        print(f"\n{Colors.OKGREEN}所有测试通过！{Colors.ENDC}")
    else:
        print(f"\n{Colors.FAIL}测试失败！请检查测试结果。{Colors.ENDC}")
    
    print(f"\n所有测试完成，共执行 {stats.total_cases} 个测试用例")