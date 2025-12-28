import numpy as np
import unittest
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.tensordef import tensor
    from riemann.autograd.grad import grad
    IMPORT_SUCCESS = True
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    IMPORT_SUCCESS = False

# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的scatter函数")
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
    rm_data = rm_result.data if hasattr(rm_result, 'data') else rm_result
    torch_data = torch_result.cpu().detach().numpy() if hasattr(torch_result, 'cpu') else torch_result
    
    # 处理形状不匹配的情况
    try:
        # 增加容差参数
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

class TestScatterFunctions(unittest.TestCase):
    def setUp(self):
        # 如果导入失败，跳过所有测试
        if not IMPORT_SUCCESS:
            self.skipTest("无法导入riemann模块")
            
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
        if IS_RUNNING_AS_SCRIPT and IMPORT_SUCCESS:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_scatter_src(self):
        """测试scatter函数（使用src参数）"""
        case_name = "scatter函数测试（src参数）"
        start_time = time.time()
        try:
            # 测试不同维度的scatter
            dimensions = [(3, 4), (2, 3, 4), (1, 5, 3)]  # 1D, 2D, 3D
            scatter_dims = [0, 1, 0]  # 不同维度的索引
            
            all_passed = True
            for i, (shape, dim) in enumerate(zip(dimensions, scatter_dims)):
                # 创建测试数据
                np_tensor = np.random.randn(*shape)
                np_index = np.random.randint(0, shape[dim], size=shape)
                np_src = np.random.randn(*shape)
                
                # Riemann实现
                rm_tensor = tensor(np_tensor, requires_grad=True)
                rm_index = tensor(np_index)
                rm_src = tensor(np_src, requires_grad=True)
                rm_result = rm_tensor.scatter(dim, rm_index, rm_src)
                
                # PyTorch实现
                torch_result = None
                if TORCH_AVAILABLE:
                    torch_tensor = torch.tensor(np_tensor, requires_grad=True)
                    torch_index = torch.tensor(np_index)
                    torch_src = torch.tensor(np_src, requires_grad=True)
                    torch_result = torch_tensor.scatter(dim, torch_index, torch_src)
                
                # 比较函数值
                value_passed = compare_values(rm_result, torch_result)
                
                # 测试梯度
                grad_passed = True
                if TORCH_AVAILABLE:
                    # Riemann梯度
                    rm_loss = rm_result.sum()
                    rm_loss.backward()
                    rm_tensor_grad = rm_tensor.grad
                    rm_src_grad = rm_src.grad
                    
                    # PyTorch梯度
                    torch_loss = torch_result.sum()
                    torch_loss.backward()
                    torch_tensor_grad = torch_tensor.grad
                    torch_src_grad = torch_src.grad
                    
                    # 比较梯度
                    tensor_grad_passed = compare_values(rm_tensor_grad, torch_tensor_grad)
                    src_grad_passed = compare_values(rm_src_grad, torch_src_grad)
                    grad_passed = tensor_grad_passed and src_grad_passed
                
                subcase_name = f"子用例 {i+1}: 维度={dim}, 形状={shape}"
                subcase_passed = value_passed and grad_passed
                all_passed = all_passed and subcase_passed
                
                # 为每个子用例单独添加统计
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(subcase_name, subcase_passed)
                    status = "通过" if subcase_passed else "失败"
                    status_color = Colors.OKGREEN if subcase_passed else Colors.FAIL
                    print(f"  子用例 {i+1}: {subcase_name.split(':')[1]} - {status_color}{status}{Colors.ENDC}")
                    if not value_passed:
                        print(f"    - 函数值比较失败")
                    if TORCH_AVAILABLE and not grad_passed:
                        print(f"    - 梯度比较失败")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                status = "通过" if all_passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if all_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(all_passed, f"scatter函数测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    # 修改test_scatter_value函数中的梯度计算部分
    def test_scatter_value(self):
        """测试scatter函数（使用value参数）"""
        case_name = "scatter函数测试（value参数）"
        start_time = time.time()
        try:
            # 测试不同维度的scatter
            dimensions = [(3, 4), (2, 3, 4)]
            scatter_dims = [1, 2]
            values = [2.0, -3.5]  # 不同的标量值
            
            all_passed = True
            for i, (shape, dim, value) in enumerate(zip(dimensions, scatter_dims, values)):
                # 创建测试数据 - 统一使用float32类型
                np_tensor = np.random.randn(*shape).astype(np.float32)
                np_index = np.random.randint(0, shape[dim], size=shape)
                
                # Riemann实现
                rm_tensor = tensor(np_tensor, requires_grad=True)
                rm_index = tensor(np_index)
                rm_result = rm_tensor.scatter(dim, rm_index, value=value)
                
                # PyTorch实现
                torch_result = None
                if TORCH_AVAILABLE:
                    torch_tensor = torch.tensor(np_tensor, requires_grad=True, dtype=torch.float32)
                    torch_index = torch.tensor(np_index)
                    torch_result = torch_tensor.scatter(dim, torch_index, value=value)
                
                # 比较函数值
                value_passed = compare_values(rm_result, torch_result)
                
                # 测试梯度 - 统一使用backward方法
                grad_passed = True
                if TORCH_AVAILABLE:
                    # Riemann梯度 - 使用backward方法
                    rm_loss = rm_result.sum()
                    rm_loss.backward()
                    rm_tensor_grad = rm_tensor.grad
                    
                    # PyTorch梯度
                    torch_loss = torch_result.sum()
                    torch_loss.backward()
                    torch_tensor_grad = torch_tensor.grad
                    
                    # 比较梯度
                    grad_passed = compare_values(rm_tensor_grad, torch_tensor_grad)
                
                subcase_name = f"子用例 {i+1}: 维度={dim}, 形状={shape}, 值={value}"
                subcase_passed = value_passed and grad_passed
                all_passed = all_passed and subcase_passed
                
                # 为每个子用例单独添加统计
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(subcase_name, subcase_passed)
                    status = "通过" if subcase_passed else "失败"
                    status_color = Colors.OKGREEN if subcase_passed else Colors.FAIL
                    print(f"  子用例 {i+1}: {subcase_name.split(':')[1]} - {status_color}{status}{Colors.ENDC}")
                    if not value_passed:
                        print(f"    - 函数值比较失败")
                    if TORCH_AVAILABLE and not grad_passed:
                        print(f"    - 梯度比较失败")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                status = "通过" if all_passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if all_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(all_passed, f"scatter函数（value参数）测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    # 修改test_scatter_函数中的梯度计算部分
    def test_scatter_(self):
        """测试scatter_函数（原地操作）"""
        case_name = "scatter_函数测试（原地操作）"
        start_time = time.time()
        try:
            # 测试不同维度的scatter_
            dimensions = [(4, 5), (2, 4, 3)]
            scatter_dims = [0, 2]
            
            all_passed = True
            for i, (shape, dim) in enumerate(zip(dimensions, scatter_dims)):
                # 创建测试数据 - 统一使用float32类型
                np_tensor = np.random.randn(*shape).astype(np.float32)
                np_index = np.random.randint(0, shape[dim], size=shape)
                np_src = np.random.randn(*shape).astype(np.float32)
                
                # Riemann实现
                rm_tensor = tensor(np_tensor, requires_grad=False)
                rm_index = tensor(np_index)
                rm_src = tensor(np_src, requires_grad=True)
                rm_result = rm_tensor.clone().scatter_(dim, rm_index, rm_src)
                
                # PyTorch实现
                if TORCH_AVAILABLE:
                    torch_tensor = torch.tensor(np_tensor, requires_grad=False, dtype=torch.float32)
                    torch_index = torch.tensor(np_index)
                    torch_src = torch.tensor(np_src, requires_grad=True, dtype=torch.float32)
                    torch_result = torch_tensor.clone().scatter_(dim, torch_index, torch_src)
                
                # 比较函数值
                value_passed = compare_values(rm_result, torch_result)
                
                # 测试src的梯度 - 统一使用backward方法
                grad_passed = True
                if TORCH_AVAILABLE:
                    # Riemann梯度 - 使用backward方法
                    rm_loss = rm_result.sum()
                    rm_loss.backward()
                    rm_src_grad = rm_src.grad
                    
                    # PyTorch梯度
                    torch_loss = torch_result.sum()
                    torch_loss.backward()
                    torch_src_grad = torch_src.grad
                    
                    # 比较梯度
                    grad_passed = compare_values(rm_src_grad, torch_src_grad)
                
                subcase_name = f"子用例 {i+1}: 维度={dim}, 形状={shape}"
                subcase_passed = value_passed and grad_passed
                all_passed = all_passed and subcase_passed
                
                # 为每个子用例单独添加统计
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(subcase_name, subcase_passed)
                    status = "通过" if subcase_passed else "失败"
                    status_color = Colors.OKGREEN if subcase_passed else Colors.FAIL
                    print(f"  子用例 {i+1}: {subcase_name.split(':')[1]} - {status_color}{status}{Colors.ENDC}")
                    if not value_passed:
                        print(f"    - 函数值比较失败")
                    if TORCH_AVAILABLE and not grad_passed:
                        print(f"    - 梯度比较失败")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                status = "通过" if all_passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if all_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(all_passed, f"scatter_函数测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_scatter_add_(self):
        """测试scatter_add_函数（原地累加）"""
        case_name = "scatter_add_函数测试（原地累加）"
        start_time = time.time()
        try:
            # 测试不同维度的scatter_add_
            dimensions = [(3, 4), (2, 3, 4), (1, 5, 3)]
            scatter_dims = [1, 0, 2]
            
            all_passed = True
            for i, (shape, dim) in enumerate(zip(dimensions, scatter_dims)):
                # 创建测试数据 - 统一使用float32类型
                np_tensor = np.random.randn(*shape).astype(np.float32)
                np_index = np.random.randint(0, shape[dim], size=shape)
                np_src = np.random.randn(*shape).astype(np.float32)
                
                # Riemann实现 - 设置self张量requires_grad=True以测试其梯度
                rm_tensor = tensor(np_tensor, requires_grad=True)
                rm_index = tensor(np_index)
                rm_src = tensor(np_src, requires_grad=True)
                rm_result = rm_tensor.clone().scatter_add_(dim, rm_index, rm_src)
                
                # PyTorch实现 - 设置self张量requires_grad=True以测试其梯度
                if TORCH_AVAILABLE:
                    torch_tensor = torch.tensor(np_tensor, requires_grad=True, dtype=torch.float32)
                    torch_index = torch.tensor(np_index)
                    torch_src = torch.tensor(np_src, requires_grad=True, dtype=torch.float32)
                    torch_result = torch_tensor.clone().scatter_add_(dim, torch_index, torch_src)
                
                # 比较函数值
                value_passed = compare_values(rm_result, torch_result)
                
                # 测试self和src的梯度 - 统一使用backward方法
                grad_passed = True
                if TORCH_AVAILABLE:
                    # Riemann梯度 - 使用backward方法
                    rm_loss = rm_result.sum()
                    rm_loss.backward()
                    rm_tensor_grad = rm_tensor.grad
                    rm_src_grad = rm_src.grad
                    
                    # PyTorch梯度
                    torch_loss = torch_result.sum()
                    torch_loss.backward()
                    torch_tensor_grad = torch_tensor.grad
                    torch_src_grad = torch_src.grad
                    
                    # 比较梯度 - 同时比较self和src的梯度
                    tensor_grad_passed = compare_values(rm_tensor_grad, torch_tensor_grad)
                    src_grad_passed = compare_values(rm_src_grad, torch_src_grad)
                    grad_passed = tensor_grad_passed and src_grad_passed
                
                # 为每个子用例单独添加统计
                subcase_name = f"子用例 {i+1}: 维度={dim}, 形状={shape}"
                subcase_passed = value_passed and grad_passed
                all_passed = all_passed and subcase_passed
                
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(subcase_name, subcase_passed)
                    status = "通过" if subcase_passed else "失败"
                    status_color = Colors.OKGREEN if subcase_passed else Colors.FAIL
                    print(f"  子用例 {i+1}: {subcase_name.split(':')[1]} - {status_color}{status}{Colors.ENDC}")
                    if not value_passed:
                        print(f"    - 函数值比较失败")
                    if TORCH_AVAILABLE and not grad_passed:
                        print(f"    - 梯度比较失败")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                status = "通过" if all_passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if all_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(all_passed, f"scatter_add_函数测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise

    def test_scatter_add(self):
        """测试scatter_add函数（非原地累加操作）"""
        case_name = "scatter_add函数测试（非原地累加操作）"
        start_time = time.time()
        try:
            # 测试不同维度的scatter_add
            dimensions = [(3, 4), (2, 3, 4), (1, 5, 3)]
            scatter_dims = [1, 0, 2]
            
            all_passed = True
            for i, (shape, dim) in enumerate(zip(dimensions, scatter_dims)):
                # 创建测试数据 - 统一使用float32类型
                np_tensor = np.random.randn(*shape).astype(np.float32)
                np_index = np.random.randint(0, shape[dim], size=shape)
                np_src = np.random.randn(*shape).astype(np.float32)
                
                # Riemann实现 - 设置self张量requires_grad=True以测试其梯度
                rm_tensor = tensor(np_tensor, requires_grad=True)
                rm_index = tensor(np_index)
                rm_src = tensor(np_src, requires_grad=True)
                # 保存原始张量的副本，用于验证非原地操作
                rm_tensor_original = rm_tensor.clone()
                # 调用scatter_add函数（非原地操作）
                rm_result = rm_tensor.scatter_add(dim, rm_index, rm_src)
                
                # 验证非原地操作 - 原始张量应该保持不变
                is_non_inplace = compare_values(rm_tensor, rm_tensor_original)
                
                # PyTorch实现 - 由于PyTorch没有直接的非原地scatter_add，我们可以使用clone和scatter_add_
                if TORCH_AVAILABLE:
                    torch_tensor = torch.tensor(np_tensor, requires_grad=True, dtype=torch.float32)
                    torch_index = torch.tensor(np_index)
                    torch_src = torch.tensor(np_src, requires_grad=True, dtype=torch.float32)
                    # 模拟非原地操作
                    torch_result = torch_tensor.clone().scatter_add_(dim, torch_index, torch_src)
                
                # 比较函数值
                value_passed = compare_values(rm_result, torch_result)
                
                # 测试self和src的梯度 - 统一使用backward方法
                grad_passed = True
                if TORCH_AVAILABLE:
                    # Riemann梯度 - 使用backward方法
                    rm_loss = rm_result.sum()
                    rm_loss.backward()
                    rm_tensor_grad = rm_tensor.grad
                    rm_src_grad = rm_src.grad
                    
                    # PyTorch梯度
                    torch_loss = torch_result.sum()
                    torch_loss.backward()
                    torch_tensor_grad = torch_tensor.grad
                    torch_src_grad = torch_src.grad
                    
                    # 比较梯度 - 同时比较self和src的梯度
                    tensor_grad_passed = compare_values(rm_tensor_grad, torch_tensor_grad)
                    src_grad_passed = compare_values(rm_src_grad, torch_src_grad)
                    grad_passed = tensor_grad_passed and src_grad_passed
                
                subcase_name = f"子用例 {i+1}: 维度={dim}, 形状={shape}"
                subcase_passed = value_passed and grad_passed and is_non_inplace
                all_passed = all_passed and subcase_passed
                
                # 为每个子用例单独添加统计
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(subcase_name, subcase_passed)
                    status = "通过" if subcase_passed else "失败"
                    status_color = Colors.OKGREEN if subcase_passed else Colors.FAIL
                    print(f"  子用例 {i+1}: {subcase_name.split(':')[1]} - {status_color}{status}{Colors.ENDC}")
                    if not value_passed:
                        print(f"    - 函数值比较失败")
                    if not is_non_inplace:
                        print(f"    - 非原地操作验证失败")
                    if TORCH_AVAILABLE and not grad_passed:
                        print(f"    - 梯度比较失败")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                status = "通过" if all_passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if all_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(all_passed, f"scatter_add函数测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
            
    def test_scatter_add_scalar_src(self):
        """测试scatter_add函数（基于标量值创建匹配形状的张量src）"""
        case_name = "scatter_add函数测试（基于标量值创建匹配形状的张量src）"
        start_time = time.time()
        try:
            # 测试不同维度的scatter_add
            dimensions = [(3, 4), (2, 3)]
            scatter_dims = [0, 1]
            scalar_values = [1.5, -2.0]
            
            all_passed = True
            for i, (shape, dim, scalar) in enumerate(zip(dimensions, scatter_dims, scalar_values)):
                # 创建测试数据 - 统一使用float32类型
                np_tensor = np.random.randn(*shape).astype(np.float32)
                np_index = np.random.randint(0, shape[dim], size=shape)
                
                # Riemann实现 - 设置self张量requires_grad=True以测试其梯度
                rm_tensor = tensor(np_tensor, requires_grad=True)
                rm_index = tensor(np_index)
                # 保存原始张量的副本，用于验证非原地操作
                rm_tensor_original = rm_tensor.clone()
                # riemann scatter_add支持src参数是标量，会自动创建匹配形状的张量
                rm_result = rm_tensor.scatter_add(dim, rm_index, scalar)
                
                # 验证非原地操作 - 原始张量应该保持不变
                is_non_inplace = compare_values(rm_tensor, rm_tensor_original)
                
                # PyTorch实现 - 由于PyTorch没有直接的非原地scatter_add，我们可以使用clone和scatter_add_
                if TORCH_AVAILABLE:
                    torch_tensor = torch.tensor(np_tensor, requires_grad=True, dtype=torch.float32)
                    torch_index = torch.tensor(np_index)
                    # 关键修复：手动将标量转换为与index形状相同的张量，torch scatter_add_不支持src参数为标量
                    torch_src = torch.tensor(np.full(shape, scalar, dtype=np.float32), requires_grad=False)
                    torch_result = torch_tensor.clone().scatter_add_(dim, torch_index, torch_src)
                
                # 比较函数值
                value_passed = compare_values(rm_result, torch_result)
                
                # 测试self的梯度 - 统一使用backward方法
                grad_passed = True
                if TORCH_AVAILABLE:
                    # Riemann梯度 - 使用backward方法
                    rm_loss = rm_result.sum()
                    rm_loss.backward()
                    rm_tensor_grad = rm_tensor.grad
                    
                    # PyTorch梯度
                    torch_loss = torch_result.sum()
                    torch_loss.backward()
                    torch_tensor_grad = torch_tensor.grad
                    
                    # 比较梯度
                    grad_passed = compare_values(rm_tensor_grad, torch_tensor_grad)
                
                subcase_name = f"子用例 {i+1}: 维度={dim}, 形状={shape}, 标量值={scalar}"
                subcase_passed = value_passed and grad_passed and is_non_inplace
                all_passed = all_passed and subcase_passed
                
                # 为每个子用例单独添加统计
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(subcase_name, subcase_passed)
                    status = "通过" if subcase_passed else "失败"
                    status_color = Colors.OKGREEN if subcase_passed else Colors.FAIL
                    print(f"  子用例 {i+1}: {subcase_name.split(':')[1]} - {status_color}{status}{Colors.ENDC}")
                    if not value_passed:
                        print(f"    - 函数值比较失败")
                    if not is_non_inplace:
                        print(f"    - 非原地操作验证失败")
                    if TORCH_AVAILABLE and not grad_passed:
                        print(f"    - 梯度比较失败")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                status = "通过" if all_passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if all_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(all_passed, f"scatter_add函数（标量src参数）测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_scatter_direct_indices(self):
        """测试_compute_direct_indices函数的正确性，验证scatter函数内部实现"""
        case_name = "scatter函数_compute_direct_indices_v2测试"
        start_time = time.time()
        try:
            # 测试场景1: 标准情况 - 确保维度数相同，但测试_compute_direct_indices_v2的内部实现
            test_cases = [
                # (target_shape, dim, test_desc)
                ((3, 4), 0, "2D张量测试，dim=0"),
                ((3, 4), 1, "2D张量测试，dim=1"),
                ((2, 3, 4), 1, "3D张量测试，中间维度"),
                ((2, 3, 4), -1, "3D张量测试，负维度索引"),
                ((1, 5, 3, 2), 2, "4D张量测试"),
            ]
            
            all_passed = True
            
            for i, (shape, dim, desc) in enumerate(test_cases, 1):
                try:
                    # 创建测试数据
                    np_tensor = np.random.randn(*shape).astype(np.float32)
                    # 创建维度数相同的index数组，但可能在其他维度上有不同的大小
                    np_index = np.random.randint(0, shape[dim] if dim >= 0 else shape[dim + len(shape)], size=shape)
                    np_src = np.random.randn(*shape).astype(np.float32)  # src形状与index相同
                    
                    # Riemann实现
                    rm_tensor = tensor(np_tensor, requires_grad=True)
                    rm_index = tensor(np_index)
                    rm_src = tensor(np_src, requires_grad=True)
                    
                    # 使用scatter函数 - 如果_compute_direct_indices_v2正确，这里应该能正常工作
                    rm_result = rm_tensor.scatter(dim, rm_index, rm_src)
                    
                    # 验证结果形状是否正确
                    self.assertEqual(rm_result.shape, shape, f"形状不匹配: 期望{shape}, 得到{rm_result.shape}")
                    
                    # 测试梯度
                    if rm_tensor.requires_grad:
                        rm_result.sum().backward()
                        # 验证梯度是否存在
                        self.assertIsNotNone(rm_tensor.grad, "梯度计算失败")
                        self.assertIsNotNone(rm_src.grad, "梯度计算失败")
                    
                    if IS_RUNNING_AS_SCRIPT:
                        subcase_name = f"测试场景: {desc}"
                        stats.add_result(subcase_name, True)
                        print(f"  测试场景: {desc} - {Colors.OKGREEN}通过{Colors.ENDC}")
                except Exception as e:
                    all_passed = False
                    if IS_RUNNING_AS_SCRIPT:
                        subcase_name = f"测试场景: {desc}"
                        stats.add_result(subcase_name, False, [str(e)])
                        print(f"  测试场景: {desc} - {Colors.FAIL}失败{Colors.ENDC} - {str(e)}")
            
            # 测试场景2: 验证使用_compute_direct_indices_v2可以避免原始实现中的错误
            try:
                # 创建特殊测试用例，这可能会暴露原始_compute_direct_indices的问题
                shape = (2, 2)
                dim = 0
                np_tensor = np.zeros(shape).astype(np.float32)
                np_index = np.array([[0, 1], [1, 0]])
                np_src = np.ones(shape).astype(np.float32)
                
                rm_tensor = tensor(np_tensor, requires_grad=True)
                rm_index = tensor(np_index)
                rm_src = tensor(np_src, requires_grad=True)
                
                # 这应该能正常工作，因为我们使用的是_compute_direct_indices_v2
                rm_result = rm_tensor.scatter(dim, rm_index, rm_src)
                
                # 验证结果
                expected_result = np.array([[1, 1], [1, 1]]).astype(np.float32)
                self.assertTrue(np.allclose(rm_result.data, expected_result), "结果验证失败")
                
                if IS_RUNNING_AS_SCRIPT:
                    subcase_name = f"测试场景: 特殊索引模式测试"
                    stats.add_result(subcase_name, True)
                    print(f"  测试场景: 特殊索引模式测试 - {Colors.OKGREEN}通过{Colors.ENDC}")
            except Exception as e:
                all_passed = False
                if IS_RUNNING_AS_SCRIPT:
                    subcase_name = f"测试场景: 特殊索引模式测试"
                    stats.add_result(subcase_name, False, [str(e)])
                    print(f"  测试场景: 特殊索引模式测试 - {Colors.FAIL}失败{Colors.ENDC} - {str(e)}")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, all_passed)
                status = "通过" if all_passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if all_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(all_passed, "scatter函数_compute_direct_indices测试失败")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
            
    def test_scatter_boundary_cases(self):
        """测试scatter函数的边界情况"""
        case_name = "scatter函数边界情况测试"
        start_time = time.time()
        try:
            boundary_cases = [
                # 测试空张量
                {"shape": (0, 0), "dim": 0, "desc": "空张量"},
                # # 测试维度边界
                # {"shape": (3, 4), "dim": -1, "desc": "负维度索引"},
                # # 测试相同索引值
                # {"shape": (2, 3), "dim": 0, "same_indices": True, "desc": "相同索引值"},
            ]
            
            all_passed = True
            for i, case in enumerate(boundary_cases):
                shape = case["shape"]
                dim = case["dim"]
                desc = case["desc"]
                
                # 创建测试数据
                np_tensor = np.random.randn(*shape)
                if case.get("same_indices", False):
                    # 使用相同的索引值
                    np_index = np.full(shape, 0)
                else:
                    np_index = np.random.randint(0, shape[dim] if dim >= 0 else shape[dim + len(shape)], size=shape)
                np_src = np.random.randn(*shape)
                
                try:
                    # Riemann实现
                    rm_tensor = tensor(np_tensor, requires_grad=True)
                    rm_index = tensor(np_index)
                    rm_src = tensor(np_src, requires_grad=True)
                    rm_result = rm_tensor.scatter(dim, rm_index, rm_src)
                                        
                    # PyTorch实现
                    torch_result = None
                    if TORCH_AVAILABLE:
                        torch_tensor = torch.tensor(np_tensor, requires_grad=True)
                        torch_index = torch.tensor(np_index)
                        torch_src = torch.tensor(np_src, requires_grad=True)
                        torch_result = torch_tensor.scatter(dim, torch_index, torch_src)
                    
                    # 比较函数值
                    value_passed = compare_values(rm_result, torch_result)
                    
                    # 测试梯度
                    grad_passed = True
                    if TORCH_AVAILABLE:
                        # Riemann梯度
                        rm_loss = rm_result.sum()
                        rm_tensor_grad, rm_src_grad = grad(rm_loss, [rm_tensor, rm_src])
                        
                        # PyTorch梯度
                        torch_loss = torch_result.sum()
                        torch_loss.backward()
                        torch_tensor_grad = torch_tensor.grad
                        torch_src_grad = torch_src.grad
                        
                        # 比较梯度
                        tensor_grad_passed = compare_values(rm_tensor_grad, torch_tensor_grad)
                        src_grad_passed = compare_values(rm_src_grad, torch_src_grad)
                        grad_passed = tensor_grad_passed and src_grad_passed
                    
                    subcase_passed = value_passed and grad_passed
                    all_passed = all_passed and subcase_passed
                    
                    if IS_RUNNING_AS_SCRIPT:
                        subcase_name = f"边界用例 {i+1}: {desc}"
                        stats.add_result(subcase_name, subcase_passed)
                        status = "通过" if subcase_passed else "失败"
                        status_color = Colors.OKGREEN if subcase_passed else Colors.FAIL
                        print(f"  边界用例 {i+1}: {desc} - {status_color}{status}{Colors.ENDC}")
                        if not value_passed:
                            print(f"    - 函数值比较失败")
                        if TORCH_AVAILABLE and not grad_passed:
                            print(f"    - 梯度比较失败")
                    
                except Exception as e:
                    # 检查是否是预期的异常
                    subcase_passed = False
                    all_passed = False
                    
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"  边界用例 {i+1}: {desc} - {Colors.FAIL}异常{Colors.ENDC} - {str(e)}")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, all_passed)
                status = "通过" if all_passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if all_passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(all_passed, f"scatter函数边界情况测试失败")
            
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
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行Scatter函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestScatterFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)