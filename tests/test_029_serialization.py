import unittest
import numpy as np
import time
import sys, os
import tempfile

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入torch
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False
    print("警告: 无法导入torch模块，将跳过PyTorch兼容性测试")

# 检测CUDA是否可用
has_cuda = rm.cuda.is_available()

# 从riemann.cuda获取cupy句柄
cp = rm.cuda.cp

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

class TestSerialization(unittest.TestCase):
    def setUp(self):
        """每个测试方法执行前的设置"""
        self.current_test_name = ""
        
    def tearDown(self):
        """每个测试方法执行后的清理"""
        if self.current_test_name:
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_tensor_serialization(self):
        """测试张量序列化功能"""
        stats.start_function("张量序列化")
        try:
            test_cases = [
            {"name": "基本浮点张量", "dtype": np.float32, "shape": (3, 4), "requires_grad": True, "device": "cpu"},
            {"name": "双精度浮点张量", "dtype": np.float64, "shape": (2, 5, 3), "requires_grad": True, "device": "cpu"},
            {"name": "整数张量", "dtype": np.int32, "shape": (6, 2), "requires_grad": False, "device": "cpu"},
            {"name": "长整数张量", "dtype": np.int64, "shape": (4, 3, 2, 1), "requires_grad": False, "device": "cpu"},
            {"name": "复数张量64", "dtype": np.complex64, "shape": (2, 3), "requires_grad": False, "device": "cpu"},
            {"name": "复数张量128", "dtype": np.complex128, "shape": (3, 2), "requires_grad": False, "device": "cpu"},
            {"name": "布尔张量", "dtype": np.bool_, "shape": (4, 5), "requires_grad": False, "device": "cpu"},
            ]
            
            # 如果CUDA可用，添加CUDA测试用例
            if has_cuda:
                test_cases.extend([
                    {"name": "CUDA基本浮点张量", "dtype": np.float32, "shape": (3, 4), "requires_grad": True, "device": "cuda"},
                    {"name": "CUDA双精度浮点张量", "dtype": np.float64, "shape": (2, 5, 3), "requires_grad": True, "device": "cuda"},
                    {"name": "CUDA复数张量64", "dtype": np.complex64, "shape": (2, 3), "requires_grad": False, "device": "cuda"},
                    {"name": "CUDA复数张量128", "dtype": np.complex128, "shape": (3, 2), "requires_grad": False, "device": "cuda"},
                ])
        
            for case in test_cases:
                case_name = f"张量序列化 - {case['name']}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    if case['shape'] == ():
                        # 处理标量情况
                        if case['dtype'] == np.bool_:
                            data = np.bool_(np.random.choice([True, False]))
                        elif case['dtype'] in [np.complex64, np.complex128]:
                            data = np.array(np.random.randn() + 1j * np.random.randn(), dtype=case['dtype'])
                        elif case['dtype'] in [np.int32, np.int64]:
                            data = np.array(np.random.randint(-10, 10), dtype=case['dtype'])
                        else:
                            data = np.array(np.random.randn(), dtype=case['dtype'])
                    else:
                        if case['dtype'] == np.bool_:
                            data = np.random.choice([True, False], case['shape']).astype(case['dtype'])
                        elif case['dtype'] in [np.complex64, np.complex128]:
                            data = (np.random.randn(*case['shape']) + 1j * np.random.randn(*case['shape'])).astype(case['dtype'])
                        elif case['dtype'] in [np.int32, np.int64]:
                            data = np.random.randint(-10, 10, case['shape']).astype(case['dtype'])
                        else:
                            data = np.random.randn(*case['shape']).astype(case['dtype'])
                    
                    original_tensor = rm.tensor(data, requires_grad=case['requires_grad'], device=case['device'])
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(original_tensor, temp_path)
                        loaded_tensor = rm.load(temp_path)
                        
                        # 验证形状
                        self.assertEqual(loaded_tensor.shape, original_tensor.shape)
                        
                        # 验证数据类型
                        self.assertEqual(str(loaded_tensor.dtype), str(original_tensor.dtype))
                        
                        # 验证数据内容
                        # 处理CUDA张量
                        orig_data = original_tensor.data.get() if cp is not None and hasattr(original_tensor.data, 'get') else original_tensor.data
                        loaded_data = loaded_tensor.data.get() if cp is not None and hasattr(loaded_tensor.data, 'get') else loaded_tensor.data
                        np.testing.assert_allclose(loaded_data, orig_data, rtol=1e-6)
                        
                        # 验证属性
                        self.assertEqual(loaded_tensor.requires_grad, original_tensor.requires_grad)
                        self.assertEqual(loaded_tensor.is_leaf, original_tensor.is_leaf)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
            
    def test_parameter_serialization(self):
        """测试参数序列化功能"""
        stats.start_function("参数序列化")
        try:
            test_cases = [
            {"name": "基本参数", "dtype": np.float32, "shape": (5, 3), "device": "cpu"},
            {"name": "双精度参数", "dtype": np.float64, "shape": (2, 4), "device": "cpu"},
            ]
            
            # 如果CUDA可用，添加CUDA测试用例
            if has_cuda:
                test_cases.extend([
                    {"name": "CUDA基本参数", "dtype": np.float32, "shape": (5, 3), "device": "cuda"},
                    {"name": "CUDA双精度参数", "dtype": np.float64, "shape": (2, 4), "device": "cuda"},
                ])
        
            for case in test_cases:
                case_name = f"参数序列化 - {case['name']}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    data = np.random.randn(*case['shape']).astype(case['dtype'])
                    original_param = rm.nn.Parameter(rm.tensor(data, device=case['device']))
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(original_param, temp_path)
                        loaded_param = rm.load(temp_path)
                        
                        # 验证形状
                        self.assertEqual(loaded_param.shape, original_param.shape)
                        
                        # 验证数据类型
                        self.assertEqual(str(loaded_param.dtype), str(original_param.dtype))
                        
                        # 验证数据内容
                        # 处理CUDA参数
                        orig_data = original_param.data.get() if cp is not None and hasattr(original_param.data, 'get') else original_param.data
                        loaded_data = loaded_param.data.get() if cp is not None and hasattr(loaded_param.data, 'get') else loaded_param.data
                        np.testing.assert_allclose(loaded_data, orig_data, rtol=1e-6)
                        
                        # 验证属性
                        self.assertEqual(loaded_param.requires_grad, original_param.requires_grad)
                        self.assertEqual(loaded_param.is_leaf, original_param.is_leaf)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
            
    def test_dict_serialization(self):
        """测试字典序列化功能"""
        stats.start_function("字典序列化")
        try:
            test_cases = [
            {"name": "基本字典", "data": {"epoch": 10, "loss": 0.123}},
            {"name": "包含张量的字典", "data": {"tensor1": None, "tensor2": None, "epoch": 10, "loss": 0.123}},
            {"name": "嵌套字典", "data": {"level1": {"level2": {"value": 42}}, "top": "test"}},
            ]
        
            # 为包含张量的测试用例创建张量
            test_cases[1]["data"]["tensor1"] = rm.randn(2, 3)
            test_cases[1]["data"]["tensor2"] = rm.randn(4, 5)
            
            for case in test_cases:
                case_name = f"字典序列化 - {case['name']}"
                start_time = time.time()
                try:
                    original_dict = case['data']
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(original_dict, temp_path)
                        loaded_dict = rm.load(temp_path)
                        
                        # 验证基本数据
                        for key, value in original_dict.items():
                            if hasattr(value, 'shape'):  # 张量
                                self.assertEqual(loaded_dict[key].shape, value.shape)
                                np.testing.assert_allclose(loaded_dict[key].data, value.data, rtol=1e-6)
                            else:  # 普通数据
                                self.assertEqual(loaded_dict[key], value)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
    
    def test_module_state_dict(self):
        """测试模块状态字典序列化"""
        stats.start_function("模块状态字典")
        try:
            test_cases = [
            {"name": "线性层", "module_type": "Linear"},
            {"name": "序列模块", "module_type": "Sequential"},
            ]
        
            for case in test_cases:
                case_name = f"模块状态字典 - {case['name']}"
                start_time = time.time()
                try:
                    # 创建模块
                    if case['module_type'] == 'Linear':
                        model = rm.nn.Linear(10, 5)
                    elif case['module_type'] == 'Sequential':
                        model = rm.nn.Sequential(
                            rm.nn.Linear(10, 20),
                            rm.nn.ReLU(),
                            rm.nn.Linear(20, 5)
                        )
                    
                    original_state_dict = model.state_dict(keep_vars=True)
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(original_state_dict, temp_path)
                        loaded_state_dict = rm.load(temp_path)
                        
                        # 验证键匹配
                        self.assertEqual(set(loaded_state_dict.keys()), set(original_state_dict.keys()))
                        
                        # 验证每个参数
                        for key in original_state_dict.keys():
                            self.assertEqual(loaded_state_dict[key].shape, original_state_dict[key].shape)
                            self.assertEqual(loaded_state_dict[key].requires_grad, original_state_dict[key].requires_grad)
                            np.testing.assert_allclose(loaded_state_dict[key].data, original_state_dict[key].data, rtol=1e-6)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
    
    def test_pickle_module_parameter(self):
        """测试pickle_module参数"""
        stats.start_function("pickle模块参数")
        try:
            import pickle
            
            test_cases = [
                {"name": "默认pickle模块", "pickle_module": None},
                {"name": "显式pickle模块", "pickle_module": pickle},
            ]
        
            for case in test_cases:
                case_name = f"pickle模块参数 - {case['name']}"
                start_time = time.time()
                try:
                    tensor = rm.randn(3, 3)
                    
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(tensor, temp_path, pickle_module=case['pickle_module'])
                        loaded = rm.load(temp_path, pickle_module=case['pickle_module'])
                        
                        self.assertEqual(loaded.shape, tensor.shape)
                        np.testing.assert_allclose(loaded.data, tensor.data, rtol=1e-6)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
    
    def test_pickle_protocol_versions(self):
        """测试pickle协议版本"""
        stats.start_function("pickle协议版本")
        try:
            import pickle
            
            # 根据系统支持的协议版本确定测试用例
            protocols = [0, 1, 2, 3, 4, 5] if hasattr(pickle, 'PROTOCOL') else [0, 1, 2]
        
            for protocol in protocols:
                case_name = f"pickle协议版本 - 协议{protocol}"
                start_time = time.time()
                try:
                    tensor = rm.randn(2, 2)
                    
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(tensor, temp_path, pickle_protocol=protocol)
                        loaded = rm.load(temp_path)
                        
                        self.assertEqual(loaded.shape, tensor.shape)
                        np.testing.assert_allclose(loaded.data, tensor.data, rtol=1e-6)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    # 协议版本不支持是可接受的，不抛出异常
                    print(f"  协议{protocol}可能不被当前系统支持")
        finally:
            stats.end_function()
    
    def test_file_like_objects(self):
        """测试文件类对象"""
        stats.start_function("文件类对象")
        try:
            import io
            
            test_cases = [
                {"name": "BytesIO对象", "file_obj": io.BytesIO()},
            ]
            
            for case in test_cases:
                case_name = f"文件类对象 - {case['name']}"
                start_time = time.time()
                try:
                    tensor = rm.randn(3, 3)
                    buffer = io.BytesIO()
                    
                    rm.save(tensor, buffer)
                    buffer.seek(0)
                    loaded = rm.load(buffer)
                    
                    self.assertEqual(loaded.shape, tensor.shape)
                    np.testing.assert_allclose(loaded.data, tensor.data, rtol=1e-6)
                    
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, True)
                    print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
    
    def test_nested_structures(self):
        """测试嵌套结构"""
        stats.start_function("嵌套结构")
        try:
            test_cases = [
            {"name": "多层嵌套", "data": None},
            {"name": "混合类型列表", "data": None},
            ]
        
            # 创建测试数据
            nested_data = {
                'level1': {
                    'level2': {
                        'tensor': rm.randn(2, 2),
                        'list': [rm.randn(1, 1), rm.randn(2, 2)]
                    },
                    'simple_list': [1, 2, 3]
                },
                'top_tensor': rm.randn(3, 3),
                'tuple_data': (rm.randn(1, 1), 'string', 42)
            }
            test_cases[0]["data"] = nested_data
            
            mixed_list = [
                rm.randn(2, 2),
                {"nested_tensor": rm.randn(1, 1)},
                (rm.randn(3, 3), "text", 123)
            ]
            test_cases[1]["data"] = mixed_list
            
            for case in test_cases:
                case_name = f"嵌套结构 - {case['name']}"
                start_time = time.time()
                try:
                    original_data = case['data']
                    
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(original_data, temp_path)
                        loaded_data = rm.load(temp_path)
                        
                        # 验证嵌套结构中的张量
                        def verify_structure(orig, loaded):
                            if hasattr(orig, 'shape'):  # 张量
                                self.assertEqual(loaded.shape, orig.shape)
                                np.testing.assert_allclose(loaded.data, orig.data, rtol=1e-6)
                            elif isinstance(orig, dict):
                                self.assertEqual(set(loaded.keys()), set(orig.keys()))
                                for key in orig:
                                    verify_structure(orig[key], loaded[key])
                            elif isinstance(orig, (list, tuple)):
                                for orig_item, loaded_item in zip(orig, loaded):
                                    verify_structure(orig_item, loaded_item)
                            else:
                                self.assertEqual(loaded, orig)
                        
                        verify_structure(original_data, loaded_data)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
    
    def test_error_handling(self):
        """测试错误处理"""
        stats.start_function("错误处理")
        import pickle
        
        test_cases = [
            {"name": "不存在的文件", "operation": "load_nonexistent"},
            {"name": "无效的pickle数据", "operation": "load_invalid"},
        ]
        
        for case in test_cases:
            case_name = f"错误处理 - {case['name']}"
            start_time = time.time()
            try:
                if case['operation'] == 'load_nonexistent':
                    # 测试加载不存在的文件
                    with self.assertRaises((FileNotFoundError, OSError)):
                        rm.load('non_existent_file.pt')
                        
                elif case['operation'] == 'load_invalid':
                    # 测试加载无效数据
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                        f.write(b'invalid pickle data')
                    
                    try:
                        with self.assertRaises((pickle.UnpicklingError, EOFError)):
                            rm.load(temp_path)
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                
                time_taken = time.time() - start_time
                stats.add_result(case_name, True)
                print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                
            except Exception as e:
                time_taken = time.time() - start_time
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                raise
    
    def test_tensor_device_serialization(self):
        """测试张量device参数序列化功能"""
        stats.start_function("张量Device参数序列化")
        try:
            start_time = time.time()
            # 测试CPU设备
            cpu_tensor = rm.tensor([1.0, 2.0, 3.0], requires_grad=True, device='cpu')
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                temp_path = f.name
            
            try:
                rm.save(cpu_tensor, temp_path)
                loaded_cpu_tensor = rm.load(temp_path)
                
                # 验证数据
                np.testing.assert_allclose(loaded_cpu_tensor.data, cpu_tensor.data, rtol=1e-6)
                
                # 验证device参数
                self.assertEqual(str(loaded_cpu_tensor.device), str(cpu_tensor.device))
                self.assertEqual(str(loaded_cpu_tensor.device), 'cpu')
                
                time_taken = time.time() - start_time
                stats.add_result("张量Device参数序列化 - CPU设备", True)
                print(f"测试用例: 张量Device参数序列化 - CPU设备 - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                
                # 测试参数的device序列化
                cpu_param = rm.nn.Parameter(rm.tensor([4.0, 5.0, 6.0], device='cpu'))
                rm.save(cpu_param, temp_path)
                loaded_cpu_param = rm.load(temp_path)
                
                # 验证参数数据
                np.testing.assert_allclose(loaded_cpu_param.data, cpu_param.data, rtol=1e-6)
                
                # 验证参数device
                self.assertEqual(str(loaded_cpu_param.device), str(cpu_param.device))
                self.assertEqual(str(loaded_cpu_param.device), 'cpu')
                
                time_taken = time.time() - start_time
                stats.add_result("参数Device参数序列化 - CPU设备", True)
                print(f"测试用例: 参数Device参数序列化 - CPU设备 - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            time_taken = time.time() - start_time
            stats.add_result("张量Device参数序列化", False, [str(e)])
            print(f"测试用例: 张量Device参数序列化 - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
        finally:
            stats.end_function()
    
    def test_basic_types(self):
        """测试基本数据类型序列化"""
        stats.start_function("基本数据类型序列化")
        try:
            test_cases = [
                {"name": "整数", "data": 42},
                {"name": "浮点数", "data": 3.14159},
                {"name": "复数", "data": 1 + 2j},
                {"name": "字符串", "data": "Hello, Riemann!"},
                {"name": "布尔值True", "data": True},
                {"name": "布尔值False", "data": False},
                {"name": "None值", "data": None},
            ]
        
            for case in test_cases:
                case_name = f"基本数据类型 - {case['name']}"
                start_time = time.time()
                try:
                    original_data = case['data']
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(original_data, temp_path)
                        loaded_data = rm.load(temp_path)
                        
                        # 验证数据
                        self.assertEqual(loaded_data, original_data)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
    
    def test_numpy_arrays(self):
        """测试NumPy数组序列化"""
        stats.start_function("NumPy数组序列化")
        try:
            test_cases = [
                {"name": "float32数组", "dtype": np.float32, "shape": (2, 3)},
                {"name": "float64数组", "dtype": np.float64, "shape": (3, 2)},
                {"name": "int32数组", "dtype": np.int32, "shape": (4,)},
                {"name": "int64数组", "dtype": np.int64, "shape": (2, 2)},
                {"name": "布尔数组", "dtype": np.bool_, "shape": (3, 3)},
                {"name": "complex64数组", "dtype": np.complex64, "shape": (2, 2)},
                {"name": "complex128数组", "dtype": np.complex128, "shape": (2, 2)},
                {"name": "空数组", "dtype": np.float32, "shape": ()},
            ]
        
            for case in test_cases:
                case_name = f"NumPy数组 - {case['name']}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    if case['shape'] == ():
                        # 处理标量情况
                        if case['dtype'] == np.bool_:
                            data = np.bool_(np.random.choice([True, False]))
                        elif case['dtype'] in [np.complex64, np.complex128]:
                            data = np.array(np.random.randn() + 1j * np.random.randn(), dtype=case['dtype'])
                        elif case['dtype'] in [np.int32, np.int64]:
                            data = np.array(np.random.randint(-10, 10), dtype=case['dtype'])
                        else:
                            data = np.array(np.random.randn(), dtype=case['dtype'])
                    else:
                        if case['dtype'] == np.bool_:
                            data = np.random.choice([True, False], case['shape']).astype(case['dtype'])
                        elif case['dtype'] in [np.complex64, np.complex128]:
                            data = (np.random.randn(*case['shape']) + 1j * np.random.randn(*case['shape'])).astype(case['dtype'])
                        elif case['dtype'] in [np.int32, np.int64]:
                            data = np.random.randint(-10, 10, case['shape']).astype(case['dtype'])
                        else:
                            data = np.random.randn(*case['shape']).astype(case['dtype'])
                    
                    original_array = data
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(original_array, temp_path)
                        loaded_array = rm.load(temp_path)
                        
                        # 验证数据
                        if isinstance(loaded_array, rm.TN):
                            # 如果加载为Riemann张量，提取其数据
                            loaded_data = loaded_array.data
                        else:
                            loaded_data = loaded_array
                        np.testing.assert_array_equal(original_array, loaded_data)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
            
    def test_pytorch_compatibility(self):
        """测试Riemann与PyTorch序列化兼容性"""
        if not has_torch:
            return
        
        stats.start_function("PyTorch兼容性")
        try:
            test_cases = [
                {"name": "Riemann加载PyTorch文件", "test_type": "riemann_load_torch"},
                {"name": "PyTorch加载Riemann文件", "test_type": "torch_load_riemann"},
            ]
        
            for case in test_cases:
                case_name = f"PyTorch兼容性 - {case['name']}"
                start_time = time.time()
                try:
                    if case['test_type'] == 'riemann_load_torch':
                        # 测试Riemann加载PyTorch保存的文件
                        # 创建PyTorch张量
                        t1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
                        t2 = torch.randn(2, 3)
                        
                        # PyTorch保存
                        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                            filename = f.name
                        
                        try:
                            torch.save({'t1': t1, 't2': t2}, filename)
                            
                            # Riemann加载
                            loaded = rm.load(filename)
                            
                            # 验证数据（转换为numpy比较）
                            t1_numpy = t1.detach().numpy() if t1.requires_grad else t1.numpy()
                            loaded_t1_numpy = loaded['t1'].detach().numpy() if loaded['t1'].requires_grad else loaded['t1'].numpy()
                            np.testing.assert_allclose(loaded_t1_numpy, t1_numpy, rtol=1e-6)
                            
                            t2_numpy = t2.detach().numpy() if t2.requires_grad else t2.numpy()
                            loaded_t2_numpy = loaded['t2'].detach().numpy() if loaded['t2'].requires_grad else loaded['t2'].numpy()
                            np.testing.assert_allclose(loaded_t2_numpy, t2_numpy, rtol=1e-6)
                            
                        finally:
                            if os.path.exists(filename):
                                os.unlink(filename)
                                
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                            
                    elif case['test_type'] == 'torch_load_riemann':
                        # 测试PyTorch加载Riemann保存的文件
                        # 创建Riemann张量
                        t1 = rm.tensor([1.0, 2.0, 3.0, 4.0])
                        t2 = rm.randn(2, 3)
                        
                        # Riemann保存
                        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                            filename = f.name
                        
                        try:
                            rm.save({'t1': t1, 't2': t2}, filename)
                            
                            # PyTorch加载（设置weights_only=False以允许加载自定义构造函数）
                            loaded = torch.load(filename, weights_only=False)
                            
                            # 验证数据（转换为numpy比较）
                            np.testing.assert_allclose(loaded['t1'].numpy(), t1.numpy(), rtol=1e-6)
                            np.testing.assert_allclose(loaded['t2'].numpy(), t2.numpy(), rtol=1e-6)
                            
                        finally:
                            if os.path.exists(filename):
                                os.unlink(filename)
                                
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
    
    def test_cupy_arrays(self):
        """测试CuPy数组序列化"""
        if cp is None:
            print("CuPy not available, skipping CuPy tests")
            return
        
        if not has_cuda:
            print("CUDA not available, skipping CuPy tests")
            return
        
        stats.start_function("CuPy数组序列化")
        try:
            
            test_cases = [
                {"name": "float32数组", "dtype": cp.float32, "shape": (2, 3)},
                {"name": "float64数组", "dtype": cp.float64, "shape": (3, 2)},
            ]
            
            # 尝试添加复数数组测试
            try:
                test_cases.extend([
                    {"name": "complex64数组", "dtype": cp.complex64, "shape": (2, 2)},
                    {"name": "complex128数组", "dtype": cp.complex128, "shape": (2, 2)},
                ])
            except Exception as e:
                print(f"创建CuPy复数数组测试用例失败: {e}")
                # 跳过CuPy复数数组测试
        
            for case in test_cases:
                case_name = f"CuPy数组 - {case['name']}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    if case['dtype'] in [cp.complex64, cp.complex128]:
                        # 使用更简单的方式创建复数数组
                        data = cp.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=case['dtype'])
                    elif case['dtype'] in [cp.int32, cp.int64]:
                        data = cp.random.randint(-10, 10, case['shape'], dtype=case['dtype'])
                    else:
                        data = cp.random.randn(*case['shape']).astype(case['dtype'])
                    
                    original_array = data
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(original_array, temp_path)
                        loaded_array = rm.load(temp_path)
                        
                        # 验证数据
                        original_np = original_array.get()
                        if isinstance(loaded_array, rm.TN):
                            # 如果加载为Riemann张量，提取其数据
                            loaded_data = loaded_array.data
                            # 确保转换为NumPy数组
                            if cp is not None and isinstance(loaded_data, cp.ndarray):
                                loaded_data = loaded_data.get()
                            elif not isinstance(loaded_data, np.ndarray):
                                loaded_data = np.asarray(loaded_data)
                        else:
                            # 确保加载的数据是NumPy数组
                            if cp is not None and isinstance(loaded_array, cp.ndarray):
                                loaded_data = loaded_array.get()
                            else:
                                loaded_data = np.asarray(loaded_array)
                        np.testing.assert_array_equal(original_np, loaded_data)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            # 确保在任何情况下都调用end_function
            if stats.current_function == "CuPy数组序列化":
                stats.end_function()
    
    def test_combination_types(self):
        """测试组合类型序列化"""
        stats.start_function("组合类型序列化")
        try:
            test_cases = [
                {
                    "name": "包含张量的字典", 
                    "data": {
                        "cpu_tensor": rm.randn(2, 3, device="cpu"),
                        "numpy_array": np.random.randn(3, 2),
                        "metadata": {"version": 1, "author": "Test"}
                    }
                },
                {
                    "name": "混合类型列表", 
                    "data": [1, "string", 3.14, True, rm.randn(2, 2)]
                },
                {
                    "name": "嵌套字典", 
                    "data": {
                        "level1": {
                            "level2": {
                                "tensor": rm.randn(1, 1),
                                "array": np.array([1, 2, 3])
                            }
                        }
                    }
                },
            ]
            
            if has_cuda:
                test_cases.append({
                    "name": "混合设备张量的字典", 
                    "data": {
                        "cpu": rm.randn(2, 3, device="cpu"),
                        "cuda": rm.randn(2, 3, device="cuda")
                    }
                })
        
            for case in test_cases:
                case_name = f"组合类型 - {case['name']}"
                start_time = time.time()
                try:
                    original_data = case['data']
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(original_data, temp_path)
                        loaded_data = rm.load(temp_path)
                        
                        # 验证数据
                        def verify_structure(orig, loaded):
                            if hasattr(orig, 'shape'):
                                # 张量
                                if isinstance(loaded, rm.TN):
                                    self.assertEqual(loaded.shape, orig.shape)
                                    # 处理CUDA张量
                                    orig_data = orig.data.get() if cp is not None and hasattr(orig.data, 'get') else orig.data
                                    loaded_data = loaded.data.get() if cp is not None and hasattr(loaded.data, 'get') else loaded.data
                                    np.testing.assert_allclose(loaded_data, orig_data, rtol=1e-6)
                                # 数组
                                elif isinstance(orig, np.ndarray):
                                    if isinstance(loaded, rm.TN):
                                        loaded_data = loaded.data.get() if cp is not None and hasattr(loaded.data, 'get') else loaded.data
                                        np.testing.assert_array_equal(orig, loaded_data)
                                    else:
                                        np.testing.assert_array_equal(orig, loaded)
                            elif isinstance(orig, dict):
                                self.assertEqual(set(loaded.keys()), set(orig.keys()))
                                for key in orig:
                                    verify_structure(orig[key], loaded[key])
                            elif isinstance(orig, (list, tuple)):
                                for orig_item, loaded_item in zip(orig, loaded):
                                    verify_structure(orig_item, loaded_item)
                            else:
                                self.assertEqual(loaded, orig)
                        
                        verify_structure(original_data, loaded_data)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()
    
    def test_special_cases(self):
        """测试特殊情况序列化"""
        stats.start_function("特殊情况序列化")
        try:
            test_cases = [
                {"name": "大型NumPy数组", "data": np.random.randn(100, 100)},
                {"name": "空张量", "data": rm.tensor([])},
            ]
        
            for case in test_cases:
                case_name = f"特殊情况 - {case['name']}"
                start_time = time.time()
                try:
                    original_data = case['data']
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        temp_path = f.name
                    
                    try:
                        rm.save(original_data, temp_path)
                        loaded_data = rm.load(temp_path)
                        
                        # 验证数据
                        if isinstance(original_data, np.ndarray):
                            if isinstance(loaded_data, rm.TN):
                                np.testing.assert_array_equal(original_data, loaded_data.data)
                            else:
                                np.testing.assert_array_equal(original_data, loaded_data)
                        elif isinstance(original_data, rm.TN):
                            self.assertEqual(loaded_data.shape, original_data.shape)
                            if original_data.shape:
                                np.testing.assert_allclose(loaded_data.data, original_data.data, rtol=1e-6)
                        
                        time_taken = time.time() - start_time
                        stats.add_result(case_name, True)
                        print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time_taken:.4f}秒)")
                        
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    time_taken = time.time() - start_time
                    stats.add_result(case_name, False, [str(e)])
                    print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
        finally:
            stats.end_function()

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行序列化函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestSerialization)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)