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
            {"name": "基本浮点张量", "dtype": np.float32, "shape": (3, 4), "requires_grad": True},
            {"name": "双精度浮点张量", "dtype": np.float64, "shape": (2, 5, 3), "requires_grad": True},
            {"name": "整数张量", "dtype": np.int32, "shape": (6, 2), "requires_grad": False},
            {"name": "长整数张量", "dtype": np.int64, "shape": (4, 3, 2, 1), "requires_grad": False},
            {"name": "复数张量64", "dtype": np.complex64, "shape": (2, 3), "requires_grad": False},
            {"name": "复数张量128", "dtype": np.complex128, "shape": (3, 2), "requires_grad": False},
            {"name": "布尔张量", "dtype": np.bool_, "shape": (4, 5), "requires_grad": False},
            ]
        
            for case in test_cases:
                case_name = f"张量序列化 - {case['name']}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    if case['dtype'] == np.bool_:
                        data = np.random.choice([True, False], case['shape']).astype(case['dtype'])
                    elif case['dtype'] in [np.complex64, np.complex128]:
                        data = (np.random.randn(*case['shape']) + 1j * np.random.randn(*case['shape'])).astype(case['dtype'])
                    elif case['dtype'] in [np.int32, np.int64]:
                        data = np.random.randint(-10, 10, case['shape']).astype(case['dtype'])
                    else:
                        data = np.random.randn(*case['shape']).astype(case['dtype'])
                    
                    original_tensor = rm.tensor(data, requires_grad=case['requires_grad'])
                    original_tensor.is_leaf = True
                    
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
                        np.testing.assert_allclose(loaded_tensor.data, original_tensor.data, rtol=1e-6)
                        
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
            {"name": "基本参数", "dtype": np.float32, "shape": (5, 3)},
            {"name": "双精度参数", "dtype": np.float64, "shape": (2, 4)},
            ]
        
            for case in test_cases:
                case_name = f"参数序列化 - {case['name']}"
                start_time = time.time()
                try:
                    # 创建测试数据
                    data = np.random.randn(*case['shape']).astype(case['dtype'])
                    original_param = rm.nn.Parameter(rm.tensor(data))
                    
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
                        np.testing.assert_allclose(loaded_param.data, original_param.data, rtol=1e-6)
                        
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
            cpu_tensor.is_leaf = True
            
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