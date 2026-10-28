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

# 检查CUDA是否可用
try:
    CUDA_AVAILABLE = rm.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

# 检查CuPy是否可用
CUPY_AVAILABLE = bool(getattr(rm.cuda, 'CUPY_AVAILABLE', False))
cp = rm.cuda.cp if CUPY_AVAILABLE else None

# 检查torch是否可用
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
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

# 辅助函数：取对象的底层数组（TN 用 .array() 零拷贝取得设备原生数组）
def _native(obj):
    return obj.detach().array() if isinstance(obj, rm.TN) else obj

# 辅助函数：判断底层数据是否在GPU上
def _is_gpu(obj):
    if CUPY_AVAILABLE and isinstance(obj, cp.ndarray):
        return True
    if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        return obj.is_cuda
    return False

# 辅助函数：取GPU数据的显存地址
def _gpu_ptr(obj):
    if CUPY_AVAILABLE and isinstance(obj, cp.ndarray):
        return obj.data.ptr
    return obj.data_ptr()

# 辅助函数：判断两个张量/数组是否共享同一块内存
# CPU 用 np.shares_memory；GPU 必须比较显存地址（cp.asarray 等会新建包装对象，
# 对象身份不同但共享显存）
def shares_memory(a, b):
    a, b = _native(a), _native(b)
    a_gpu, b_gpu = _is_gpu(a), _is_gpu(b)
    if a_gpu != b_gpu:
        return False
    if a_gpu:
        return _gpu_ptr(a) == _gpu_ptr(b)
    return np.shares_memory(np.asarray(a), np.asarray(b))

# 辅助函数：取CPU端numpy数组用于数值断言（GPU数据显式拉回CPU）
def _values(obj):
    obj = _native(obj)
    if CUPY_AVAILABLE and isinstance(obj, cp.ndarray):
        return cp.asnumpy(obj)
    if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)

# 测试统计类
class StatisticsCollector:
    def __init__(self):
        self.total_cases = 0
        self.passed_cases = 0
        self.total_time = 0.0
        self.function_stats = {}
        self.current_function = None
        self.current_function_start_time = 0
        self.current_test_details = []

    def start_function(self, function_name):
        self.current_function = function_name
        self.current_function_start_time = time.time()
        self.current_test_details = []
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
            if '\u4e00' <= char <= '\u9fff':
                width += 2
            else:
                width += 1
        return width

    def print_summary(self):
        headers = ['用例名', '通过/总数', '通过率', '耗时(秒)']
        header_widths = [self._get_display_width(h) for h in headers]

        max_func_name_width = header_widths[0]
        for func_name in self.function_stats.keys():
            max_func_name_width = max(max_func_name_width, self._get_display_width(func_name))

        col_widths = [
            max(max_func_name_width, header_widths[0]) + 2,
            header_widths[1] + 4,
            header_widths[2] + 4,
            header_widths[3] + 4
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

        header_line = ""
        for i, header in enumerate(headers):
            padding = col_widths[i] - self._get_display_width(header)
            header_line += header + " " * padding
        print(header_line)
        print("-"*total_width)

        for func_name, func_stat in self.function_stats.items():
            pass_rate = func_stat["passed"]/func_stat["total"]*100 if func_stat["total"] > 0 else 0
            status_color = Colors.OKGREEN if pass_rate == 100 else Colors.FAIL

            func_name_padding = col_widths[0] - self._get_display_width(func_name)

            pass_total_display = f"{func_stat['passed']}/{func_stat['total']}"
            pass_total_padding = col_widths[1] - self._get_display_width(pass_total_display)

            pass_rate_display = f"{pass_rate:.1f}%"
            pass_rate_padding = col_widths[2] - self._get_display_width(pass_rate_display)

            time_display = f"{func_stat['time']:.4f}"
            time_padding = col_widths[3] - self._get_display_width(time_display)

            data_line = (func_name + " " * func_name_padding +
                         pass_total_display + " " * pass_total_padding +
                         status_color + pass_rate_display + Colors.ENDC + " " * pass_rate_padding +
                         time_display + " " * time_padding)
            print(data_line)

        print("="*total_width)

# 定义测试统计实例
stats = StatisticsCollector()

# 是否作为独立脚本运行
IS_RUNNING_AS_SCRIPT = False


class TestTensorInterop(unittest.TestCase):
    """测试Riemann张量与 numpy数组 / cupy数组 / torch CPU与CUDA张量的互相转换"""

    def setUp(self):
        # 设备解析依赖全局默认设备，统一归位到CPU，避免其它测试残留的默认设备影响断言
        try:
            rm.cuda.set_default_device('cpu')
        except Exception:
            pass
        if IS_RUNNING_AS_SCRIPT:
            test_method_name = self._testMethodName
            self.current_test_name = test_method_name
            stats.start_function(test_method_name)
            print(f"\n{Colors.HEADER}开始测试: {test_method_name}{Colors.ENDC}")

    def tearDown(self):
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")

    def _run_subcase(self, doc, name, test_func):
        """运行单个子用例：统计、打印分组明细，失败时向上抛出异常"""
        case_name = f"{doc} - {name}"
        start_time = time.time()
        try:
            test_func()
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, True)
                print(f"测试用例: {case_name} - {Colors.OKGREEN}通过{Colors.ENDC} ({time.time() - start_time:.4f}秒)")
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time.time() - start_time:.4f}秒) - {str(e)}")
            raise

    # ------------------------------------------------------------------
    # numpy数组 -> Riemann张量
    # ------------------------------------------------------------------
    def test_from_numpy(self):
        """numpy数组转Riemann张量"""
        doc = self.test_from_numpy.__doc__

        def _tensor_copy():
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            t = rm.tensor(arr, device='cpu')
            self.assertEqual(t.device.type, 'cpu')
            self.assertEqual(t.dtype, rm.float32, "tensor应保留numpy数组的dtype")
            self.assertFalse(shares_memory(t, arr), "tensor(numpy数组)不应共享内存")
            arr[0] = 42.0
            self.assertEqual(float(t.array()[0]), 1.0, "tensor(numpy数组)应独立拷贝")
        self._run_subcase(doc, "tensor-复制内存", _tensor_copy)

        def _tensor_dtype_diff():
            arr = np.array([1.0, 2.0], dtype=np.float32)
            t = rm.tensor(arr, dtype=rm.float64, device='cpu')
            self.assertEqual(t.dtype, rm.float64, "tensor指定dtype应生效")
            self.assertFalse(shares_memory(t, arr), "dtype不一致时tensor不应共享内存")
        self._run_subcase(doc, "tensor-dtype不一致", _tensor_dtype_diff)

        def _as_tensor_shared():
            arr = np.array([1.0, 2.0], dtype=np.float32)
            t = rm.as_tensor(arr, device='cpu')
            self.assertEqual(t.device.type, 'cpu')
            self.assertEqual(t.dtype, rm.float32, "as_tensor应保留numpy数组的dtype")
            self.assertTrue(shares_memory(t, arr), "同dtype同设备时as_tensor(numpy数组)应共享内存")
            arr[0] = 42.0
            self.assertEqual(float(t.array()[0]), 42.0, "as_tensor(numpy数组)应与源数组共享内存")
        self._run_subcase(doc, "as_tensor-共享内存", _as_tensor_shared)

        def _as_tensor_dtype_diff():
            arr = np.array([1.0, 2.0], dtype=np.float32)
            t = rm.as_tensor(arr, dtype=rm.float64, device='cpu')
            self.assertEqual(t.dtype, rm.float64, "as_tensor指定dtype应生效")
            self.assertFalse(shares_memory(t, arr), "dtype不一致时as_tensor(numpy数组)不应共享内存")
        self._run_subcase(doc, "as_tensor-dtype不一致则复制", _as_tensor_dtype_diff)

        def _as_tensor_int():
            arr = np.array([1, 2, 3], dtype=np.int32)
            t = rm.as_tensor(arr, device='cpu')
            self.assertEqual(t.dtype, rm.int32, "as_tensor应保留int32 dtype")
            self.assertTrue(shares_memory(t, arr), "as_tensor(int32 numpy数组)应共享内存")
        self._run_subcase(doc, "as_tensor-int32共享内存", _as_tensor_int)

        def _from_dlpack_shared():
            arr = np.array([1.0, 2.0], dtype=np.float32)
            t = rm.from_dlpack(arr)
            self.assertEqual(t.device.type, 'cpu')
            self.assertEqual(t.dtype, rm.float32, "from_dlpack应保留numpy数组的dtype")
            self.assertTrue(shares_memory(t, arr), "from_dlpack(numpy数组)应共享内存")
            arr[1] = 7.0
            self.assertEqual(float(t.array()[1]), 7.0, "from_dlpack(numpy数组)应与源数组共享内存")
        self._run_subcase(doc, "from_dlpack-共享内存", _from_dlpack_shared)

        def _from_dlpack_float64():
            arr = np.array([1.0, 2.0], dtype=np.float64)
            t = rm.from_dlpack(arr)
            self.assertEqual(t.dtype, rm.float64, "from_dlpack应保留float64 dtype")
            self.assertTrue(shares_memory(t, arr), "from_dlpack(float64 numpy数组)应共享内存")
        self._run_subcase(doc, "from_dlpack-float64共享内存", _from_dlpack_float64)

    # ------------------------------------------------------------------
    # torch CPU张量 -> Riemann张量
    # ------------------------------------------------------------------
    def test_from_torch_cpu(self):
        """torch CPU张量转Riemann张量"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        doc = self.test_from_torch_cpu.__doc__

        def _tensor_copy():
            src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            t = rm.tensor(src, device='cpu')
            self.assertEqual(t.device.type, 'cpu')
            self.assertEqual(t.dtype, rm.float32)
            self.assertFalse(shares_memory(t, src), "tensor(torch张量)不应共享内存")
            src[0] = 42.0
            self.assertEqual(float(t.array()[0]), 1.0, "tensor(torch张量)应独立拷贝")
        self._run_subcase(doc, "tensor-复制内存", _tensor_copy)

        def _tensor_dtype_diff():
            src = torch.tensor([1.0, 2.0], dtype=torch.float32)
            t = rm.tensor(src, dtype=rm.float64, device='cpu')
            self.assertEqual(t.dtype, rm.float64, "tensor指定dtype应生效")
            self.assertFalse(shares_memory(t, src), "dtype不一致时tensor不应共享内存")
        self._run_subcase(doc, "tensor-dtype不一致", _tensor_dtype_diff)

        def _as_tensor_shared():
            src = torch.tensor([1.0, 2.0], dtype=torch.float32)
            t = rm.as_tensor(src, device='cpu')
            self.assertEqual(t.device.type, 'cpu')
            self.assertEqual(t.dtype, rm.float32)
            self.assertTrue(shares_memory(t, src), "同dtype同设备时as_tensor(torch张量)应共享内存")
            src[0] = 42.0
            self.assertEqual(float(t.array()[0]), 42.0, "as_tensor(torch张量)应与源张量共享内存")
        self._run_subcase(doc, "as_tensor-共享内存", _as_tensor_shared)

        def _as_tensor_dtype_diff():
            src = torch.tensor([1.0, 2.0], dtype=torch.float32)
            t = rm.as_tensor(src, dtype=rm.float64, device='cpu')
            self.assertEqual(t.dtype, rm.float64)
            self.assertFalse(shares_memory(t, src), "dtype不一致时as_tensor(torch张量)不应共享内存")
        self._run_subcase(doc, "as_tensor-dtype不一致则复制", _as_tensor_dtype_diff)

        def _from_dlpack_shared():
            src = torch.tensor([1.0, 2.0], dtype=torch.float32)
            t = rm.from_dlpack(src)
            self.assertEqual(t.device.type, 'cpu')
            self.assertEqual(t.dtype, rm.float32)
            self.assertTrue(shares_memory(t, src), "from_dlpack(torch CPU张量)应共享内存")
        self._run_subcase(doc, "from_dlpack-共享内存", _from_dlpack_shared)

        def _from_dlpack_float64():
            src = torch.tensor([1.0, 2.0], dtype=torch.float64)
            t = rm.from_dlpack(src)
            self.assertEqual(t.dtype, rm.float64, "from_dlpack应保留torch张量的dtype")
            self.assertTrue(shares_memory(t, src), "from_dlpack(float64 torch张量)应共享内存")
        self._run_subcase(doc, "from_dlpack-dtype保留", _from_dlpack_float64)

    # ------------------------------------------------------------------
    # torch CUDA张量 -> Riemann张量
    # ------------------------------------------------------------------
    def test_from_torch_cuda(self):
        """torch CUDA张量转Riemann张量"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA不可用")
        doc = self.test_from_torch_cuda.__doc__

        def _tensor_copy():
            src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cuda:0')
            t = rm.tensor(src)
            self.assertEqual(t.device.type, 'cuda', "tensor应沿用torch CUDA张量的设备")
            self.assertEqual(t.device.index, 0)
            self.assertEqual(t.dtype, rm.float32)
            self.assertFalse(shares_memory(t, src), "tensor(torch CUDA张量)不应共享内存")
            src[0] = 42.0
            self.assertEqual(float(_values(t)[0]), 1.0, "tensor(torch CUDA张量)应独立拷贝")
        self._run_subcase(doc, "tensor-复制内存且保持设备", _tensor_copy)

        def _as_tensor_shared():
            src = torch.tensor([1.0, 2.0], dtype=torch.float32, device='cuda:0')
            t = rm.as_tensor(src)
            self.assertEqual(t.device.type, 'cuda', "as_tensor应沿用torch CUDA张量的设备")
            self.assertEqual(t.dtype, rm.float32)
            self.assertTrue(shares_memory(t, src), "同设备同dtype时as_tensor(torch CUDA张量)应共享显存")
        self._run_subcase(doc, "as_tensor-共享内存", _as_tensor_shared)

        def _as_tensor_dtype_diff():
            src = torch.tensor([1.0, 2.0], dtype=torch.float32, device='cuda:0')
            t = rm.as_tensor(src, dtype=rm.float64)
            self.assertEqual(t.dtype, rm.float64)
            self.assertEqual(t.device.type, 'cuda')
            self.assertFalse(shares_memory(t, src), "dtype不一致时as_tensor(torch CUDA张量)不应共享内存")
        self._run_subcase(doc, "as_tensor-dtype不一致则复制", _as_tensor_dtype_diff)

        def _from_dlpack_shared():
            src = torch.tensor([1.0, 2.0], dtype=torch.float32, device='cuda:0')
            t = rm.from_dlpack(src)
            self.assertEqual(t.device.type, 'cuda', "from_dlpack应沿用torch CUDA张量的设备")
            self.assertEqual(t.dtype, rm.float32)
            self.assertTrue(shares_memory(t, src), "from_dlpack(torch CUDA张量)应共享显存")
        self._run_subcase(doc, "from_dlpack-共享内存", _from_dlpack_shared)

        def _from_dlpack_float64():
            src = torch.tensor([1.0, 2.0], dtype=torch.float64, device='cuda:0')
            t = rm.from_dlpack(src)
            self.assertEqual(t.dtype, rm.float64, "from_dlpack应保留torch CUDA张量的dtype")
            self.assertTrue(shares_memory(t, src), "from_dlpack(float64 torch CUDA张量)应共享显存")
        self._run_subcase(doc, "from_dlpack-dtype保留", _from_dlpack_float64)

    # ------------------------------------------------------------------
    # cupy数组 -> Riemann张量
    # ------------------------------------------------------------------
    def test_from_cupy(self):
        """cupy数组转Riemann张量"""
        if not CUPY_AVAILABLE:
            self.skipTest("CuPy不可用")
        doc = self.test_from_cupy.__doc__

        def _tensor_copy():
            arr = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
            t = rm.tensor(arr)
            self.assertEqual(t.device.type, 'cuda', "tensor应沿用cupy数组的设备")
            self.assertEqual(t.dtype, rm.float32)
            self.assertFalse(shares_memory(t, arr), "tensor(cupy数组)不应共享显存")
            arr[0] = 42.0
            self.assertEqual(float(_values(t)[0]), 1.0, "tensor(cupy数组)应独立拷贝")
        self._run_subcase(doc, "tensor-复制内存", _tensor_copy)

        def _as_tensor_shared():
            arr = cp.array([1.0, 2.0], dtype=cp.float32)
            t = rm.as_tensor(arr)
            self.assertEqual(t.device.type, 'cuda')
            self.assertEqual(t.dtype, rm.float32)
            self.assertTrue(shares_memory(t, arr), "同设备同dtype时as_tensor(cupy数组)应共享显存")
        self._run_subcase(doc, "as_tensor-共享内存", _as_tensor_shared)

        def _as_tensor_dtype_diff():
            arr = cp.array([1.0, 2.0], dtype=cp.float32)
            t = rm.as_tensor(arr, dtype=rm.float64)
            self.assertEqual(t.dtype, rm.float64)
            self.assertEqual(t.device.type, 'cuda')
            self.assertFalse(shares_memory(t, arr), "dtype不一致时as_tensor(cupy数组)不应共享显存")
        self._run_subcase(doc, "as_tensor-dtype不一致则复制", _as_tensor_dtype_diff)

        def _from_dlpack_shared():
            arr = cp.array([1.0, 2.0], dtype=cp.float32)
            t = rm.from_dlpack(arr)
            self.assertEqual(t.device.type, 'cuda', "from_dlpack应沿用cupy数组的设备")
            self.assertEqual(t.dtype, rm.float32)
            self.assertTrue(shares_memory(t, arr), "from_dlpack(cupy数组)应共享显存")
        self._run_subcase(doc, "from_dlpack-共享内存", _from_dlpack_shared)

        def _from_dlpack_float64():
            arr = cp.array([1.0, 2.0], dtype=cp.float64)
            t = rm.from_dlpack(arr)
            self.assertEqual(t.dtype, rm.float64, "from_dlpack应保留cupy数组的dtype")
            self.assertTrue(shares_memory(t, arr), "from_dlpack(float64 cupy数组)应共享显存")
        self._run_subcase(doc, "from_dlpack-dtype保留", _from_dlpack_float64)

    # ------------------------------------------------------------------
    # Riemann CPU张量 -> torch张量
    # ------------------------------------------------------------------
    def test_to_torch_cpu(self):
        """Riemann CPU张量转torch张量"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        doc = self.test_to_torch_cpu.__doc__

        def _torch_tensor_copy():
            t = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float32)
            out = torch.tensor(t)
            self.assertEqual(out.device.type, 'cpu')
            self.assertEqual(str(out.dtype), 'torch.float32')
            self.assertFalse(shares_memory(out, t), "torch.tensor(Riemann张量)不应共享内存")
            t[0] = 42.0
            self.assertEqual(float(out[0]), 1.0, "torch.tensor(Riemann张量)应独立拷贝")
        self._run_subcase(doc, "tensor-复制内存", _torch_tensor_copy)

        def _torch_tensor_dtype_diff():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32)
            out = torch.tensor(t, dtype=torch.float64)
            self.assertEqual(str(out.dtype), 'torch.float64', "torch.tensor指定dtype应生效")
            self.assertFalse(shares_memory(out, t), "dtype不一致时torch.tensor不应共享内存")
        self._run_subcase(doc, "tensor-dtype不一致", _torch_tensor_dtype_diff)

        def _torch_as_tensor_shared():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32)
            out = torch.as_tensor(t)
            self.assertEqual(out.device.type, 'cpu')
            self.assertEqual(str(out.dtype), 'torch.float32')
            self.assertTrue(shares_memory(out, t), "同dtype时torch.as_tensor(Riemann张量)应共享内存")
            t[0] = 42.0
            self.assertEqual(float(out[0]), 42.0, "torch.as_tensor(Riemann张量)应与张量共享内存")
        self._run_subcase(doc, "as_tensor-共享内存", _torch_as_tensor_shared)

        def _torch_as_tensor_dtype_diff():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32)
            out = torch.as_tensor(t, dtype=torch.float64)
            self.assertEqual(str(out.dtype), 'torch.float64')
            self.assertFalse(shares_memory(out, t), "dtype不一致时torch.as_tensor不应共享内存")
        self._run_subcase(doc, "as_tensor-dtype不一致则复制", _torch_as_tensor_dtype_diff)

        def _torch_from_dlpack_shared():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32)
            out = torch.from_dlpack(t)
            self.assertEqual(out.device.type, 'cpu')
            self.assertEqual(str(out.dtype), 'torch.float32')
            self.assertTrue(shares_memory(out, t), "torch.from_dlpack(Riemann张量)应共享内存")
        self._run_subcase(doc, "from_dlpack-共享内存", _torch_from_dlpack_shared)

        def _torch_from_dlpack_float64():
            t = rm.tensor([1.0, 2.0], dtype=rm.float64)
            out = torch.from_dlpack(t)
            self.assertEqual(str(out.dtype), 'torch.float64', "torch.from_dlpack应保留dtype")
            self.assertTrue(shares_memory(out, t), "torch.from_dlpack(float64张量)应共享内存")
        self._run_subcase(doc, "from_dlpack-dtype保留", _torch_from_dlpack_float64)

    # ------------------------------------------------------------------
    # Riemann CUDA张量 -> torch张量
    # ------------------------------------------------------------------
    def test_to_torch_cuda(self):
        """Riemann CUDA张量转torch张量"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA不可用")
        doc = self.test_to_torch_cuda.__doc__

        def _torch_tensor_copy():
            t = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float32, device='cuda:0')
            out = torch.tensor(t)
            self.assertEqual(out.device.type, 'cuda', "torch.tensor应保持张量所在设备")
            self.assertEqual(out.device.index, 0)
            self.assertEqual(str(out.dtype), 'torch.float32')
            self.assertFalse(shares_memory(out, t), "torch.tensor(Riemann CUDA张量)不应共享显存")
        self._run_subcase(doc, "tensor-复制内存且保持设备", _torch_tensor_copy)

        def _torch_as_tensor_shared():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda:0')
            out = torch.as_tensor(t)
            self.assertEqual(out.device.type, 'cuda')
            self.assertEqual(str(out.dtype), 'torch.float32')
            self.assertTrue(shares_memory(out, t), "同设备同dtype时torch.as_tensor(Riemann CUDA张量)应共享显存")
        self._run_subcase(doc, "as_tensor-共享内存", _torch_as_tensor_shared)

        def _torch_as_tensor_dtype_diff():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda:0')
            out = torch.as_tensor(t, dtype=torch.float64)
            self.assertEqual(str(out.dtype), 'torch.float64')
            self.assertFalse(shares_memory(out, t), "dtype不一致时torch.as_tensor不应共享显存")
        self._run_subcase(doc, "as_tensor-dtype不一致则复制", _torch_as_tensor_dtype_diff)

        def _torch_from_dlpack_shared():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda:0')
            out = torch.from_dlpack(t)
            self.assertEqual(out.device.type, 'cuda', "torch.from_dlpack应保持张量所在设备")
            self.assertEqual(out.device.index, 0)
            self.assertEqual(str(out.dtype), 'torch.float32')
            self.assertTrue(shares_memory(out, t), "torch.from_dlpack(Riemann CUDA张量)应共享显存")
        self._run_subcase(doc, "from_dlpack-共享内存", _torch_from_dlpack_shared)

    # ------------------------------------------------------------------
    # Riemann张量 -> numpy数组
    # ------------------------------------------------------------------
    def test_to_numpy(self):
        """Riemann张量转numpy数组"""
        doc = self.test_to_numpy.__doc__

        def _np_array_copy():
            t = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float32)
            out = np.array(t)
            self.assertIsInstance(out, np.ndarray, "np.array(TN)应返回numpy数组")
            self.assertEqual(out.dtype, np.float32)
            self.assertFalse(shares_memory(out, t), "np.array(CPU张量)不应共享内存")
            t[0] = 42.0
            self.assertEqual(float(out[0]), 1.0, "np.array(CPU张量)应独立拷贝")
        self._run_subcase(doc, "array-CPU张量复制内存", _np_array_copy)

        def _np_array_dtype_diff():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32)
            out = np.array(t, dtype=np.float64)
            self.assertEqual(out.dtype, np.float64, "np.array指定dtype应生效")
            self.assertFalse(shares_memory(out, t), "dtype不一致时np.array不应共享内存")
        self._run_subcase(doc, "array-dtype不一致", _np_array_dtype_diff)

        def _np_asarray_shared():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32)
            out = np.asarray(t)
            self.assertIsInstance(out, np.ndarray, "np.asarray(TN)应返回numpy数组")
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(shares_memory(out, t), "np.asarray(CPU张量)应共享内存")
            out[0] = 42.0
            self.assertEqual(float(t.array()[0]), 42.0, "np.asarray(CPU张量)应与张量共享内存")
        self._run_subcase(doc, "asarray-CPU张量共享内存", _np_asarray_shared)

        def _np_asarray_dtype_diff():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32)
            out = np.asarray(t, dtype=np.float64)
            self.assertEqual(out.dtype, np.float64)
            self.assertFalse(shares_memory(out, t), "dtype不一致时np.asarray不应共享内存")
        self._run_subcase(doc, "asarray-dtype不一致则复制", _np_asarray_dtype_diff)

        if CUDA_AVAILABLE:
            def _np_array_cuda():
                t = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda:0')
                out = np.array(t)
                self.assertIsInstance(out, np.ndarray, "np.array(CUDA张量)应返回CPU端numpy数组")
                self.assertEqual(out.dtype, np.float32, "np.array(CUDA张量)应保留dtype")
                self.assertTrue(np.allclose(out, np.array([1.0, 2.0], dtype=np.float32)))
                self.assertFalse(shares_memory(out, t), "np.array(CUDA张量)是拉回CPU的副本，不应共享")
            self._run_subcase(doc, "array-CUDA张量拉回CPU", _np_array_cuda)

    # ------------------------------------------------------------------
    # Riemann CUDA张量 -> cupy数组
    # ------------------------------------------------------------------
    def test_to_cupy(self):
        """Riemann CUDA张量转cupy数组"""
        if not CUPY_AVAILABLE:
            self.skipTest("CuPy不可用")
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA不可用")
        doc = self.test_to_cupy.__doc__

        def _cp_array_copy():
            t = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float32, device='cuda:0')
            out = cp.array(t)
            self.assertIsInstance(out, cp.ndarray, "cp.array(TN)应返回cupy数组")
            self.assertEqual(out.dtype, cp.float32)
            self.assertFalse(shares_memory(out, t), "cp.array(CUDA张量)不应共享显存")
            out[0] = 42.0
            self.assertEqual(float(_values(t)[0]), 1.0, "cp.array(CUDA张量)应独立拷贝")
        self._run_subcase(doc, "array-CUDA张量复制内存", _cp_array_copy)

        def _cp_array_dtype_diff():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda:0')
            out = cp.array(t, dtype=cp.float64)
            self.assertEqual(out.dtype, cp.float64, "cp.array指定dtype应生效")
            self.assertFalse(shares_memory(out, t), "dtype不一致时cp.array不应共享显存")
        self._run_subcase(doc, "array-dtype不一致", _cp_array_dtype_diff)

        def _cp_asarray_shared():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda:0')
            out = cp.asarray(t)
            self.assertIsInstance(out, cp.ndarray, "cp.asarray(TN)应返回cupy数组")
            self.assertEqual(out.dtype, cp.float32)
            self.assertTrue(shares_memory(out, t), "cp.asarray(CUDA张量)应共享显存")
            out[0] = 42.0
            self.assertEqual(float(_values(t)[0]), 42.0, "cp.asarray(CUDA张量)应与张量共享显存")
        self._run_subcase(doc, "asarray-CUDA张量共享内存", _cp_asarray_shared)

        def _cp_asarray_dtype_diff():
            t = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda:0')
            out = cp.asarray(t, dtype=cp.float64)
            self.assertEqual(out.dtype, cp.float64)
            self.assertFalse(shares_memory(out, t), "dtype不一致时cp.asarray不应共享显存")
        self._run_subcase(doc, "asarray-dtype不一致则复制", _cp_asarray_dtype_diff)


    # ------------------------------------------------------------------
    # buffer接口协议（memoryview / bytes / np.frombuffer）
    # ------------------------------------------------------------------
    def test_buffer_protocol(self):
        """Riemann张量buffer接口协议测试"""
        doc = self.test_buffer_protocol.__doc__

        def _memview_basic():
            t = rm.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=rm.float32)
            mv = memoryview(t)
            self.assertEqual(mv.format, 'f', "float32张量的memoryview格式应为'f'")
            self.assertEqual(mv.itemsize, 4, "float32张量元素的字节数应为4")
            self.assertEqual(mv.ndim, 2, "memoryview应保留张量的维度数")
            self.assertEqual(tuple(mv.shape), (2, 2), "memoryview应保留张量的形状")
            self.assertFalse(mv.readonly, "CPU张量的memoryview应可写")
        self._run_subcase(doc, "memoryview-格式与形状", _memview_basic)

        def _memview_shared_write():
            t = rm.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=rm.float32)
            mv = memoryview(t)
            self.assertTrue(shares_memory(np.asarray(mv), t), "CPU张量的buffer应与张量共享内存")
            mv[0, 0] = 42.0
            self.assertEqual(float(t.array()[0, 0]), 42.0, "通过memoryview写入应反映到张量上")
        self._run_subcase(doc, "memoryview-共享内存且可写", _memview_shared_write)

        def _memview_dtype_formats():
            self.assertEqual(memoryview(rm.tensor([1, 2], dtype=rm.float64)).format, 'd',
                             "float64张量的memoryview格式应为'd'")
            self.assertEqual(memoryview(rm.tensor([1, 2], dtype=rm.complex64)).format, 'Zf',
                             "complex64张量的memoryview格式应为'Zf'")
            self.assertEqual(memoryview(rm.tensor([1, 2], dtype=rm.int32)).itemsize, 4,
                             "int32张量元素的字节数应为4")
            self.assertEqual(memoryview(rm.tensor([1, 2], dtype=rm.int64)).itemsize, 8,
                             "int64张量元素的字节数应为8")
        self._run_subcase(doc, "memoryview-dtype格式映射", _memview_dtype_formats)

        def _memview_scalar():
            t = rm.tensor(3.5)
            mv = memoryview(t)
            self.assertEqual(mv.ndim, 0, "标量张量的memoryview应为0维")
            self.assertEqual(tuple(mv.shape), (), "标量张量的memoryview形状应为空元组")
            self.assertEqual(float(np.asarray(mv)), 3.5, "标量张量buffer的值应正确")
        self._run_subcase(doc, "memoryview-标量张量", _memview_scalar)

        def _bytes_export():
            t = rm.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=rm.float32)
            data = bytes(t)
            self.assertEqual(len(data), 16, "2x2 float32张量应导出16字节")
            self.assertEqual(data, t.array().tobytes(), "bytes(张量)应按C顺序导出原始字节")
        self._run_subcase(doc, "bytes-按C顺序导出", _bytes_export)

        def _np_frombuffer_view():
            t = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float32)
            view = np.frombuffer(memoryview(t), dtype=np.float32)
            self.assertTrue(shares_memory(view, t), "np.frombuffer(memoryview(张量))应为零拷贝视图")
            self.assertTrue(np.array_equal(view, np.array([1.0, 2.0, 3.0], dtype=np.float32)))
            view[1] = 7.0
            self.assertEqual(float(t.array()[1]), 7.0, "通过np.frombuffer视图写入应反映到张量上")
        self._run_subcase(doc, "np.frombuffer-零拷贝视图", _np_frombuffer_view)

        def _memview_requires_grad():
            t = rm.tensor([1.0, 2.0], requires_grad=True)
            mv = memoryview(t)
            self.assertTrue(np.array_equal(np.asarray(mv), np.array([1.0, 2.0], dtype=np.float32)),
                            "requires_grad张量的buffer应可读且值正确")
        self._run_subcase(doc, "memoryview-requires_grad张量可读", _memview_requires_grad)

        if CUDA_AVAILABLE:
            def _memview_cuda():
                t = rm.tensor([[1.0, 2.0]], dtype=rm.float32, device='cuda:0')
                mv = memoryview(t)
                self.assertEqual(tuple(mv.shape), (1, 2), "CUDA张量的buffer应保留形状")
                self.assertTrue(np.array_equal(np.asarray(mv), np.array([[1.0, 2.0]], dtype=np.float32)),
                                "CUDA张量的buffer应返回正确的CPU副本数据")
                mv[0, 0] = 9.0
                self.assertEqual(float(_values(t)[0, 0]), 1.0,
                                 "CUDA张量的buffer是拉回CPU的副本，写入不应影响张量")
            self._run_subcase(doc, "memoryview-CUDA张量返回CPU副本", _memview_cuda)


# 如果作为独立脚本运行
if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True

    # 清屏
    clear_screen()

    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行张量互转测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs NumPy / CuPy / PyTorch{Colors.ENDC}")
    print(f"{Colors.OKBLUE}CUDA可用: {CUDA_AVAILABLE}, CuPy可用: {CUPY_AVAILABLE}, PyTorch可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print()

    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTensorInterop)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(test_suite)

    # 打印测试统计摘要
    stats.print_summary()

    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)
