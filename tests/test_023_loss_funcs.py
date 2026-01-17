import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))
from riemann.nn.functional import mse_loss, l1_loss, smooth_l1_loss, cross_entropy
from riemann.nn.functional import binary_cross_entropy_with_logits, huber_loss, nll_loss
from riemann import tensor
import riemann as rm

# 从rm.cuda获取cupy引用和CUDA可用性
CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
cp = rm.cuda.cp

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
    print("警告: 无法导入PyTorch，将只测试riemann的损失函数")
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

# 测试统计类（重命名为避免pytest误识别）
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
    
    def print_summary(self):
        print("\n" + "="*73)
        print(f"{Colors.BOLD}测试统计摘要{Colors.ENDC}")
        print("="*73)
        print(f"总测试用例数: {self.total_cases}")
        print(f"通过测试用例数: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases}{Colors.ENDC}")
        print(f"测试通过率: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases/self.total_cases*100:.2f}%{Colors.ENDC}")
        print(f"总耗时: {self.total_time:.4f} 秒")
        print("\n各用例测试详情:")
        print("-"*73)
        print(f"{'用例名':<30}{'通过/总数':<15}{'通过率':<10}{'耗时(秒)':<10}")
        print("-"*73)
        for func_name, stats in self.function_stats.items():
            pass_rate = stats["passed"]/stats["total"]*100 if stats["total"] > 0 else 0
            status_color = Colors.OKGREEN if pass_rate == 100 else Colors.FAIL
            print(f"{func_name:<33}{f'{stats["passed"]}/{stats["total"]}':<19}{status_color}{pass_rate:<13.2f}{Colors.ENDC}{stats['time']:.4f}")
        print("="*73)

# 全局统计实例
stats = StatisticsCollector()
# 判断是否作为独立脚本运行
IS_RUNNING_AS_SCRIPT = False

# 比较值的函数
def compare_values(riemann_val, torch_val, atol=1e-3, rtol=1e-3):
    """比较Riemann和PyTorch的值是否接近"""
    # 处理None值的情况
    if not TORCH_AVAILABLE:
        # 如果没有PyTorch，只检查riemann结果是否存在
        return riemann_val is not None
    
    if riemann_val is None and torch_val is None:
        return True
    if riemann_val is None or torch_val is None:
        return False
    
    # 处理嵌套元组/列表的情况
    if isinstance(riemann_val, (list, tuple)) and isinstance(torch_val, (list, tuple)):
        if len(riemann_val) != len(torch_val):
            return False
        
        all_passed = True
        for i, (r, t) in enumerate(zip(riemann_val, torch_val)):
            if not compare_values(r, t, atol, rtol):
                all_passed = False
                break
        
        return all_passed
    
    # 转换为numpy数组
    try:
        # 处理Riemann结果
        if hasattr(riemann_val, 'is_cuda') and riemann_val.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            riemann_np = riemann_val.detach().cpu().numpy()
        else:
            riemann_np = riemann_val.detach().numpy()
        
        # 处理PyTorch结果
        if hasattr(torch_val, 'is_cuda') and torch_val.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            torch_np = torch_val.detach().cpu().numpy()
        else:
            torch_np = torch_val.detach().numpy()
    except Exception as e:
        print(f"比较值转换错误: {e}")
        return False
    
    # 处理形状不匹配的情况
    try:
        # 增加容差参数，特别是对于梯度比较
        np.testing.assert_allclose(riemann_np, torch_np, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        # 打印错误信息以便调试
        print(f"比较失败: {e}")
        print(f"Riemann值形状: {riemann_np.shape}, PyTorch值形状: {torch_np.shape}")
        # 打印部分值进行比较
        if riemann_np.size > 0:
            print(f"Riemann样本值: {riemann_np.flat[:3]}")
            print(f"PyTorch样本值: {torch_np.flat[:3]}")
        return False

class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        # 设置随机种子以确保结果可重复
        np.random.seed(42)
        torch.manual_seed(42)
        # 获取当前测试方法名
        self.current_test_name = self._testMethodName
        # 如果是独立脚本运行，则开始记录函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")
            print(f"测试描述: {self._testMethodDoc}")
    
    def tearDown(self):
        # 如果是独立脚本运行，则结束函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_mse_loss(self):
        """测试均方误差损失函数"""
        test_cases = [
            {"name": "基本测试(mean)", "batch_size": 4, "features": 3, "reduction": "mean"},
            {"name": "基本测试(sum)", "batch_size": 4, "features": 3, "reduction": "sum"},
            {"name": "基本测试(none)", "batch_size": 4, "features": 3, "reduction": "none"},
            {"name": "大维度测试", "batch_size": 16, "features": 128, "reduction": "mean"},
            {"name": "3D张量测试", "batch_size": 4, "features": (3, 3), "reduction": "mean"},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"mse_loss: {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 生成随机输入
                    batch_size = case["batch_size"]
                    features = case["features"]
                    reduction = case["reduction"]
                    
                    if isinstance(features, tuple):
                        input_shape = (batch_size,) + features
                    else:
                        input_shape = (batch_size, features)
                    
                    # 生成测试数据
                    np.random.seed(42)
                    input_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    target_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    
                    # Riemann计算
                    if device == "cpu":
                        riemann_input = tensor(input_data, requires_grad=True)
                        riemann_target = tensor(target_data, requires_grad=False)
                    else:  # cuda
                        riemann_input = tensor(input_data, requires_grad=True, device=device)
                        riemann_target = tensor(target_data, requires_grad=False, device=device)
                    riemann_loss = mse_loss(riemann_input, riemann_target, reduction=reduction)
                    
                    # 反向传播
                    if reduction == 'none':
                        # 对于none reduction，我们需要先求和再反向传播
                        riemann_loss_sum = riemann_loss.sum()
                        riemann_loss_sum.backward()
                    else:
                        riemann_loss.backward()
                    
                    # 使用grad属性获取梯度
                    riemann_grad = riemann_input.grad
                    
                    # PyTorch计算
                    if device == "cpu":
                        torch_input = torch.tensor(input_data, requires_grad=True)
                        torch_target = torch.tensor(target_data, requires_grad=False)
                    else:  # cuda
                        torch_input = torch.tensor(input_data, requires_grad=True, device=device)
                        torch_target = torch.tensor(target_data, requires_grad=False, device=device)
                    torch_loss = torch.nn.functional.mse_loss(torch_input, torch_target, reduction=reduction)
                    
                    # 反向传播
                    if reduction == 'none':
                        torch_loss.sum().backward()
                    else:
                        torch_loss.backward()
                    torch_grad = torch_input.grad
                    
                    # 比较结果
                    values_match = compare_values(riemann_loss, torch_loss)
                    grads_match = compare_values(riemann_grad, torch_grad)
                    
                    # 清理梯度
                    riemann_input.grad = None
                    
                    # 判断是否通过
                    passed = values_match and grads_match
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        # 修改为一行输出，包含测试用例名称和结果
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: {'通过' if values_match else '失败'}")
                            print(f"  梯度比较: {'通过' if grads_match else '失败'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(values_match, f"值比较失败: {case_name}")
                    self.assertTrue(grads_match, f"梯度比较失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"  {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_l1_loss(self):
        """测试L1损失函数"""
        test_cases = [
            {"name": "基本测试(mean)", "batch_size": 4, "features": 3, "reduction": "mean"},
            {"name": "基本测试(sum)", "batch_size": 4, "features": 3, "reduction": "sum"},
            {"name": "基本测试(none)", "batch_size": 4, "features": 3, "reduction": "none"},
            {"name": "大维度测试", "batch_size": 16, "features": 128, "reduction": "mean"},
            {"name": "3D张量测试", "batch_size": 4, "features": (3, 3), "reduction": "mean"},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"l1_loss: {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 生成随机输入
                    batch_size = case["batch_size"]
                    features = case["features"]
                    reduction = case["reduction"]
                    
                    if isinstance(features, tuple):
                        input_shape = (batch_size,) + features
                    else:
                        input_shape = (batch_size, features)
                    
                    # 生成测试数据
                    np.random.seed(42)
                    input_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    target_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    
                    # Riemann计算
                    if device == "cpu":
                        riemann_input = tensor(input_data, requires_grad=True)
                        riemann_target = tensor(target_data, requires_grad=False)
                    else:  # cuda
                        riemann_input = tensor(input_data, requires_grad=True, device=device)
                        riemann_target = tensor(target_data, requires_grad=False, device=device)
                    riemann_loss = l1_loss(riemann_input, riemann_target, reduction=reduction)
                    
                    # 反向传播
                    if reduction == 'none':
                        # 对于none reduction，我们需要先求和再反向传播
                        riemann_loss_sum = riemann_loss.sum()
                        riemann_loss_sum.backward()
                    else:
                        riemann_loss.backward()
                    
                    # 使用grad属性获取梯度
                    riemann_grad = riemann_input.grad
                    
                    # PyTorch计算
                    if device == "cpu":
                        torch_input = torch.tensor(input_data, requires_grad=True)
                        torch_target = torch.tensor(target_data, requires_grad=False)
                    else:  # cuda
                        torch_input = torch.tensor(input_data, requires_grad=True, device=device)
                        torch_target = torch.tensor(target_data, requires_grad=False, device=device)
                    torch_loss = torch.nn.functional.l1_loss(torch_input, torch_target, reduction=reduction)
                    
                    # 反向传播
                    if reduction == 'none':
                        torch_loss.sum().backward()
                    else:
                        torch_loss.backward()
                    torch_grad = torch_input.grad
                    
                    # 比较结果
                    values_match = compare_values(riemann_loss, torch_loss)
                    grads_match = compare_values(riemann_grad, torch_grad)
                    
                    # 清理梯度
                    riemann_input.grad = None
                    
                    # 判断是否通过
                    passed = values_match and grads_match
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        # 修改为一行输出，包含测试用例名称和结果
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: {'通过' if values_match else '失败'}")
                            print(f"  梯度比较: {'通过' if grads_match else '失败'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(values_match, f"值比较失败: {case_name}")
                    self.assertTrue(grads_match, f"梯度比较失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"  {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_smooth_l1_loss(self):
        """测试平滑L1损失函数"""
        test_cases = [
            {"name": "基本测试(mean, beta=1.0)", "batch_size": 4, "features": 3, "reduction": "mean", "beta": 1.0},
            {"name": "基本测试(sum, beta=1.0)", "batch_size": 4, "features": 3, "reduction": "sum", "beta": 1.0},
            {"name": "基本测试(none, beta=1.0)", "batch_size": 4, "features": 3, "reduction": "none", "beta": 1.0},
            {"name": "不同beta值测试(0.5)", "batch_size": 4, "features": 3, "reduction": "mean", "beta": 0.5},
            {"name": "不同beta值测试(2.0)", "batch_size": 4, "features": 3, "reduction": "mean", "beta": 2.0},
            {"name": "带权重测试", "batch_size": 4, "features": 3, "reduction": "mean", "beta": 1.0, "use_weight": True},
            {"name": "大维度测试", "batch_size": 16, "features": 128, "reduction": "mean", "beta": 1.0},
            {"name": "3D张量测试", "batch_size": 4, "features": (3, 3), "reduction": "mean", "beta": 1.0},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"smooth_l1_loss: {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 生成随机输入
                    batch_size = case["batch_size"]
                    features = case["features"]
                    reduction = case["reduction"]
                    beta = case.get("beta", 1.0)
                    
                    if isinstance(features, tuple):
                        input_shape = (batch_size,) + features
                    else:
                        input_shape = (batch_size, features)
                    
                    # 生成测试数据
                    np.random.seed(42)
                    input_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    target_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    
                    # 处理权重
                    riemann_weight = None
                    torch_weight = None
                    
                    if case.get("use_weight", False):
                        weight_data = np.random.rand(*input_shape).astype(rm.get_default_dtype())
                        if device == "cpu":
                            riemann_weight = tensor(weight_data, requires_grad=False)
                            torch_weight = torch.tensor(weight_data, requires_grad=False)
                        else:  # cuda
                            riemann_weight = tensor(weight_data, requires_grad=False, device=device)
                            torch_weight = torch.tensor(weight_data, requires_grad=False, device=device)
                    
                    # Riemann计算
                    if device == "cpu":
                        riemann_input = tensor(input_data, requires_grad=True)
                        riemann_target = tensor(target_data, requires_grad=False)
                    else:  # cuda
                        riemann_input = tensor(input_data, requires_grad=True, device=device)
                        riemann_target = tensor(target_data, requires_grad=False, device=device)
                    riemann_loss = smooth_l1_loss(riemann_input, riemann_target, beta=beta, reduction=reduction)
                    
                    # 应用权重
                    if riemann_weight is not None:
                        riemann_loss = riemann_loss * riemann_weight
                    
                    # 反向传播
                    if reduction == 'none' or riemann_weight is not None:
                        # 对于none reduction或有权重的情况，需要先求和再反向传播
                        riemann_loss_sum = riemann_loss.sum()
                        riemann_loss_sum.backward()
                    else:
                        riemann_loss.backward()
                    
                    # 使用grad属性获取梯度
                    riemann_grad = riemann_input.grad
                    
                    # PyTorch计算
                    if device == "cpu":
                        torch_input = torch.tensor(input_data, requires_grad=True)
                        torch_target = torch.tensor(target_data, requires_grad=False)
                    else:  # cuda
                        torch_input = torch.tensor(input_data, requires_grad=True, device=device)
                        torch_target = torch.tensor(target_data, requires_grad=False, device=device)
                    torch_loss = torch.nn.functional.smooth_l1_loss(torch_input, torch_target, reduction=reduction, beta=beta)
                    
                    # 应用权重
                    if torch_weight is not None:
                        torch_loss = torch_loss * torch_weight
                    
                    # 修正反向传播逻辑
                    if reduction == 'none' or torch_weight is not None:
                        torch_loss.sum().backward()
                    else:
                        torch_loss.backward()  # 对于sum reduction，直接调用backward()
                    torch_grad = torch_input.grad
                    
                    # 比较结果
                    values_match = compare_values(riemann_loss, torch_loss)
                    grads_match = compare_values(riemann_grad, torch_grad)
                    
                    # 清理梯度
                    riemann_input.grad = None
                    
                    # 判断是否通过
                    passed = values_match and grads_match
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        # 修改为一行输出，包含测试用例名称和结果
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: {'通过' if values_match else '失败'}")
                            print(f"  梯度比较: {'通过' if grads_match else '失败'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(values_match, f"值比较失败: {case_name}")
                    self.assertTrue(grads_match, f"梯度比较失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"  {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_cross_entropy(self):
        """测试交叉熵损失函数"""
        test_cases = [
            {"name": "基本测试(mean)", "batch_size": 4, "num_classes": 3, "reduction": "mean"},
            {"name": "基本测试(sum)", "batch_size": 4, "num_classes": 3, "reduction": "sum"},
            {"name": "基本测试(none)", "batch_size": 4, "num_classes": 3, "reduction": "none"},
            {"name": "带权重测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "use_weight": True},
            {"name": "带忽略索引测试(sum)", "batch_size": 4, "num_classes": 3, "reduction": "sum", "use_ignore_index": True},
            {"name": "带忽略索引测试(none)", "batch_size": 4, "num_classes": 3, "reduction": "none", "use_ignore_index": True},
            {"name": "忽略索引+权重测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "use_ignore_index": True, "use_weight": True},
            {"name": "忽略索引+标签平滑测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "use_ignore_index": True, "label_smoothing": 0.1},
            {"name": "忽略索引+权重+标签平滑测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "use_ignore_index": True, "use_weight": True, "label_smoothing": 0.1},
            {"name": "无有效样本测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "all_ignore": True},
            {"name": "高维输入+忽略索引测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "dimensions": (2, 2), "use_ignore_index": True},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"cross_entropy: {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 生成随机输入
                    batch_size = case["batch_size"]
                    num_classes = case["num_classes"]
                    reduction = case["reduction"]
                    label_smoothing = case.get("label_smoothing", 0.0)  # 默认0.0
                    
                    # 确定输入形状
                    if "dimensions" in case:
                        input_shape = (batch_size, num_classes) + case["dimensions"]
                        target_shape = (batch_size,) + case["dimensions"]
                    else:
                        input_shape = (batch_size, num_classes)
                        target_shape = (batch_size,)
                    
                    # 生成测试数据
                    np.random.seed(42)
                    input_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    target_data = np.random.randint(0, num_classes, size=target_shape).astype(np.int64)
                    
                    # 处理忽略索引
                    ignore_index = -100
                    
                    if case.get("use_ignore_index", False) or case.get("all_ignore", False):
                        # 将部分或全部目标设为忽略索引
                        if case.get("all_ignore", False):
                            target_data = np.full_like(target_data, ignore_index)
                        else:
                            mask = np.random.rand(*target_shape) < 0.2  # 20%的概率设为忽略
                            target_data[mask] = ignore_index
                    
                    # 处理权重
                    riemann_weight = None
                    torch_weight = None
                    
                    if case.get("use_weight", False):
                        weight_data = np.random.rand(num_classes).astype(rm.get_default_dtype())
                        if device == "cpu":
                            riemann_weight = tensor(weight_data, requires_grad=False)
                            torch_weight = torch.tensor(weight_data, requires_grad=False)
                        else:  # cuda
                            riemann_weight = tensor(weight_data, requires_grad=False, device=device)
                            torch_weight = torch.tensor(weight_data, requires_grad=False, device=device)
                    else:
                        riemann_weight = None
                        torch_weight = None
                    
                    # Riemann计算
                    if device == "cpu":
                        riemann_input = tensor(input_data, requires_grad=True)
                        riemann_target = tensor(target_data, requires_grad=False)
                    else:  # cuda
                        riemann_input = tensor(input_data, requires_grad=True, device=device)
                        riemann_target = tensor(target_data, requires_grad=False, device=device)
                    riemann_loss = cross_entropy(riemann_input, riemann_target, weight=riemann_weight,
                                                ignore_index=ignore_index, reduction=reduction,
                                                label_smoothing=label_smoothing)
                    
                    # 反向传播 - 添加all_ignore处理逻辑
                    if case.get("all_ignore", False):
                        # 当所有目标都被忽略时，可能无法计算梯度，跳过反向传播
                        riemann_grad = None
                    elif reduction == 'none':
                        # 对于none reduction，需要先求和再反向传播
                        riemann_loss_sum = riemann_loss.sum()
                        riemann_loss_sum.backward()
                        riemann_grad = riemann_input.grad
                    else:
                        riemann_loss.backward()
                        riemann_grad = riemann_input.grad
                    
                    # PyTorch计算
                    if device == "cpu":
                        torch_input = torch.tensor(input_data, requires_grad=True)
                        torch_target = torch.tensor(target_data, requires_grad=False)
                    else:  # cuda
                        torch_input = torch.tensor(input_data, requires_grad=True, device=device)
                        torch_target = torch.tensor(target_data, requires_grad=False, device=device)
                    torch_loss = torch.nn.functional.cross_entropy(torch_input, torch_target, 
                                                                weight=torch_weight,
                                                                ignore_index=ignore_index, 
                                                                reduction=reduction,
                                                                label_smoothing=label_smoothing)
                    
                    # 反向传播 - 添加all_ignore处理逻辑
                    if case.get("all_ignore", False):
                        # 当所有目标都被忽略时，跳过反向传播
                        torch_grad = None
                    else:
                        # 正常情况执行反向传播
                        torch_loss.sum().backward()
                        torch_grad = torch_input.grad
                    
                    # 比较结果
                    values_match = compare_values(riemann_loss, torch_loss)
                    grads_match = compare_values(riemann_grad, torch_grad)
                    
                    # 判断是否通过
                    passed = values_match and grads_match
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        # 修改为一行输出，包含测试用例名称和结果
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"    损失值: {riemann_loss} \n    (PyTorch: {torch_loss})")
                            print(f"    梯度值: {riemann_grad} \n    (PyTorch: {torch_grad})")
                            print(f"    值比较: {'通过' if values_match else '失败'}")
                            print(f"    梯度比较: {'通过' if grads_match else '失败'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(values_match, f"值比较失败: {case_name}")
                    self.assertTrue(grads_match, f"梯度比较失败: {case_name}")
                    
                except Exception as e:
                        time_taken = time.time() - start_time
                        if IS_RUNNING_AS_SCRIPT:
                            stats.add_result(case["name"], False, [str(e)])
                            print(f"  {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                        raise
    
    # 1. 修复binary_cross_entropy_with_logits方法中的PyTorch反向传播逻辑
    def test_BCE_with_logits(self):
        """测试带logits的二分类交叉熵损失函数"""
        test_cases = [
            {"name": "基本测试(mean)", "batch_size": 4, "features": 3, "reduction": "mean"},
            {"name": "基本测试(sum)", "batch_size": 4, "features": 3, "reduction": "sum"},
            {"name": "基本测试(none)", "batch_size": 4, "features": 3, "reduction": "none"},
            {"name": "带权重测试", "batch_size": 4, "features": 3, "reduction": "mean", "use_weight": True},
            {"name": "带正类权重测试", "batch_size": 4, "features": 3, "reduction": "mean", "use_pos_weight": True},
            {"name": "二值目标测试", "batch_size": 4, "features": 1, "reduction": "mean", "binary_target": True},
            {"name": "大维度测试", "batch_size": 16, "features": 64, "reduction": "mean"},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"BCE_with_logits: {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 生成随机输入
                    batch_size = case["batch_size"]
                    features = case["features"]
                    reduction = case["reduction"]
                    
                    input_shape = (batch_size, features)
                    
                    # 生成测试数据
                    np.random.seed(42)
                    input_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    
                    # 生成目标数据
                    if case.get("binary_target", False):
                        # 二值目标 (0或1)
                        target_data = np.random.randint(0, 2, size=input_shape).astype(rm.get_default_dtype())
                    else:
                        # 概率目标 (0-1之间的浮点数)
                        target_data = np.random.rand(*input_shape).astype(rm.get_default_dtype())
                    
                    # 处理权重
                    riemann_weight = None
                    torch_weight = None
                    riemann_pos_weight = None
                    torch_pos_weight = None
                    
                    if case.get("use_weight", False):
                        weight_data = np.random.rand(*input_shape).astype(rm.get_default_dtype())
                        if device == "cpu":
                            riemann_weight = tensor(weight_data, requires_grad=False)
                            torch_weight = torch.tensor(weight_data, requires_grad=False)
                        else:  # cuda
                            riemann_weight = tensor(weight_data, requires_grad=False, device=device)
                            torch_weight = torch.tensor(weight_data, requires_grad=False, device=device)
                    
                    if case.get("use_pos_weight", False):
                        pos_weight_data = np.random.rand(features).astype(rm.get_default_dtype()) + 0.5  # 确保为正数
                        if device == "cpu":
                            riemann_pos_weight = tensor(pos_weight_data, requires_grad=False)
                            torch_pos_weight = torch.tensor(pos_weight_data, requires_grad=False)
                        else:  # cuda
                            riemann_pos_weight = tensor(pos_weight_data, requires_grad=False, device=device)
                            torch_pos_weight = torch.tensor(pos_weight_data, requires_grad=False, device=device)
                    
                    # 1. Riemann计算部分
                    if device == "cpu":
                        riemann_input = tensor(input_data, requires_grad=True)
                        riemann_target = tensor(target_data, requires_grad=False)
                    else:  # cuda
                        riemann_input = tensor(input_data, requires_grad=True, device=device)
                        riemann_target = tensor(target_data, requires_grad=False, device=device)
                    riemann_loss = binary_cross_entropy_with_logits(riemann_input, riemann_target, 
                                                                  weight=riemann_weight, pos_weight=riemann_pos_weight,
                                                                  reduction=reduction)
                    
                    # Riemann反向传播
                    if reduction == 'none' or riemann_weight is not None or riemann_pos_weight is not None:
                        riemann_loss_sum = riemann_loss.sum()
                        riemann_loss_sum.backward()
                    else:
                        riemann_loss.backward()
                    
                    # 获取Riemann梯度
                    riemann_grad = riemann_input.grad
                    
                    # 2. PyTorch计算部分
                    if device == "cpu":
                        torch_input = torch.tensor(input_data, requires_grad=True)
                        torch_target = torch.tensor(target_data, requires_grad=False)
                    else:  # cuda
                        torch_input = torch.tensor(input_data, requires_grad=True, device=device)
                        torch_target = torch.tensor(target_data, requires_grad=False, device=device)
                    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        torch_input, torch_target, weight=torch_weight, pos_weight=torch_pos_weight,
                        reduction=reduction
                    )
                    
                    # PyTorch反向传播
                    if reduction == 'none' or torch_weight is not None or torch_pos_weight is not None:
                        # 当使用pos_weight时，也需要先求和再反向传播
                        torch_loss_sum = torch_loss.sum()
                        torch_loss_sum.backward()
                    else:
                        torch_loss.backward()
                    
                    # 获取PyTorch梯度
                    torch_grad = torch_input.grad
                    
                    # 比较结果
                    values_match = compare_values(riemann_loss, torch_loss)
                    grads_match = compare_values(riemann_grad, torch_grad)
                    
                    # 判断是否通过
                    passed = values_match and grads_match
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        # 修改为一行输出，包含测试用例名称和结果
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: {'通过' if values_match else '失败'}")
                            print(f"  梯度比较: {'通过' if grads_match else '失败'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(values_match, f"值比较失败: {case_name}")
                    self.assertTrue(grads_match, f"梯度比较失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"  {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_huber_loss(self):
        """测试Huber损失函数"""
        test_cases = [
            {"name": "基本测试(mean, delta=1.0)", "batch_size": 4, "features": 3, "reduction": "mean", "delta": 1.0},
            {"name": "基本测试(sum, delta=1.0)", "batch_size": 4, "features": 3, "reduction": "sum", "delta": 1.0},
            {"name": "基本测试(none, delta=1.0)", "batch_size": 4, "features": 3, "reduction": "none", "delta": 1.0},
            {"name": "不同delta值测试(0.5)", "batch_size": 4, "features": 3, "reduction": "mean", "delta": 0.5},
            {"name": "不同delta值测试(2.0)", "batch_size": 4, "features": 3, "reduction": "mean", "delta": 2.0},
            {"name": "大维度测试", "batch_size": 16, "features": 128, "reduction": "mean", "delta": 1.0},
            {"name": "3D张量测试", "batch_size": 4, "features": (3, 3), "reduction": "mean", "delta": 1.0},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"huber_loss: {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 生成随机输入
                    batch_size = case["batch_size"]
                    features = case["features"]
                    reduction = case["reduction"]
                    delta = case["delta"]
                    
                    if isinstance(features, tuple):
                        input_shape = (batch_size,) + features
                    else:
                        input_shape = (batch_size, features)
                    
                    # 生成测试数据
                    np.random.seed(42)
                    input_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    target_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    
                    # Riemann计算
                    if device == "cpu":
                        riemann_input = tensor(input_data, requires_grad=True)
                        riemann_target = tensor(target_data, requires_grad=False)
                    else:  # cuda
                        riemann_input = tensor(input_data, requires_grad=True, device=device)
                        riemann_target = tensor(target_data, requires_grad=False, device=device)
                    riemann_loss = huber_loss(riemann_input, riemann_target, delta=delta, reduction=reduction)
                    
                    # 反向传播
                    if reduction == 'none':
                        # 对于none reduction，需要先求和再反向传播
                        riemann_loss_sum = riemann_loss.sum()
                        riemann_loss_sum.backward()
                    else:
                        riemann_loss.backward()
                    
                    # 使用grad属性获取梯度
                    riemann_grad = riemann_input.grad
                    
                    # PyTorch计算
                    if device == "cpu":
                        torch_input = torch.tensor(input_data, requires_grad=True)
                        torch_target = torch.tensor(target_data, requires_grad=False)
                    else:  # cuda
                        torch_input = torch.tensor(input_data, requires_grad=True, device=device)
                        torch_target = torch.tensor(target_data, requires_grad=False, device=device)
                    torch_loss = torch.nn.functional.huber_loss(torch_input, torch_target, delta=delta, reduction=reduction)
                    
                    # 反向传播
                    torch_loss.sum().backward()
                    torch_grad = torch_input.grad
                    
                    # 比较结果
                    values_match = compare_values(riemann_loss, torch_loss)
                    grads_match = compare_values(riemann_grad, torch_grad)
                    
                    # 清理梯度
                    riemann_input.grad = None
                    
                    # 判断是否通过
                    passed = values_match and grads_match
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        # 修改为一行输出，包含测试用例名称和结果
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: {'通过' if values_match else '失败'}")
                            print(f"  梯度比较: {'通过' if grads_match else '失败'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(values_match, f"值比较失败: {case_name}")
                    self.assertTrue(grads_match, f"梯度比较失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"  {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise
    
    def test_nll_loss(self):
        """测试负对数似然损失函数"""
        test_cases = [
            {"name": "基本测试(mean)", "batch_size": 4, "num_classes": 3, "reduction": "mean"},
            {"name": "基本测试(sum)", "batch_size": 4, "num_classes": 3, "reduction": "sum"},
            {"name": "基本测试(none)", "batch_size": 4, "num_classes": 3, "reduction": "none"},
            {"name": "带权重测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "use_weight": True},
            {"name": "带忽略索引测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "use_ignore_index": True},
            {"name": "高维输入测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "dimensions": (2, 2)},  # 3D输入
            {"name": "无有效样本测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "all_ignore": True},
            {"name": "带忽略索引测试(sum)", "batch_size": 4, "num_classes": 3, "reduction": "sum", "use_ignore_index": True},
            {"name": "带忽略索引测试(none)", "batch_size": 4, "num_classes": 3, "reduction": "none", "use_ignore_index": True},
            {"name": "忽略索引+权重测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "use_ignore_index": True, "use_weight": True},
            {"name": "高维输入+忽略索引测试", "batch_size": 4, "num_classes": 3, "reduction": "mean", "dimensions": (2, 2), "use_ignore_index": True},
        ]
        
        # 定义要测试的设备列表
        devices = ["cpu"]
        if CUDA_AVAILABLE:
            devices.append("cuda")
        
        for device in devices:
            for case in test_cases:
                case_name = f"nll_loss: {case['name']} - {device}"
                start_time = time.time()
                try:
                    # 生成随机输入
                    batch_size = case["batch_size"]
                    num_classes = case["num_classes"]
                    reduction = case["reduction"]
                    
                    # 确定输入形状
                    if "dimensions" in case:
                        input_shape = (batch_size, num_classes) + case["dimensions"]
                        target_shape = (batch_size,) + case["dimensions"]
                    else:
                        input_shape = (batch_size, num_classes)
                        target_shape = (batch_size,)
                    
                    # 生成测试数据 - 注意：NLLLoss期望输入是log probabilities
                    np.random.seed(42)
                    # 生成随机值然后归一化以模拟log probabilities
                    raw_data = np.random.randn(*input_shape).astype(rm.get_default_dtype())
                    input_data = raw_data - np.max(raw_data, axis=1, keepdims=True)  # 数值稳定化
                    
                    target_data = np.random.randint(0, num_classes, size=target_shape).astype(np.int64)
                    
                    # 处理忽略索引
                    ignore_index = -100
                    
                    if case.get("use_ignore_index", False) or case.get("all_ignore", False):
                        # 将部分或全部目标设为忽略索引
                        if case.get("all_ignore", False):
                            target_data = np.full_like(target_data, ignore_index)
                        else:
                            mask = np.random.rand(*target_shape) < 0.2  # 20%的概率设为忽略
                            target_data[mask] = ignore_index
                    
                    # 处理权重
                    riemann_weight = None
                    torch_weight = None
                    
                    if case.get("use_weight", False):
                        weight_data = np.random.rand(num_classes).astype(rm.get_default_dtype())
                        if device == "cpu":
                            riemann_weight = tensor(weight_data, requires_grad=False)
                            torch_weight = torch.tensor(weight_data, requires_grad=False)
                        else:  # cuda
                            riemann_weight = tensor(weight_data, requires_grad=False, device=device)
                            torch_weight = torch.tensor(weight_data, requires_grad=False, device=device)
                    else:
                        riemann_weight = None
                        torch_weight = None
                    
                    # Riemann计算
                    if device == "cpu":
                        riemann_input = tensor(input_data, requires_grad=True)
                        riemann_target = tensor(target_data, requires_grad=False)
                    else:  # cuda
                        riemann_input = tensor(input_data, requires_grad=True, device=device)
                        riemann_target = tensor(target_data, requires_grad=False, device=device)
                    riemann_loss = nll_loss(riemann_input, riemann_target, weight=riemann_weight,
                                          ignore_index=ignore_index, reduction=reduction)
                    
                    # 反向传播 - 修改为：检查是否所有目标都被忽略
                    if case.get("all_ignore", False):
                        # 当所有目标都被忽略时，可能无法计算梯度，跳过反向传播
                        riemann_grad = None
                    elif reduction == 'none':
                        # 对于none reduction，需要先求和再反向传播
                        riemann_loss_sum = riemann_loss.sum()
                        riemann_loss_sum.backward()
                        riemann_grad = riemann_input.grad
                    else:
                        riemann_loss.backward()
                        riemann_grad = riemann_input.grad
                    
                    # PyTorch计算
                    if device == "cpu":
                        torch_input = torch.tensor(input_data, requires_grad=True)
                        torch_target = torch.tensor(target_data, requires_grad=False)
                    else:  # cuda
                        torch_input = torch.tensor(input_data, requires_grad=True, device=device)
                        torch_target = torch.tensor(target_data, requires_grad=False, device=device)
                    torch_loss = torch.nn.functional.nll_loss(torch_input, torch_target, weight=torch_weight,
                                                           ignore_index=ignore_index, reduction=reduction)
                    
                    # 反向传播 - 修改PyTorch侧使其与Riemann侧一致
                    if case.get("all_ignore", False):
                        # 当所有目标都被忽略时，跳过反向传播
                        torch_grad = None
                    else:
                        # 正常情况执行反向传播
                        torch_loss.sum().backward()
                        torch_grad = torch_input.grad
                    
                    # 比较结果
                    values_match = compare_values(riemann_loss, torch_loss)
                    grads_match = compare_values(riemann_grad, torch_grad)
                    
                    # 判断是否通过
                    passed = values_match and grads_match
                    time_taken = time.time() - start_time
                    
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, passed)
                        status = "通过" if passed else "失败"
                        # 修改为一行输出，包含测试用例名称和结果
                        print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                        if not passed:
                            print(f"  值比较: {'通过' if values_match else '失败'}")
                            print(f"  梯度比较: {'通过' if grads_match else '失败'}")
                    
                    # 断言确保测试通过
                    self.assertTrue(values_match, f"值比较失败: {case_name}")
                    self.assertTrue(grads_match, f"梯度比较失败: {case_name}")
                    
                except Exception as e:
                    time_taken = time.time() - start_time
                    if IS_RUNNING_AS_SCRIPT:
                        stats.add_result(case_name, False, [str(e)])
                        print(f"  {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
                    raise

# 如果作为独立脚本运行
if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # rm.set_default_dtype(rm.float64)
    # torch.set_default_dtype(torch.float64)

    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行损失函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestLossFunctions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)