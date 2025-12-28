import unittest
import numpy as np
import sys, os
import time

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.tensordef import tensor
    from riemann.autograd import grad
    RIEMANN_AVAILABLE = True
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    RIEMANN_AVAILABLE = False

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
    print("警告: 无法导入PyTorch，将只测试riemann的gather函数")
    TORCH_AVAILABLE = False

# 在模块级别进行Riemann预热
if RIEMANN_AVAILABLE:
    print("预热Riemann系统...")
    warmup_start = time.time()
    
    # 执行简单的Riemann操作以触发初始化
    warmup_input = tensor([[0.0]], requires_grad=True)
    warmup_output = warmup_input.sum()
    grad(warmup_output, warmup_input)
    
    print(f"Riemann预热完成，耗时: {time.time() - warmup_start:.4f}秒")

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
    
    def print_summary(self):
        print("\n" + "="*73)
        print(f"{Colors.BOLD}测试统计摘要{Colors.ENDC}")
        print("="*73)
        print(f"总测试用例数: {self.total_cases}")
        print(f"通过测试用例数: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases}{Colors.ENDC}")
        print(f"测试通过率: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases/self.total_cases*100:.2f}%{Colors.ENDC}")
        print(f"总耗时: {self.total_time:.4f} 秒")
        print("\n各函数测试详情:")
        print("-"*73)
        print(f"{'函数名':<30}{'通过/总数':<15}{'通过率':<10}{'耗时(秒)':<10}")
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

class TestGather(unittest.TestCase):
    def setUp(self):
        # 确保riemann可用
        if not RIEMANN_AVAILABLE:
            self.skipTest("Riemann模块不可用")
            
        # 设置随机种子以确保结果可重复
        np.random.seed(42)
        if TORCH_AVAILABLE:
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
    
    def compare_tensors(self, rm_tensor, pt_tensor, test_name):
        """比较riemann和pytorch张量是否一致"""
        # 比较形状
        self.assertEqual(rm_tensor.shape, pt_tensor.shape, 
                         f"{test_name} - 形状不匹配: rm={rm_tensor.shape}, pt={pt_tensor.shape}")
        
        # 比较数据类型
        rm_dtype = str(rm_tensor.data.dtype)
        pt_dtype = str(pt_tensor.dtype)
        
        # 规范化类型名称
        type_mapping = {
            'int64': 'int64', 'long': 'int64',
            'float32': 'float32', 'float': 'float32',
            'float64': 'float64', 'double': 'float64'
        }
        
        rm_dtype_normalized = type_mapping.get(rm_dtype.replace('torch.', '').replace('numpy.', ''), rm_dtype)
        pt_dtype_normalized = type_mapping.get(pt_dtype.replace('torch.', '').replace('numpy.', ''), pt_dtype)
        
        self.assertEqual(rm_dtype_normalized, pt_dtype_normalized, 
                         f"{test_name} - 数据类型不匹配: rm={rm_dtype}, pt={pt_dtype}")
        
        # 比较值
        rm_data = rm_tensor.data if hasattr(rm_tensor, 'data') else rm_tensor
        pt_data = pt_tensor.detach().numpy()
        
        if rm_data.size == 0 and pt_data.size == 0:
            max_diff = 0.0
        else:
            max_diff = np.max(np.abs(rm_data - pt_data))
        
        self.assertLess(max_diff, 1e-6, f"{test_name} - 值不匹配: 最大差异={max_diff}")
        
        if IS_RUNNING_AS_SCRIPT:
            print(f"  {Colors.OKGREEN}✓ 通过{Colors.ENDC} - {test_name} (最大差异: {max_diff})")
            stats.add_result(test_name, True)
        
        return True

    def test_gather_basic_functionality(self):
        """测试gather函数的基础功能"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        
        # 测试场景1: 2D张量，dim=0
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景1: 2D张量，dim=0")
        
        rm_input = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32')
        pt_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        
        rm_index = tensor([[0, 1], [1, 0]], dtype='int64')
        pt_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        rm_output = rm_input.gather(dim=0, index=rm_index)
        pt_output = pt_input.gather(dim=0, index=pt_index)
        self.compare_tensors(rm_output, pt_output, "2D张量，dim=0")
        
        # 测试场景2: 2D张量，dim=1
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景2: 2D张量，dim=1")
        
        rm_index = tensor([[0, 2], [1, 0]], dtype='int64')
        pt_index = torch.tensor([[0, 2], [1, 0]], dtype=torch.long)
        
        rm_output = rm_input.gather(dim=1, index=rm_index)
        pt_output = pt_input.gather(dim=1, index=pt_index)
        self.compare_tensors(rm_output, pt_output, "2D张量，dim=1")
        
        # 测试场景3: 3D张量，不同维度
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景3: 3D张量，不同维度")
        
        rm_input_3d = tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype='float32')
        pt_input_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32)
        
        # dim=0
        rm_index_3d = tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype='int64')
        pt_index_3d = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=torch.long)
        
        rm_output = rm_input_3d.gather(dim=0, index=rm_index_3d)
        pt_output = pt_input_3d.gather(dim=0, index=pt_index_3d)
        self.compare_tensors(rm_output, pt_output, "3D张量，dim=0")
        
        # dim=2
        rm_index_3d_dim2 = tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype='int64')
        pt_index_3d_dim2 = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=torch.long)
        
        rm_output = rm_input_3d.gather(dim=2, index=rm_index_3d_dim2)
        pt_output = pt_input_3d.gather(dim=2, index=pt_index_3d_dim2)
        self.compare_tensors(rm_output, pt_output, "3D张量，dim=2")

    def test_gather_edge_cases(self):
        """测试gather函数的边界情况"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        
        # 测试场景1: 空张量
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景1: 空张量处理")
        
        rm_input = tensor([], dtype='float32')
        pt_input = torch.tensor([], dtype=torch.float32)
        rm_index = tensor([], dtype='int64')
        pt_index = torch.tensor([], dtype=torch.long)
        
        rm_output = rm_input.gather(dim=0, index=rm_index)
        pt_output = pt_input.gather(dim=0, index=pt_index)
        self.compare_tensors(rm_output, pt_output, "空张量处理")
        
        # 测试场景2: 单元素张量
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景2: 单元素张量")
        
        rm_input = tensor([[5.0]], dtype='float32')
        pt_input = torch.tensor([[5.0]], dtype=torch.float32)
        rm_index = tensor([[0]], dtype='int64')
        pt_index = torch.tensor([[0]], dtype=torch.long)
        
        rm_output = rm_input.gather(dim=1, index=rm_index)
        pt_output = pt_input.gather(dim=1, index=pt_index)
        self.compare_tensors(rm_output, pt_output, "单元素张量")
        
        # 测试场景3: 索引边界值
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景3: 索引边界值")
        
        rm_input = tensor([[10.0, 20.0], [30.0, 40.0]], dtype='float32')
        pt_input = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
        
        # 测试最小和最大索引值
        rm_index_min = tensor([[0, 0]], dtype='int64')
        rm_index_max = tensor([[1, 1]], dtype='int64')
        pt_index_min = torch.tensor([[0, 0]], dtype=torch.long)
        pt_index_max = torch.tensor([[1, 1]], dtype=torch.long)
        
        rm_output_min = rm_input.gather(dim=0, index=rm_index_min)
        rm_output_max = rm_input.gather(dim=0, index=rm_index_max)
        pt_output_min = pt_input.gather(dim=0, index=pt_index_min)
        pt_output_max = pt_input.gather(dim=0, index=pt_index_max)
        
        self.compare_tensors(rm_output_min, pt_output_min, "最小索引值")
        self.compare_tensors(rm_output_max, pt_output_max, "最大索引值")

    def test_gather_broadcasting(self):
        """测试gather函数的广播行为"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        
        # 测试场景1: 索引形状与输入形状的匹配
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景1: 不同形状的索引")
        
        rm_input = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32')
        pt_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        
        # 比输入形状小的索引
        rm_index_small = tensor([[0]], dtype='int64')
        pt_index_small = torch.tensor([[0]], dtype=torch.long)
        
        rm_output = rm_input.gather(dim=0, index=rm_index_small)
        pt_output = pt_input.gather(dim=0, index=pt_index_small)
        self.compare_tensors(rm_output, pt_output, "小形状索引")
        
        # 与输入形状匹配的索引
        rm_index_match = tensor([[0, 1, 0], [1, 0, 1]], dtype='int64')
        pt_index_match = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.long)
        
        rm_output = rm_input.gather(dim=0, index=rm_index_match)
        pt_output = pt_input.gather(dim=0, index=pt_index_match)
        self.compare_tensors(rm_output, pt_output, "匹配形状索引")

    def test_gather_advanced_scenarios(self):
        """测试gather函数的高级场景"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        
        # 测试场景1: 每行采集多个元素
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景1: 每行采集多个元素")
        
        # 显式指定dtype为int64，确保与PyTorch行为一致
        rm_input = tensor(np.arange(15).astype(np.int64)).view(3, 5)
        pt_input = torch.arange(15).view(3, 5)
        
        rm_index = tensor([[0, 1], [2, 3], [1, 2]], dtype='int64')
        pt_index = torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.long)
        
        rm_output = rm_input.gather(dim=1, index=rm_index)
        pt_output = pt_input.gather(dim=1, index=pt_index)
        self.compare_tensors(rm_output, pt_output, "每行多元素采集")
        
        # 测试场景2: 负维度索引
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景2: 负维度索引")
        
        rm_input = tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='float32')
        pt_input = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32)
        
        rm_index = tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype='int64')
        pt_index = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=torch.long)
        
        rm_output = rm_input.gather(dim=-1, index=rm_index)  # dim=-1 表示最后一个维度
        pt_output = pt_input.gather(dim=-1, index=pt_index)
        self.compare_tensors(rm_output, pt_output, "负维度索引")

    def test_gather_backpropagation(self):
        """测试gather函数的反向传播"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        
        # 预热已在模块级别完成，这里直接进行测试
        
        # 测试场景1: 基本反向传播
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景1: 基本反向传播")
        
        # 预创建索引张量以减少重复创建的开销
        rm_index = tensor([[0, 1], [1, 0]], dtype='int64')
        pt_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        # Riemann前向传播和梯度计算
        rm_input = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32', requires_grad=True)
        rm_output = rm_input.gather(dim=0, index=rm_index)
        rm_loss = rm_output.sum()
        rm_grad = grad(rm_loss, rm_input)
        
        # PyTorch前向传播和梯度计算
        pt_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
        pt_output = pt_input.gather(dim=0, index=pt_index)
        pt_loss = pt_output.sum()
        pt_loss.backward()
        pt_grad = pt_input.grad
        
        self.compare_tensors(rm_grad[0], pt_grad, "一阶梯度")
        
        # 测试场景2: 复杂计算图
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景2: 复杂计算图")
        
        # 预创建索引张量
        rm_index_small = tensor([[0, 1]], dtype='int64')
        rm_index_single = tensor([[0]], dtype='int64')
        pt_index_small = torch.tensor([[0, 1]], dtype=torch.long)
        pt_index_single = torch.tensor([[0]], dtype=torch.long)
        
        def rm_complex_operation(x):
            temp = x.gather(dim=0, index=rm_index_small)
            final = temp.gather(dim=1, index=rm_index_single)
            return final.sum()
        
        def pt_complex_operation(x):
            temp = x.gather(dim=0, index=pt_index_small)
            final = temp.gather(dim=1, index=pt_index_single)
            return final.sum()
        
        # Riemann复杂计算图
        rm_input2 = tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32', requires_grad=True)
        rm_loss2 = rm_complex_operation(rm_input2)
        rm_grad2 = grad(rm_loss2, rm_input2)
        
        # PyTorch复杂计算图
        pt_input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
        pt_loss2 = pt_complex_operation(pt_input2)
        pt_loss2.backward()
        pt_grad2 = pt_input2.grad
        
        self.compare_tensors(rm_grad2[0], pt_grad2, "复杂计算图梯度")

    def test_gather_real_world(self):
        """测试gather函数的实际应用场景"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        
        # 测试场景1: 模拟分类任务中提取目标类别概率
        if IS_RUNNING_AS_SCRIPT:
            print("测试场景1: 分类任务目标概率提取")
        
        logits_rm = tensor([[1.2, 0.5, 2.1], [0.1, 3.2, 0.3]], dtype='float32')
        logits_pt = torch.tensor([[1.2, 0.5, 2.1], [0.1, 3.2, 0.3]], dtype=torch.float32)
        
        labels_rm = tensor([[2], [1]], dtype='int64')  # 目标类别索引
        labels_pt = torch.tensor([[2], [1]], dtype=torch.long)
        
        # 使用gather提取目标类别概率
        target_probs_rm = logits_rm.gather(dim=1, index=labels_rm)
        target_probs_pt = logits_pt.gather(dim=1, index=labels_pt)
        
        self.compare_tensors(target_probs_rm, target_probs_pt, "目标概率提取")

# 运行测试的函数
def run_all_tests():
    global IS_RUNNING_AS_SCRIPT
    IS_RUNNING_AS_SCRIPT = True
    
    print(f"{Colors.HEADER}===== 开始测试 Riemann gather 函数 ====={Colors.ENDC}")
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGather)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 关闭默认输出，我们使用自己的输出格式
    result = runner.run(suite)
    
    # 打印统计摘要
    stats.print_summary()
    
    # 返回测试结果
    return result.wasSuccessful()

# 当作为独立脚本运行时
if __name__ == "__main__":
    clear_screen()
    success = run_all_tests()
    sys.exit(0 if success else 1)