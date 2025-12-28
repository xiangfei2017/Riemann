import unittest
import numpy as np
import time
import sys, os
from collections import deque

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    # 导入riemann.nn.functional模块，用于激活函数
    from riemann.nn import functional as F
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的性能")
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

IS_RUNNING_AS_SCRIPT = False

# 性能测试统计类
class PerformanceCollector:
    def __init__(self):
        self.results = []
        self.comparison_results = []  # 存储一致性比较结果
    
    def add_result(self, test_name, graph_size, forward_time, backward_time, framework="Riemann"):
        self.results.append({
            'test_name': test_name,
            'framework': framework,
            'graph_size': graph_size,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': forward_time + backward_time
        })
    
    def add_comparison_result(self, test_name, forward_consistent, grad_consistent):
        self.comparison_results.append({
            'test_name': test_name,
            'forward_consistent': forward_consistent,
            'grad_consistent': grad_consistent
        })
    
    def print_summary(self):
        # 使用Colors类进行格式化输出
        print(f"\n{Colors.HEADER}===== 性能测试结果汇总 ====={Colors.ENDC}")
        # 确保表头与数据对齐 - 精确计算中文字符宽度的影响
        # 使用等宽字体的中文字符对齐方案
        header = [
            'TestCase', 'Frame', 'CalcGraphSize', 'Forward(s)', 
            'Backward(s)', 'Total(s)', 'Value', 'Grad'
        ]
        # 定义每列宽度，精确计算中文字符宽度（每个中文字符占2个宽度单位）
        col_widths = [
            16,  # 测试名称 (额外加2个单位作为边距)
            12,   # 框架 (额外加4个单位作为边距)
            16,  # 计算图大小 (额外加2个单位作为边距)
            16,  # 前向时间(秒) (额外加2个单位作为边距)
            16,  # 反向时间(秒) (额外加2个单位作为边距)
            10,  # 总时间(秒) (额外加2个单位作为边距)
        ]
        
        # 格式化表头
        header_line = Colors.BOLD
        for i, (h, w) in enumerate(zip(header, col_widths)):
            # 左对齐每个表头标题
            header_line += f"{h:<{w}}"
        header_line += Colors.ENDC
        print(header_line)
        
        # 打印分隔线
        print("-" * sum(col_widths))
        
        # 按测试名称和框架排序结果，确保小型->中型->大型的顺序
        test_order = {"小型计算图": 0, "中型计算图": 1, "大型计算图": 2}
        sorted_results = sorted(self.results, key=lambda x: (test_order.get(x['test_name'], 999), x['framework']))
        
        # 创建比较结果映射
        comparison_map = {}
        for comp in self.comparison_results:
            comparison_map[comp['test_name']] = comp
        
        for result in sorted_results:
            # 获取一致性比较结果
            test_name = result['test_name']
                        
            # 格式化每一行 - 全部左对齐
            line = ""
            line += f"{result['test_name']:<{col_widths[0]-len(result['test_name'])}}"
            line += f"{result['framework']:<{col_widths[1]}}"
            line += f"{result['graph_size']:<{col_widths[2]}}"
            line += f"{result['forward_time']:<{col_widths[3]}.6f}"
            line += f"{result['backward_time']:<{col_widths[4]}.6f}"
            line += f"{result['total_time']:<{col_widths[5]}.6f}"
            print(line)
        
        print("-" * sum(col_widths))

# 统计计算图节点数量的函数 - 利用TN的可哈希性优化
def count_graph_nodes(root_tensor):
    """计算计算图中的节点数量
    
    优化版本：利用TN类的__hash__方法直接将张量放入set中进行跟踪
    """
    visited = set()
    queue = deque([root_tensor])
    count = 0
    
    while queue:
        tensor = queue.popleft()
        # 利用TN的可哈希性直接检查是否已访问
        if tensor in visited or not tensor.requires_grad:
            continue
        
        visited.add(tensor)
        count += 1
        
        # 遍历梯度函数的输入变量
        if hasattr(tensor, 'fromvars') and tensor.fromvars:
            for var in tensor.fromvars:
                if var not in visited:
                    queue.append(var)
    
    return count

# 辅助函数：比较两个值是否接近（考虑浮点数精度）
def values_are_close(val1, val2, rtol=1e-4, atol=1e-6):
    """比较两个值是否足够接近"""
    if isinstance(val1, rm.TN) and isinstance(val2, torch.Tensor):
        return np.allclose(val1.numpy(), val2.detach().numpy(), rtol=rtol, atol=atol)
    elif isinstance(val1, np.ndarray) and isinstance(val2, torch.Tensor):
        return np.allclose(val1, val2.detach().numpy(), rtol=rtol, atol=atol)
    elif isinstance(val1, torch.Tensor) and isinstance(val2, rm.TN):
        return np.allclose(val1.detach().numpy(), val2.numpy(), rtol=rtol, atol=atol)
    elif isinstance(val1, rm.TN) and isinstance(val2, rm.TN):
        return np.allclose(val1.numpy(), val2.numpy(), rtol=rtol, atol=atol)
    elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
        return torch.allclose(val1, val2, rtol=rtol, atol=atol)
    else:
        # 处理标量值比较
        return abs(val1 - val2) < (atol + rtol * abs(val2))

# 创建PyTorch计算图的函数，用于性能比较
def create_torch_graph(x, depth=4, width=3):
    """创建PyTorch版本的复杂计算图"""
    current_tensors = [x]
    inputs = [x]
    
    for i in range(depth):
        next_tensors = []
        
        for j in range(width):
            for tensor in current_tensors:
                if j % 5 == 0:
                    y = torch.sin(tensor) * torch.cos(tensor)
                elif j % 5 == 1:
                    scaled_tensor = tensor * 0.05
                    y = torch.exp(scaled_tensor) + torch.log(torch.abs(tensor) + 1e-8)
                elif j % 5 == 2:
                    y = tensor ** 2 - tensor * 0.5
                elif j % 5 == 3:
                    y = torch.relu(tensor) + torch.sigmoid(tensor * 0.5)
                else:
                    y = torch.tanh(tensor) * tensor
                
                next_tensors.append(y)
        
        if i > 0 and i < depth - 1 and j < 1:
            for k in range(min(1, len(inputs))):
                if k < len(next_tensors):
                    next_tensors[k] = next_tensors[k] + inputs[k] * (i + 1) * 0.1
        
        if next_tensors:
            save_count = min(1, len(next_tensors))
            inputs.extend(next_tensors[:save_count])
            
        current_tensors = next_tensors
        
        if i % 3 == 2 and current_tensors:
            mean_tensor = sum(current_tensors) / len(current_tensors)
            var_tensor = sum((t - mean_tensor) ** 2 for t in current_tensors) / len(current_tensors)
            
            for idx, tensor in enumerate(current_tensors):
                current_tensors[idx] = (tensor - mean_tensor) / torch.sqrt(var_tensor + 1e-8)
    
    weights = torch.randn(len(current_tensors), requires_grad=True) * 0.1
    inputs.append(weights)
    
    output = sum(w * t for w, t in zip(weights, current_tensors))
    if len(output) > 0:
        # 修复reshape逻辑
        total_size = len(output)
        output_shape = min(5, max(1, total_size // 10))
        while output_shape > 0 and total_size % output_shape != 0:
            output_shape -= 1
        if output_shape == 0:
            output_shape = 1
        output = torch.softmax(output.reshape(output_shape, -1), dim=1).sum()
    
    return output, inputs

# 复杂计算图生成函数 - 优化性能和数值稳定性
def create_complex_graph(input_size=20, depth=4, width=2, seed=None):
    """创建复杂的计算图用于性能测试
    
    参数:
    input_size: 输入张量的大小
    depth: 计算图的深度
    width: 每一层的宽度（并行计算分支数）
    seed: 随机种子，用于复现结果
    
    返回:
    output: 最终的输出张量
    input_tensors: 输入张量列表，用于后续梯度计算
    """
    # 设置随机种子以确保可重现性
    if seed is not None:
        np.random.seed(seed)
    
    # 创建输入张量 - 使用较小的数值范围以避免溢出
    x = rm.tensor(np.random.randn(input_size) * 0.1, requires_grad=True)
    inputs = [x]
    
    # 构建计算图
    current_tensors = [x]
    
    for i in range(depth):
        next_tensors = []
        
        # 每一层创建多个计算分支
        for j in range(width):
            # 对每个当前张量进行不同的操作
            for tensor in current_tensors:
                # 使用不同的数学运算组合
                if j % 5 == 0:
                    y = rm.sin(tensor) * rm.cos(tensor)
                elif j % 5 == 1:
                    # 限制指数运算的输入范围以避免溢出
                    scaled_tensor = tensor * 0.05
                    y = rm.exp(scaled_tensor) + rm.log(rm.abs(tensor) + 1e-8)
                elif j % 5 == 2:
                    # 使用较小的幂次运算以避免数值爆炸
                    y = tensor ** 2 - tensor * 0.5
                elif j % 5 == 3:
                    # 使用riemann.nn.functional中的激活函数
                    y = F.relu(tensor) + F.sigmoid(tensor * 0.5)
                else:
                    # 使用riemann.nn.functional中的激活函数
                    y = F.tanh(tensor) * tensor
                
                next_tensors.append(y)
        
        # 添加少量跨层连接以增加图的复杂度
        if i > 0 and i < depth - 1 and j < 1:
            for k in range(min(1, len(inputs))):
                if k < len(next_tensors):
                    next_tensors[k] = next_tensors[k] + inputs[k] * (i + 1) * 0.1
        
        # 保存少量张量用于跨层连接
        if next_tensors:
            save_count = min(1, len(next_tensors))
            inputs.extend(next_tensors[:save_count])
            
        current_tensors = next_tensors
        
        # 每三层进行一次归一化操作，提高数值稳定性
        if i % 3 == 2 and current_tensors:
            mean_tensor = sum(current_tensors) / len(current_tensors)
            var_tensor = sum((t - mean_tensor) ** 2 for t in current_tensors) / len(current_tensors)
            
            # 应用层归一化
            for idx, tensor in enumerate(current_tensors):
                current_tensors[idx] = (tensor - mean_tensor) / rm.sqrt(var_tensor + 1e-8)
    
    # 最终输出是所有张量的加权和
    weights = rm.tensor(np.random.randn(len(current_tensors)) * 0.1, requires_grad=True)
    inputs.append(weights)
    
    # 计算加权和作为最终输出
    output = sum(w * t for w, t in zip(weights, current_tensors))
    
    # 添加最终的非线性操作 - 修复reshape形状问题
    if len(output) > 0:
        # 计算有效的output_shape，确保总大小能被output_shape整除
        total_size = len(output)
        # 从可能的最大输出形状开始尝试
        output_shape = min(5, max(1, total_size // 10))
        # 确保能被整除，否则减小output_shape
        while output_shape > 0 and total_size % output_shape != 0:
            output_shape -= 1
        # 如果找不到合适的形状，就使用1
        if output_shape == 0:
            output_shape = 1
        # 确保使用正确的dim参数
        output = F.softmax(output.reshape(output_shape, -1), dim=1).sum()
    
    return output, inputs

class TestBackwardPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """在所有测试前初始化一次性能收集器"""
        cls.perf_collector = PerformanceCollector()
        # 设置随机种子以确保结果可重现
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
        # 禁用pytest的默认输出
        cls.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def setUp(self):
        """设置测试环境"""
        # 重新启用输出用于测试
        sys.stdout = self.original_stdout
    
    def test_01_performance_small_graph(self):
        """测试小型计算图的性能"""
        test_name = "小型计算图"
        
        # 测试参数
        input_size = 10
        depth = 5
        width = 2
        
        # 打印详细的测试标题信息
        print(f"\n{Colors.OKBLUE}测试: {test_name} - 节点数: {count_graph_nodes(create_complex_graph(input_size, depth, width, seed=42)[0])}, 深度: {depth}, 宽度: {width}, 输入形状: ({input_size},){Colors.ENDC}")
        
        # ===== Riemann 版本 =====
        print("  Riemann 版本:")
        # 创建并测试小型计算图 - 只统计计算图创建和前向传播的时间
        start_time = time.time()
        output_rm, inputs_rm = create_complex_graph(input_size=input_size, depth=depth, width=width, seed=42)
        rm_forward_time = time.time() - start_time
        
        # 在时间统计之外计算图节点数量（不再打印，因为标题中已有）
        graph_size = count_graph_nodes(output_rm)
        # 移除重复的节点数量打印
        print(f"  前向计算时间: {rm_forward_time:.6f}秒")
        
        # 测试反向传播时间
        start_time = time.time()
        output_rm.backward()
        rm_backward_time = time.time() - start_time
        print(f"  反向传播时间: {rm_backward_time:.6f}秒")
        
        # 验证梯度是否正确计算（时间不计入统计）
        has_valid_grad = False
        for i, inp in enumerate(inputs_rm):
            if inp.grad is not None:
                # print(f"  输入张量 {i} 的梯度形状: {inp.grad.shape}")
                has_valid_grad = True
                break
        
        if not has_valid_grad:
            print(f"  {Colors.WARNING}警告: 未检测到有效梯度{Colors.ENDC}")
        
        # 记录Riemann的性能结果
        self.perf_collector.add_result(test_name, graph_size, rm_forward_time, rm_backward_time, framework="Riemann")
        
        # ===== PyTorch 版本 =====
        if TORCH_AVAILABLE:
            print("  PyTorch 版本:")
            # 设置相同的随机种子
            torch.manual_seed(42)
            # 创建相同规模的输入数据
            x_np = np.random.randn(input_size) * 0.1
            x_torch = torch.tensor(x_np, requires_grad=True)
            
            # 只统计计算图创建和前向传播的时间
            start_time = time.time()
            output_torch, _ = create_torch_graph(x_torch, depth=depth, width=width)
            torch_forward_time = time.time() - start_time
            print(f"  前向计算时间: {torch_forward_time:.6f}秒")
            
            # 测试反向传播时间
            start_time = time.time()
            output_torch.backward()
            torch_backward_time = time.time() - start_time
            print(f"  反向传播时间: {torch_backward_time:.6f}秒")
            
            # 记录PyTorch的性能结果，使用相同的图大小
            self.perf_collector.add_result(test_name, graph_size, torch_forward_time, torch_backward_time, framework="PyTorch")
            
            # 比较前向计算值和梯度值的一致性
            # 前向计算值比较
            forward_consistent = values_are_close(output_rm.item(), output_torch.item())
            
            # 梯度值比较
            grad_consistent = False
            if has_valid_grad and x_torch.grad is not None:
                # 比较第一个输入张量的梯度
                grad_consistent = values_are_close(inputs_rm[0].grad, x_torch.grad)
            
            # 显示一致性比较结果
            forward_status = f"{Colors.OKGREEN}一致{Colors.ENDC}" if forward_consistent else f"{Colors.FAIL}不一致{Colors.ENDC}"
            grad_status = f"{Colors.OKGREEN}一致{Colors.ENDC}" if grad_consistent else f"{Colors.FAIL}不一致{Colors.ENDC}"
            print(f"  前向计算值比较: {forward_status}")
            print(f"  梯度值比较: {grad_status}")
            
            # 保存比较结果
            self.perf_collector.add_comparison_result(test_name, forward_consistent, grad_consistent)
    
    def test_02_performance_medium_graph(self):
        """测试中型计算图的性能"""
        test_name = "中型计算图"

        if IS_RUNNING_AS_SCRIPT:
            # 测试参数
            input_size = 25
            depth = 12
            width = 2
        else:
            # 测试参数
            input_size = 10
            depth = 5
            width = 2
        
        
        # 打印详细的测试标题信息
        print(f"\n{Colors.OKBLUE}测试: {test_name} - 节点数: {count_graph_nodes(create_complex_graph(input_size, depth, width, seed=42)[0])}, 深度: {depth}, 宽度: {width}, 输入形状: ({input_size},){Colors.ENDC}")
        
        # ===== Riemann 版本 =====
        print("  Riemann 版本:")
        # 创建并测试中型计算图 - 只统计计算图创建和前向传播的时间
        start_time = time.time()
        output_rm, inputs_rm = create_complex_graph(input_size=input_size, depth=depth, width=width, seed=42)
        rm_forward_time = time.time() - start_time
        
        # 在时间统计之外计算图节点数量（不再打印，因为标题中已有）
        graph_size = count_graph_nodes(output_rm)
        # 移除重复的节点数量打印
        print(f"  前向计算时间: {rm_forward_time:.6f}秒")
        
        # 测试反向传播时间
        start_time = time.time()
        output_rm.backward()
        rm_backward_time = time.time() - start_time
        print(f"  反向传播时间: {rm_backward_time:.6f}秒")
        
        # 验证梯度是否正确计算（时间不计入统计）
        has_valid_grad = False
        for i, inp in enumerate(inputs_rm):
            if inp.grad is not None:
                # print(f"  输入张量 {i} 的梯度形状: {inp.grad.shape}")
                has_valid_grad = True
                break
        
        if not has_valid_grad:
            print(f"  {Colors.WARNING}警告: 未检测到有效梯度{Colors.ENDC}")
        
        # 记录Riemann的性能结果
        self.perf_collector.add_result(test_name, graph_size, rm_forward_time, rm_backward_time, framework="Riemann")
        
        # ===== PyTorch 版本 =====
        if TORCH_AVAILABLE:
            print("  PyTorch 版本:")
            # 设置相同的随机种子
            torch.manual_seed(42)
            # 创建相同规模的输入数据
            x_np = np.random.randn(input_size) * 0.1
            x_torch = torch.tensor(x_np, requires_grad=True)
            
            # 只统计计算图创建和前向传播的时间
            start_time = time.time()
            output_torch, _ = create_torch_graph(x_torch, depth=depth, width=width)
            torch_forward_time = time.time() - start_time
            print(f"  前向计算时间: {torch_forward_time:.6f}秒")
            
            # 测试反向传播时间
            start_time = time.time()
            output_torch.backward()
            torch_backward_time = time.time() - start_time
            print(f"  反向传播时间: {torch_backward_time:.6f}秒")
            
            # 记录PyTorch的性能结果，使用相同的图大小
            self.perf_collector.add_result(test_name, graph_size, torch_forward_time, torch_backward_time, framework="PyTorch")
            
            # 比较前向计算值和梯度值的一致性
            # 前向计算值比较
            forward_consistent = values_are_close(output_rm.item(), output_torch.item())
            
            # 梯度值比较
            grad_consistent = False
            if has_valid_grad and x_torch.grad is not None:
                # 比较第一个输入张量的梯度
                grad_consistent = values_are_close(inputs_rm[0].grad, x_torch.grad)
            
            # 显示一致性比较结果
            forward_status = f"{Colors.OKGREEN}一致{Colors.ENDC}" if forward_consistent else f"{Colors.FAIL}不一致{Colors.ENDC}"
            grad_status = f"{Colors.OKGREEN}一致{Colors.ENDC}" if grad_consistent else f"{Colors.FAIL}不一致{Colors.ENDC}"
            print(f"  前向计算值比较: {forward_status}")
            print(f"  梯度值比较: {grad_status}")
            
            # 保存比较结果
            self.perf_collector.add_comparison_result(test_name, forward_consistent, grad_consistent)
    
    def test_03_performance_large_graph(self):
        """测试大型计算图的性能"""
        test_name = "大型计算图"

        if IS_RUNNING_AS_SCRIPT:
            # 测试参数
            input_size = 30
            depth = 7
            width = 4
        else:
            # 测试参数
            input_size = 10
            depth = 5
            width = 2

        # 打印详细的测试标题信息
        print(f"\n{Colors.OKBLUE}测试: {test_name} - 节点数: {count_graph_nodes(create_complex_graph(input_size, depth, width, seed=42)[0])}, 深度: {depth}, 宽度: {width}, 输入形状: ({input_size},){Colors.ENDC}")
        
        # ===== Riemann 版本 =====
        print("  Riemann 版本:")
        # 创建并测试大型计算图 - 只统计计算图创建和前向传播的时间
        start_time = time.time()
        output_rm, inputs_rm = create_complex_graph(input_size=input_size, depth=depth, width=width, seed=42)
        rm_forward_time = time.time() - start_time
        
        # 在时间统计之外计算图节点数量（不再打印，因为标题中已有）
        graph_size = count_graph_nodes(output_rm)
        # 移除重复的节点数量打印
        print(f"  前向计算时间: {rm_forward_time:.6f}秒")
        
        # 测试反向传播时间
        start_time = time.time()
        output_rm.backward()
        rm_backward_time = time.time() - start_time
        print(f"  反向传播时间: {rm_backward_time:.6f}秒")
        
        # 验证梯度是否正确计算（时间不计入统计）
        has_valid_grad = False
        for i, inp in enumerate(inputs_rm):
            if inp.grad is not None:
                # print(f"  输入张量 {i} 的梯度形状: {inp.grad.shape}")
                has_valid_grad = True
                break
        
        if not has_valid_grad:
            print(f"  {Colors.WARNING}警告: 未检测到有效梯度{Colors.ENDC}")
        
        # 记录Riemann的性能结果
        self.perf_collector.add_result(test_name, graph_size, rm_forward_time, rm_backward_time, framework="Riemann")
        
        # ===== PyTorch 版本 =====
        if TORCH_AVAILABLE:
            print("  PyTorch 版本:")
            # 设置相同的随机种子
            torch.manual_seed(42)
            # 创建相同规模的输入数据
            x_np = np.random.randn(input_size) * 0.1
            x_torch = torch.tensor(x_np, requires_grad=True)
            
            # 只统计计算图创建和前向传播的时间
            start_time = time.time()
            output_torch, _ = create_torch_graph(x_torch, depth=depth, width=width)
            torch_forward_time = time.time() - start_time
            print(f"  前向计算时间: {torch_forward_time:.6f}秒")
            
            # 测试反向传播时间
            start_time = time.time()
            output_torch.backward()
            torch_backward_time = time.time() - start_time
            print(f"  反向传播时间: {torch_backward_time:.6f}秒")
            
            # 记录PyTorch的性能结果，使用相同的图大小
            self.perf_collector.add_result(test_name, graph_size, torch_forward_time, torch_backward_time, framework="PyTorch")
            
            # 比较前向计算值和梯度值的一致性
            # 前向计算值比较
            forward_consistent = values_are_close(output_rm.item(), output_torch.item())
            
            # 梯度值比较
            grad_consistent = False
            if has_valid_grad and x_torch.grad is not None:
                # 比较第一个输入张量的梯度
                grad_consistent = values_are_close(inputs_rm[0].grad, x_torch.grad)
            
            # 显示一致性比较结果
            forward_status = f"{Colors.OKGREEN}一致{Colors.ENDC}" if forward_consistent else f"{Colors.FAIL}不一致{Colors.ENDC}"
            grad_status = f"{Colors.OKGREEN}一致{Colors.ENDC}" if grad_consistent else f"{Colors.FAIL}不一致{Colors.ENDC}"
            print(f"  前向计算值比较: {forward_status}")
            print(f"  梯度值比较: {grad_status}")
            
            # 保存比较结果
            self.perf_collector.add_comparison_result(test_name, forward_consistent, grad_consistent)
    
    def test_04_performance_comparison_summary(self):
        """性能比较汇总"""
        if not TORCH_AVAILABLE:
            return
        
        print(f"\n{Colors.OKBLUE}测试: 性能比较汇总{Colors.ENDC}")
        print("\n性能比较详情:")
        
        # 按测试名称分组结果 - 确保小型->中型->大型的顺序
        test_order = {"小型计算图": 0, "中型计算图": 1, "大型计算图": 2}
        by_test_name = {}
        for result in sorted(self.perf_collector.results, key=lambda x: test_order.get(x['test_name'], 999)):
            if result['test_name'] not in by_test_name:
                by_test_name[result['test_name']] = {}
            by_test_name[result['test_name']][result['framework']] = result
        
        # 打印比较结果
        for test_name in sorted(by_test_name.keys(), key=lambda x: test_order.get(x, 999)):
            frameworks = by_test_name[test_name]
            if 'Riemann' in frameworks and 'PyTorch' in frameworks:
                rm = frameworks['Riemann']
                torch = frameworks['PyTorch']
                
                forward_ratio = rm['forward_time'] / torch['forward_time'] if torch['forward_time'] > 0 else float('inf')
                backward_ratio = rm['backward_time'] / torch['backward_time'] if torch['backward_time'] > 0 else float('inf')
                total_ratio = rm['total_time'] / torch['total_time'] if torch['total_time'] > 0 else float('inf')
                
                print(f"\n{test_name}:")
                print(f"  Riemann前向时间 / PyTorch前向时间: {forward_ratio:.2f}x")
                print(f"  Riemann反向时间 / PyTorch反向时间: {backward_ratio:.2f}x")
                print(f"  Riemann总时间 / PyTorch总时间: {total_ratio:.2f}x")
    
    @classmethod
    def tearDownClass(cls):
        """所有测试结束后打印一次汇总表"""
        # 确保输出被重新启用
        sys.stdout = cls.original_stdout
        cls.perf_collector.print_summary()

# 用于独立运行测试的代码
if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True

    # 清屏
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{Colors.HEADER}===== Riemann 反向传播性能测试 ====={Colors.ENDC}")
    print(f"测试日期: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"PyTorch可用: {TORCH_AVAILABLE}")
    
    # 运行测试但禁用默认输出
    test_runner = unittest.TextTestRunner(stream=open(os.devnull, 'w'), verbosity=0)
    unittest.main(argv=[sys.argv[0]], testRunner=test_runner, exit=False)
    
    # 打印完成消息
    print(f"\n{Colors.HEADER}===== 性能测试完成 ====={Colors.ENDC}")