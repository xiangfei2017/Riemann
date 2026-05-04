import unittest
import numpy as np
import time
import sys, os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 导入riemann模块
try:
    import riemann as rm
    import riemann.nn as rm_nn
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    import torch.nn as torch_nn
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的MultiheadAttention")
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
            header_width = self._get_display_width(header)
            padding = col_widths[i] - header_width
            header_line += header + " " * padding
        print(header_line)
        print("-"*total_width)
        
        for func_name, stats in self.function_stats.items():
            pass_rate = stats["passed"]/stats["total"]*100 if stats["total"] > 0 else 0
            status_color = Colors.OKGREEN if pass_rate == 100 else Colors.FAIL
            
            func_name_display = func_name
            func_name_width = self._get_display_width(func_name_display)
            func_name_padding = col_widths[0] - func_name_width
            
            pass_total_display = f"{stats['passed']}/{stats['total']}"
            pass_total_width = self._get_display_width(pass_total_display)
            pass_total_padding = col_widths[1] - pass_total_width
            
            pass_rate_display = f"{pass_rate:.2f}"
            pass_rate_width = self._get_display_width(pass_rate_display)
            pass_rate_padding = col_widths[2] - pass_rate_width
            
            time_display = f"{stats['time']:.4f}"
            time_width = self._get_display_width(time_display)
            time_padding = col_widths[3] - time_width
            
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
def compare_values(rm_result, torch_result, atol=1e-4, rtol=1e-4):
    """比较Riemann和PyTorch的值是否接近，同时检查形状是否一致"""
    if not TORCH_AVAILABLE:
        return rm_result is not None
    
    if rm_result is None and torch_result is None:
        return True
    if rm_result is None or torch_result is None:
        return False
    
    if isinstance(rm_result, (list, tuple)) and isinstance(torch_result, (list, tuple)):
        if len(rm_result) != len(torch_result):
            return False
        
        all_passed = True
        for r, t in zip(rm_result, torch_result):
            if not compare_values(r, t, atol, rtol):
                all_passed = False
                break
        
        return all_passed
    
    if rm_result.shape != torch_result.shape:
        return False
    
    # 处理需要梯度的张量
    if hasattr(rm_result, 'requires_grad') and rm_result.requires_grad:
        rm_data = rm_result.detach().numpy()
    else:
        rm_data = rm_result.numpy()
    
    if hasattr(torch_result, 'requires_grad') and torch_result.requires_grad:
        torch_data = torch_result.detach().numpy()
    else:
        torch_data = torch_result.numpy()
    
    try:
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

def sync_parameters(rm_attn, torch_attn, embed_dim, kdim=None):
    """同步Riemann和PyTorch的参数"""
    if rm_attn._qkv_same_embed_dim:
        rm_attn.in_proj_weight.data[:] = rm.tensor(torch_attn.in_proj_weight.detach().numpy())
        if torch_attn.in_proj_bias is not None:
            rm_attn.in_proj_bias.data[:] = rm.tensor(torch_attn.in_proj_bias.detach().numpy())
    else:
        rm_attn.q_proj_weight.data[:] = rm.tensor(torch_attn.q_proj_weight.detach().numpy())
        rm_attn.k_proj_weight.data[:] = rm.tensor(torch_attn.k_proj_weight.detach().numpy())
        rm_attn.v_proj_weight.data[:] = rm.tensor(torch_attn.v_proj_weight.detach().numpy())
        if torch_attn.in_proj_bias is not None:
            torch_bias = torch_attn.in_proj_bias.detach().numpy()
            rm_attn.q_proj_bias.data[:] = rm.tensor(torch_bias[:embed_dim])
            rm_attn.k_proj_bias.data[:] = rm.tensor(torch_bias[embed_dim:embed_dim+embed_dim])
            rm_attn.v_proj_bias.data[:] = rm.tensor(torch_bias[embed_dim+embed_dim:])
    
    rm_attn.out_proj.weight.data[:] = rm.tensor(torch_attn.out_proj.weight.detach().numpy())
    if torch_attn.out_proj.bias is not None:
        rm_attn.out_proj.bias.data[:] = rm.tensor(torch_attn.out_proj.bias.detach().numpy())
    
    if hasattr(torch_attn, 'bias_k') and torch_attn.bias_k is not None:
        rm_attn.bias_k.data[:] = rm.tensor(torch_attn.bias_k.detach().numpy())
        rm_attn.bias_v.data[:] = rm.tensor(torch_attn.bias_v.detach().numpy())


def check_forward_backward(rm_tensors, torch_tensors, rm_attn, torch_attn, rm_output, torch_output,
                          test_name, check_params=True):
    """
    测试前向和反向传播的一致性
    
    Args:
        rm_tensors: Riemann输入张量列表 [rm_q, rm_k, rm_v]
        torch_tensors: PyTorch输入张量列表 [torch_q, torch_k, torch_v]
        rm_attn: Riemann MultiheadAttention模块
        torch_attn: PyTorch MultiheadAttention模块
        rm_output: Riemann前向输出
        torch_output: PyTorch前向输出
        test_name: 测试名称
        check_params: 是否检查参数梯度
    
    Returns:
        tuple: (forward_passed, backward_passed, input_grad_passed, param_grad_passed)
    """
    rm_q, rm_k, rm_v = rm_tensors
    torch_q, torch_k, torch_v = torch_tensors
    
    # 测试前向输出
    forward_passed = compare_values(rm_output, torch_output)
    
    if not forward_passed:
        return False, False, False, False
    
    # 反向传播测试
    rm_loss = rm_output.sum()
    torch_loss = torch_output.sum()
    
    rm_loss.backward()
    torch_loss.backward()
    
    # 测试输入梯度
    input_grad_passed = True
    if rm_q.grad is not None and torch_q.grad is not None:
        input_grad_passed = input_grad_passed and compare_values(rm_q.grad, torch_q.grad)
    if rm_k.grad is not None and torch_k.grad is not None:
        input_grad_passed = input_grad_passed and compare_values(rm_k.grad, torch_k.grad)
    if rm_v.grad is not None and torch_v.grad is not None:
        input_grad_passed = input_grad_passed and compare_values(rm_v.grad, torch_v.grad)
    
    # 测试参数梯度
    param_grad_passed = True
    if check_params:
        if rm_attn._qkv_same_embed_dim:
            if rm_attn.in_proj_weight.grad is not None:
                param_grad_passed = param_grad_passed and compare_values(
                    rm_attn.in_proj_weight.grad, torch_attn.in_proj_weight.grad)
        else:
            if rm_attn.q_proj_weight.grad is not None:
                param_grad_passed = param_grad_passed and compare_values(
                    rm_attn.q_proj_weight.grad, torch_attn.q_proj_weight.grad)
                param_grad_passed = param_grad_passed and compare_values(
                    rm_attn.k_proj_weight.grad, torch_attn.k_proj_weight.grad)
                param_grad_passed = param_grad_passed and compare_values(
                    rm_attn.v_proj_weight.grad, torch_attn.v_proj_weight.grad)
        if rm_attn.out_proj.weight.grad is not None:
            param_grad_passed = param_grad_passed and compare_values(
                rm_attn.out_proj.weight.grad, torch_attn.out_proj.weight.grad)
    
    backward_passed = input_grad_passed and param_grad_passed
    
    return forward_passed, backward_passed, input_grad_passed, param_grad_passed

class TestTransformerMHA(unittest.TestCase):
    def setUp(self):
        # 设置随机种子以确保结果可重复
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch.set_default_dtype(torch.float32)
        self.current_test_name = self._testMethodName
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")
                
    def tearDown(self):
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def _run_test_case(self, case_name, **kwargs):
        """运行单个测试用例，测试前向和反向一致性"""
        embed_dim = kwargs.get('embed_dim', 128)
        num_heads = kwargs.get('num_heads', 4)
        batch_size = kwargs.get('batch_size', 8)
        seq_len = kwargs.get('seq_len', 5)
        kdim = kwargs.get('kdim', None)
        vdim = kwargs.get('vdim', None)
        bias = kwargs.get('bias', True)
        add_bias_kv = kwargs.get('add_bias_kv', False)
        add_zero_attn = kwargs.get('add_zero_attn', False)
        batch_first = kwargs.get('batch_first', False)
        dropout = kwargs.get('dropout', 0.0)
        use_key_padding_mask = kwargs.get('use_key_padding_mask', False)
        use_attn_mask_2d = kwargs.get('use_attn_mask_2d', False)
        need_weights = kwargs.get('need_weights', True)
        average_attn_weights = kwargs.get('average_attn_weights', True)
        use_same_qkv = kwargs.get('use_same_qkv', False)
        is_causal = kwargs.get('is_causal', False)
        
        start_time = time.time()
        
        try:
            # 确定输入形状
            if batch_first:
                input_shape_q = (batch_size, seq_len, embed_dim)
                if kdim is None:
                    input_shape_kv = (batch_size, seq_len, embed_dim)
                else:
                    input_shape_kv = (batch_size, seq_len, kdim if kdim is not None else embed_dim)
            else:
                input_shape_q = (seq_len, batch_size, embed_dim)
                if kdim is None:
                    input_shape_kv = (seq_len, batch_size, embed_dim)
                else:
                    input_shape_kv = (seq_len, batch_size, kdim if kdim is not None else embed_dim)
            
            # 创建输入数据
            if use_same_qkv:
                # q/k/v 是同一对象
                numpy_x = np.random.randn(*input_shape_q).astype(np.float32)
                
                # Riemann 张量
                rm_x = rm.tensor(numpy_x, requires_grad=True)
                rm_q = rm_k = rm_v = rm_x
                
                # PyTorch 张量
                torch_x = torch.tensor(numpy_x, requires_grad=True)
                torch_q = torch_k = torch_v = torch_x
            else:
                numpy_q = np.random.randn(*input_shape_q).astype(np.float32)
                numpy_k = np.random.randn(*input_shape_kv).astype(np.float32)
                numpy_v = np.random.randn(*input_shape_kv).astype(np.float32)
                
                # Riemann 张量
                rm_q = rm.tensor(numpy_q, requires_grad=True)
                rm_k = rm.tensor(numpy_k, requires_grad=True)
                rm_v = rm.tensor(numpy_v, requires_grad=True)
                
                # PyTorch 张量
                torch_q = torch.tensor(numpy_q, requires_grad=True)
                torch_k = torch.tensor(numpy_k, requires_grad=True)
                torch_v = torch.tensor(numpy_v, requires_grad=True)
            
            # 创建掩码
            rm_key_padding_mask = None
            torch_key_padding_mask = None
            if use_key_padding_mask:
                mask_shape = (batch_size, seq_len)
                # 使用float类型与attn_mask保持一致，避免PyTorch警告
                mask_data = np.random.choice([0.0, float('-inf')], size=mask_shape, p=[0.8, 0.2]).astype(np.float32)
                rm_key_padding_mask = rm.tensor(mask_data)
                torch_key_padding_mask = torch.tensor(mask_data)
            
            rm_attn_mask = None
            torch_attn_mask = None
            if use_attn_mask_2d:
                mask_shape = (seq_len, seq_len)
                mask_data = np.random.choice([0.0, -1e4], size=mask_shape, p=[0.8, 0.2]).astype(np.float32)
                rm_attn_mask = rm.tensor(mask_data)
                torch_attn_mask = torch.tensor(mask_data)
            
            # 创建注意力模块
            rm_attn = rm_nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                kdim=kdim,
                vdim=vdim,
                batch_first=batch_first
            )
            
            torch_attn = torch_nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                kdim=kdim,
                vdim=vdim,
                batch_first=batch_first
            )
            
            # 同步参数
            sync_parameters(rm_attn, torch_attn, embed_dim, kdim)
            
            # PyTorch 要求 is_causal 时需要同时提供 attn_mask
            if is_causal:
                # 创建明确的因果掩码
                tgt_len = seq_len
                src_len = seq_len
                np_causal_mask = np.triu(np.ones((tgt_len, src_len), dtype=np.float32), k=1) * -1e4
                if torch_attn_mask is not None:
                    # 叠加到现有的 attn_mask
                    np_causal_mask = np_causal_mask + torch_attn_mask.numpy()
                torch_attn_mask = torch.tensor(np_causal_mask)
                rm_attn_mask_actual = rm.tensor(np_causal_mask)
                # 两者都使用显式掩码，不使用 is_causal
                is_causal_actual = False
            else:
                rm_attn_mask_actual = rm_attn_mask
                is_causal_actual = False
            
            # 前向传播
            rm_output, rm_attn_weights = rm_attn(
                rm_q, rm_k, rm_v,
                key_padding_mask=rm_key_padding_mask,
                need_weights=need_weights,
                attn_mask=rm_attn_mask_actual,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal_actual
            )
            
            torch_output, torch_attn_weights = torch_attn(
                torch_q, torch_k, torch_v,
                key_padding_mask=torch_key_padding_mask,
                need_weights=need_weights,
                attn_mask=torch_attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal_actual
            )
            
            # 比较前向输出
            passed_output = compare_values(rm_output, torch_output)
            passed_weights = True
            if need_weights:
                passed_weights = compare_values(rm_attn_weights, torch_attn_weights)
            
            # 反向传播
            rm_loss = rm_output.sum()
            rm_loss.backward()
            
            torch_loss = torch_output.sum()
            torch_loss.backward()
            
            # 比较梯度
            passed_q_grad = compare_values(rm_q.grad, torch_q.grad)
            passed_k_grad = compare_values(rm_k.grad, torch_k.grad)
            passed_v_grad = compare_values(rm_v.grad, torch_v.grad)
            
            # 比较参数梯度
            passed_param_grad = True
            if rm_attn._qkv_same_embed_dim:
                passed_param_grad = passed_param_grad and compare_values(rm_attn.in_proj_weight.grad, torch_attn.in_proj_weight.grad)
            else:
                passed_param_grad = passed_param_grad and compare_values(rm_attn.q_proj_weight.grad, torch_attn.q_proj_weight.grad)
                passed_param_grad = passed_param_grad and compare_values(rm_attn.k_proj_weight.grad, torch_attn.k_proj_weight.grad)
                passed_param_grad = passed_param_grad and compare_values(rm_attn.v_proj_weight.grad, torch_attn.v_proj_weight.grad)
            passed_param_grad = passed_param_grad and compare_values(rm_attn.out_proj.weight.grad, torch_attn.out_proj.weight.grad)
            
            passed = passed_output and passed_weights and passed_q_grad and passed_k_grad and passed_v_grad and passed_param_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(f"{case_name} - 输出", passed_output)
                if need_weights:
                    stats.add_result(f"{case_name} - 注意力权重", passed_weights)
                stats.add_result(f"{case_name} - 输入梯度", passed_q_grad and passed_k_grad and passed_v_grad)
                stats.add_result(f"{case_name} - 参数梯度", passed_param_grad)
                
                status = "通过" if passed else "失败"
                status_color = Colors.OKGREEN if passed else Colors.FAIL
                print(f"测试用例: {case_name} - {status_color}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if passed_output:
                    rm_out_np = rm_output.detach().numpy() if rm_output.requires_grad else rm_output.numpy()
                    torch_out_np = torch_output.detach().numpy() if torch_output.requires_grad else torch_output.numpy()
                    print(f"  输出最大差异: {np.max(np.abs(rm_out_np - torch_out_np)):.2e}")
            
            # 断言确保测试通过
            self.assertTrue(passed_output, f"输出不一致: {case_name}")
            self.assertTrue(passed_weights, f"注意力权重不一致: {case_name}")
            self.assertTrue(passed_q_grad, f"查询梯度不一致: {case_name}")
            self.assertTrue(passed_k_grad, f"键梯度不一致: {case_name}")
            self.assertTrue(passed_v_grad, f"值梯度不一致: {case_name}")
            self.assertTrue(passed_param_grad, f"参数梯度不一致: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_basic(self):
        """测试基础配置"""
        self._run_test_case("基础配置")
    
    def test_no_bias(self):
        """测试bias=False"""
        self._run_test_case("bias=False", bias=False)
    
    def test_batch_first(self):
        """测试batch_first=True"""
        self._run_test_case("batch_first=True", batch_first=True)
    
    def test_add_bias_kv(self):
        """测试add_bias_kv=True"""
        self._run_test_case("add_bias_kv=True", add_bias_kv=True)
    
    def test_add_zero_attn(self):
        """测试add_zero_attn=True"""
        self._run_test_case("add_zero_attn=True", add_zero_attn=True)
    
    def test_kdim_vdim_different(self):
        """测试kdim和vdim不同"""
        self._run_test_case("kdim=64, vdim=64", kdim=64, vdim=64)
    
    def test_batch_first_no_bias(self):
        """测试batch_first=True和bias=False组合"""
        self._run_test_case("batch_first=True, bias=False", batch_first=True, bias=False)
    
    def test_key_padding_mask(self):
        """测试带key_padding_mask"""
        self._run_test_case("带key_padding_mask", use_key_padding_mask=True)
    
    def test_attn_mask_2d(self):
        """测试带2D attn_mask"""
        self._run_test_case("带2D attn_mask", use_attn_mask_2d=True)
    
    def test_no_need_weights(self):
        """测试need_weights=False"""
        self._run_test_case("need_weights=False", need_weights=False)
    
    def test_no_average_attn_weights(self):
        """测试average_attn_weights=False"""
        self._run_test_case("average_attn_weights=False", average_attn_weights=False)
    
    def test_same_qkv(self):
        """测试query=key=value是同一对象"""
        self._run_test_case("query=key=value同一对象", use_same_qkv=True)
    
    def test_is_causal(self):
        """测试is_causal=True（因果掩码）"""
        self._run_test_case("is_causal=True", is_causal=True)
    
    def test_is_causal_with_attn_mask(self):
        """测试is_causal=True与attn_mask叠加"""
        self._run_test_case("is_causal=True与attn_mask叠加", is_causal=True, use_attn_mask_2d=True)
    
    def test_is_causal_no_need_weights(self):
        """测试is_causal=True且need_weights=False"""
        self._run_test_case("is_causal=True且need_weights=False", is_causal=True, need_weights=False)
    
    def test_different_seq_len(self):
        """测试不同的tgt_len和src_len（交叉注意力）"""
        # 手动创建测试，因为_run_test_case假设seq_len相同
        embed_dim, num_heads = 128, 4
        tgt_len, src_len, batch_size = 5, 10, 8
        
        np.random.seed(42)
        numpy_q = np.random.randn(tgt_len, batch_size, embed_dim).astype(np.float32)
        numpy_k = np.random.randn(src_len, batch_size, embed_dim).astype(np.float32)
        numpy_v = np.random.randn(src_len, batch_size, embed_dim).astype(np.float32)
        
        rm_q = rm.tensor(numpy_q, requires_grad=True)
        rm_k = rm.tensor(numpy_k, requires_grad=True)
        rm_v = rm.tensor(numpy_v, requires_grad=True)
        
        rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch_q = torch.tensor(numpy_q, requires_grad=True)
            torch_k = torch.tensor(numpy_k, requires_grad=True)
            torch_v = torch.tensor(numpy_v, requires_grad=True)
            
            torch_attn = torch_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            
            # 同步参数
            sync_parameters(rm_attn, torch_attn, embed_dim)
            
            rm_output, _ = rm_attn(rm_q, rm_k, rm_v)
            torch_output, _ = torch_attn(torch_q, torch_k, torch_v)
            
            # 验证输出形状
            self.assertEqual(rm_output.shape, (tgt_len, batch_size, embed_dim))
            passed = compare_values(rm_output, torch_output)
            self.assertTrue(passed)
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("不同seq_len对比", passed)
        else:
            rm_output, _ = rm_attn(rm_q, rm_k, rm_v)
            # 验证输出形状
            self.assertEqual(rm_output.shape, (tgt_len, batch_size, embed_dim))
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("不同seq_len形状", True)
    
    def test_3d_attn_mask(self):
        """测试3D attn_mask (batch_size * num_heads, tgt_len, src_len)"""
        embed_dim, num_heads = 128, 4
        seq_len, batch_size = 5, 8
        
        np.random.seed(42)
        numpy_q = np.random.randn(seq_len, batch_size, embed_dim).astype(np.float32)
        numpy_k = np.random.randn(seq_len, batch_size, embed_dim).astype(np.float32)
        numpy_v = np.random.randn(seq_len, batch_size, embed_dim).astype(np.float32)
        
        # 3D mask: (batch_size * num_heads, tgt_len, src_len)
        mask_shape = (batch_size * num_heads, seq_len, seq_len)
        mask_data = np.random.choice([0.0, -1e4], size=mask_shape, p=[0.8, 0.2]).astype(np.float32)
        rm_attn_mask = rm.tensor(mask_data)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch_q = torch.tensor(numpy_q, requires_grad=True)
            torch_k = torch.tensor(numpy_k, requires_grad=True)
            torch_v = torch.tensor(numpy_v, requires_grad=True)
            torch_attn = torch_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            
            rm_q = rm.tensor(numpy_q, requires_grad=True)
            rm_k = rm.tensor(numpy_k, requires_grad=True)
            rm_v = rm.tensor(numpy_v, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            sync_parameters(rm_attn, torch_attn, embed_dim)
            
            rm_output, _ = rm_attn(rm_q, rm_k, rm_v, attn_mask=rm_attn_mask)
            
            torch_attn_mask = torch.tensor(mask_data)
            torch_output, _ = torch_attn(torch_q, torch_k, torch_v, attn_mask=torch_attn_mask)
            
            # 测试前向和反向
            f_passed, b_passed, ig_passed, pg_passed = check_forward_backward(
                [rm_q, rm_k, rm_v], [torch_q, torch_k, torch_v],
                rm_attn, torch_attn, rm_output, torch_output, "3D attn_mask"
            )
            
            self.assertTrue(f_passed, "前向传播不一致")
            self.assertTrue(b_passed, "反向传播不一致")
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("3D attn_mask前向", f_passed)
                stats.add_result("3D attn_mask输入梯度", ig_passed)
                stats.add_result("3D attn_mask参数梯度", pg_passed)
        else:
            rm_q = rm.tensor(numpy_q, requires_grad=True)
            rm_k = rm.tensor(numpy_k, requires_grad=True)
            rm_v = rm.tensor(numpy_v, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            rm_output, _ = rm_attn(rm_q, rm_k, rm_v, attn_mask=rm_attn_mask)
            
            self.assertIsNotNone(rm_output)
            self.assertEqual(rm_output.shape, (seq_len, batch_size, embed_dim))
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("3D attn_mask形状", True)
    
    def test_key_padding_mask_with_is_causal(self):
        """测试key_padding_mask与is_causal同时使用"""
        self._run_test_case("key_padding_mask与is_causal", use_key_padding_mask=True, is_causal=True)
    
    def test_add_bias_kv_and_add_zero_attn(self):
        """测试add_bias_kv与add_zero_attn同时使用"""
        self._run_test_case("add_bias_kv与add_zero_attn", add_bias_kv=True, add_zero_attn=True)
    
    def test_single_head(self):
        """测试单头注意力(num_heads=1)"""
        self._run_test_case("单头注意力", embed_dim=128, num_heads=1)
    
    def test_single_seq_len(self):
        """测试序列长度为1的边界情况"""
        self._run_test_case("序列长度为1", seq_len=1)
    
    def test_causal_mask_effect(self):
        """验证因果掩码是否正确阻止对未来位置的关注"""
        embed_dim, num_heads = 8, 2
        seq_len, batch_size = 4, 2
        
        np.random.seed(42)
        numpy_x = np.random.randn(seq_len, batch_size, embed_dim).astype(np.float32)
        
        rm_x = rm.tensor(numpy_x, requires_grad=True)
        
        rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        rm_output, rm_weights = rm_attn(rm_x, rm_x, rm_x, is_causal=True, need_weights=True, average_attn_weights=False)
        
        # 验证权重形状: (batch_size, num_heads, tgt_len, src_len)
        self.assertEqual(rm_weights.shape, (batch_size, num_heads, seq_len, seq_len))
        
        # 转换为numpy检查上三角是否为0（softmax后的权重）
        weights_np = rm_weights.detach().numpy()
        
        # 检查上三角（不包括对角线）是否接近0（被mask掉）
        all_masked = True
        for b in range(batch_size):
            for h in range(num_heads):
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):  # 上三角，j > i
                        if abs(weights_np[b, h, i, j]) > 1e-5:
                            all_masked = False
                        self.assertAlmostEqual(weights_np[b, h, i, j], 0.0, places=5,
                                               msg=f"位置({b},{h},{i},{j})应该被mask为0")
        if IS_RUNNING_AS_SCRIPT:
            stats.add_result("因果掩码效果", all_masked)
    
    def test_float_key_padding_mask(self):
        """测试float类型的key_padding_mask（与bool类型行为一致）"""
        embed_dim, num_heads = 64, 4
        seq_len, batch_size = 5, 8
        
        np.random.seed(42)
        numpy_x = np.random.randn(seq_len, batch_size, embed_dim).astype(np.float32)
        
        # float类型的key_padding_mask: 0表示保留，-inf表示mask
        key_padding_mask_float = np.zeros((batch_size, seq_len), dtype=np.float32)
        key_padding_mask_float[:, -2:] = float('-inf')  # 最后两个位置mask
        rm_key_padding_mask = rm.tensor(key_padding_mask_float)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch_x = torch.tensor(numpy_x, requires_grad=True)
            torch_attn = torch_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            
            # 先创建Riemann模块并同步PyTorch的参数
            rm_x = rm.tensor(numpy_x, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            sync_parameters(rm_attn, torch_attn, embed_dim)
            
            rm_output, _ = rm_attn(rm_x, rm_x, rm_x, key_padding_mask=rm_key_padding_mask)
            
            torch_key_padding_mask = torch.tensor(key_padding_mask_float)
            torch_output, _ = torch_attn(torch_x, torch_x, torch_x, key_padding_mask=torch_key_padding_mask)
            
            # 测试前向和反向
            f_passed, b_passed, ig_passed, pg_passed = check_forward_backward(
                [rm_x, rm_x, rm_x], [torch_x, torch_x, torch_x],
                rm_attn, torch_attn, rm_output, torch_output, "float key_padding_mask"
            )
            
            self.assertTrue(f_passed, "前向传播不一致")
            self.assertTrue(b_passed, "反向传播不一致")
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("float key_padding_mask前向", f_passed)
                stats.add_result("float key_padding_mask输入梯度", ig_passed)
                stats.add_result("float key_padding_mask参数梯度", pg_passed)
        else:
            rm_x = rm.tensor(numpy_x, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            rm_output, _ = rm_attn(rm_x, rm_x, rm_x, key_padding_mask=rm_key_padding_mask)
            
            self.assertIsNotNone(rm_output)
            self.assertEqual(rm_output.shape, (seq_len, batch_size, embed_dim))
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("float key_padding_mask形状", True)
    
    def test_3d_attn_mask_batch_first(self):
        """测试3D attn_mask与batch_first=True组合"""
        embed_dim, num_heads = 64, 4
        seq_len, batch_size = 5, 8
        
        np.random.seed(42)
        # batch_first=True: (batch_size, seq_len, embed_dim)
        numpy_x = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        
        # 3D mask: (batch_size * num_heads, tgt_len, src_len)
        mask_shape = (batch_size * num_heads, seq_len, seq_len)
        mask_data = np.random.choice([0.0, -1e4], size=mask_shape, p=[0.8, 0.2]).astype(np.float32)
        rm_attn_mask = rm.tensor(mask_data)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch_x = torch.tensor(numpy_x, requires_grad=True)
            torch_attn = torch_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            
            rm_x = rm.tensor(numpy_x, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            sync_parameters(rm_attn, torch_attn, embed_dim)
            
            rm_output, _ = rm_attn(rm_x, rm_x, rm_x, attn_mask=rm_attn_mask)
            
            torch_attn_mask = torch.tensor(mask_data)
            torch_output, _ = torch_attn(torch_x, torch_x, torch_x, attn_mask=torch_attn_mask)
            
            # 测试前向和反向
            f_passed, b_passed, ig_passed, pg_passed = check_forward_backward(
                [rm_x, rm_x, rm_x], [torch_x, torch_x, torch_x],
                rm_attn, torch_attn, rm_output, torch_output, "3D attn_mask + batch_first"
            )
            
            self.assertTrue(f_passed, "前向传播不一致")
            self.assertTrue(b_passed, "反向传播不一致")
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("3D attn_mask + batch_first前向", f_passed)
                stats.add_result("3D attn_mask + batch_first输入梯度", ig_passed)
                stats.add_result("3D attn_mask + batch_first参数梯度", pg_passed)
        else:
            rm_x = rm.tensor(numpy_x, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            rm_output, _ = rm_attn(rm_x, rm_x, rm_x, attn_mask=rm_attn_mask)
            
            self.assertEqual(rm_output.shape, (batch_size, seq_len, embed_dim))
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("3D attn_mask + batch_first形状", True)
    
    def test_is_causal_different_seq_len(self):
        """测试is_causal与不同序列长度组合（tgt_len != src_len）"""
        embed_dim, num_heads = 64, 4
        tgt_len, src_len, batch_size = 3, 5, 8
        
        np.random.seed(42)
        numpy_q = np.random.randn(tgt_len, batch_size, embed_dim).astype(np.float32)
        numpy_kv = np.random.randn(src_len, batch_size, embed_dim).astype(np.float32)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch_q = torch.tensor(numpy_q, requires_grad=True)
            torch_k = torch.tensor(numpy_kv, requires_grad=True)
            torch_v = torch.tensor(numpy_kv, requires_grad=True)
            torch_attn = torch_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            
            rm_q = rm.tensor(numpy_q, requires_grad=True)
            rm_k = rm.tensor(numpy_kv, requires_grad=True)
            rm_v = rm.tensor(numpy_kv, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            sync_parameters(rm_attn, torch_attn, embed_dim)
            
            # 当 tgt_len != src_len 时，PyTorch要求提供attn_mask
            # 创建一个合适的causal mask
            attn_mask_data = np.zeros((tgt_len, src_len), dtype=np.float32)
            for i in range(tgt_len):
                for j in range(i + 1, src_len):
                    attn_mask_data[i, j] = float('-inf')
            rm_attn_mask = rm.tensor(attn_mask_data)
            torch_attn_mask = torch.tensor(attn_mask_data)
            
            rm_output, rm_weights = rm_attn(rm_q, rm_k, rm_v, attn_mask=rm_attn_mask, is_causal=True, need_weights=True, average_attn_weights=False)
            torch_output, torch_weights = torch_attn(torch_q, torch_k, torch_v, attn_mask=torch_attn_mask, is_causal=True, need_weights=True, average_attn_weights=False)
            
            # 测试前向和反向
            f_passed, b_passed, ig_passed, pg_passed = check_forward_backward(
                [rm_q, rm_k, rm_v], [torch_q, torch_k, torch_v],
                rm_attn, torch_attn, rm_output, torch_output, "is_causal + different_seq_len"
            )
            
            self.assertTrue(f_passed, "前向传播不一致")
            self.assertTrue(b_passed, "反向传播不一致")
            
            # 验证权重形状和上三角mask
            self.assertEqual(rm_weights.shape, (batch_size, num_heads, tgt_len, src_len))
            weights_np = rm_weights.detach().numpy()
            for b in range(batch_size):
                for h in range(num_heads):
                    for i in range(tgt_len):
                        for j in range(i + 1, src_len):
                            self.assertAlmostEqual(weights_np[b, h, i, j], 0.0, places=5)
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("is_causal + different_seq_len", f_passed and b_passed)
                stats.add_result("is_causal + different_seq_len前向", f_passed)
                stats.add_result("is_causal + different_seq_len输入梯度", ig_passed)
                stats.add_result("is_causal + different_seq_len参数梯度", pg_passed)
        else:
            rm_q = rm.tensor(numpy_q, requires_grad=True)
            rm_k = rm.tensor(numpy_kv, requires_grad=True)
            rm_v = rm.tensor(numpy_kv, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            rm_output, rm_weights = rm_attn(rm_q, rm_k, rm_v, is_causal=True, need_weights=True, average_attn_weights=False)
            
            self.assertEqual(rm_output.shape, (tgt_len, batch_size, embed_dim))
            self.assertEqual(rm_weights.shape, (batch_size, num_heads, tgt_len, src_len))
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("is_causal + different_seq_len形状", True)
    
    def test_no_need_weights_no_average(self):
        """测试need_weights=False + average_attn_weights=False组合"""
        embed_dim, num_heads = 64, 4
        seq_len, batch_size = 5, 8
        
        np.random.seed(42)
        numpy_x = np.random.randn(seq_len, batch_size, embed_dim).astype(np.float32)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch_x = torch.tensor(numpy_x, requires_grad=True)
            torch_attn = torch_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            
            rm_x = rm.tensor(numpy_x, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            sync_parameters(rm_attn, torch_attn, embed_dim)
            
            rm_output, rm_weights = rm_attn(rm_x, rm_x, rm_x, need_weights=False, average_attn_weights=False)
            
            torch_output, torch_weights = torch_attn(torch_x, torch_x, torch_x, need_weights=False, average_attn_weights=False)
            
            self.assertIsNone(torch_weights)
            self.assertIsNone(rm_weights)
            
            # 测试前向和反向
            f_passed, b_passed, ig_passed, pg_passed = check_forward_backward(
                [rm_x, rm_x, rm_x], [torch_x, torch_x, torch_x],
                rm_attn, torch_attn, rm_output, torch_output, "need_weights=False + average=False"
            )
            
            self.assertTrue(f_passed, "前向传播不一致")
            self.assertTrue(b_passed, "反向传播不一致")
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("need_weights=False + average=False前向", f_passed)
                stats.add_result("need_weights=False + average=False输入梯度", ig_passed)
                stats.add_result("need_weights=False + average=False参数梯度", pg_passed)
        else:
            rm_x = rm.tensor(numpy_x, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            rm_output, rm_weights = rm_attn(rm_x, rm_x, rm_x, need_weights=False, average_attn_weights=False)
            
            self.assertEqual(rm_output.shape, (seq_len, batch_size, embed_dim))
            self.assertIsNone(rm_weights)
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("need_weights=False + average=False形状", True)
    
    def test_kdim_vdim_batch_first(self):
        """测试kdim/vdim不同与batch_first=True组合"""
        embed_dim, kdim, vdim, num_heads = 64, 32, 48, 4
        seq_len, batch_size = 5, 8
        
        np.random.seed(42)
        # batch_first=True
        numpy_q = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        numpy_k = np.random.randn(batch_size, seq_len, kdim).astype(np.float32)
        numpy_v = np.random.randn(batch_size, seq_len, vdim).astype(np.float32)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch_q = torch.tensor(numpy_q, requires_grad=True)
            torch_k = torch.tensor(numpy_k, requires_grad=True)
            torch_v = torch.tensor(numpy_v, requires_grad=True)
            
            torch_attn = torch_nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads,
                kdim=kdim, vdim=vdim, batch_first=True
            )
            
            rm_q = rm.tensor(numpy_q, requires_grad=True)
            rm_k = rm.tensor(numpy_k, requires_grad=True)
            rm_v = rm.tensor(numpy_v, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads,
                kdim=kdim, vdim=vdim, batch_first=True
            )
            sync_parameters(rm_attn, torch_attn, embed_dim, kdim=kdim)
            
            rm_output, _ = rm_attn(rm_q, rm_k, rm_v)
            
            torch_output, _ = torch_attn(torch_q, torch_k, torch_v)
            
            # 测试前向和反向
            f_passed, b_passed, ig_passed, pg_passed = check_forward_backward(
                [rm_q, rm_k, rm_v], [torch_q, torch_k, torch_v],
                rm_attn, torch_attn, rm_output, torch_output, "kdim/vdim不同 + batch_first"
            )
            
            self.assertTrue(f_passed, "前向传播不一致")
            self.assertTrue(b_passed, "反向传播不一致")
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("kdim/vdim不同 + batch_first前向", f_passed)
                stats.add_result("kdim/vdim不同 + batch_first输入梯度", ig_passed)
                stats.add_result("kdim/vdim不同 + batch_first参数梯度", pg_passed)
        else:
            rm_q = rm.tensor(numpy_q, requires_grad=True)
            rm_k = rm.tensor(numpy_k, requires_grad=True)
            rm_v = rm.tensor(numpy_v, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads,
                kdim=kdim, vdim=vdim, batch_first=True
            )
            rm_output, _ = rm_attn(rm_q, rm_k, rm_v)
            
            self.assertEqual(rm_output.shape, (batch_size, seq_len, embed_dim))
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("kdim/vdim不同 + batch_first形状", True)
    
    def test_all_masks_combined(self):
        """测试所有掩码同时使用：attn_mask + key_padding_mask + is_causal"""
        embed_dim, num_heads = 64, 4
        seq_len, batch_size = 5, 8
        
        np.random.seed(42)
        numpy_x = np.random.randn(seq_len, batch_size, embed_dim).astype(np.float32)
        
        # 2D attn_mask (k=2表示保留对角线和下对角线)
        attn_mask_data = np.zeros((seq_len, seq_len), dtype=np.float32)
        for i in range(seq_len):
            for j in range(i + 2, seq_len):  # j > i + 1
                attn_mask_data[i, j] = float('-inf')
        
        # float key_padding_mask (与attn_mask类型一致)
        key_padding_mask_data = np.zeros((batch_size, seq_len), dtype=np.float32)
        key_padding_mask_data[:, -1] = float('-inf')  # mask最后一个位置
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch_x = torch.tensor(numpy_x, requires_grad=True)
            torch_attn = torch_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            
            rm_x = rm.tensor(numpy_x, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            sync_parameters(rm_attn, torch_attn, embed_dim)
            
            # PyTorch需要将is_causal的mask合并到attn_mask中
            # 生成causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            combined_attn_mask = torch.tensor(attn_mask_data) + causal_mask
            
            rm_attn_mask = rm.tensor(attn_mask_data)  # Riemann单独处理is_causal
            rm_key_padding_mask = rm.tensor(key_padding_mask_data)
            
            rm_output, rm_weights = rm_attn(
                rm_x, rm_x, rm_x,
                attn_mask=rm_attn_mask,
                key_padding_mask=rm_key_padding_mask,
                is_causal=True,
                need_weights=True,
                average_attn_weights=False
            )
            
            torch_key_padding_mask = torch.tensor(key_padding_mask_data)
            
            torch_output, torch_weights = torch_attn(
                torch_x, torch_x, torch_x,
                attn_mask=combined_attn_mask,
                key_padding_mask=torch_key_padding_mask,
                is_causal=False,  # PyTorch的is_causal已合并到attn_mask
                need_weights=True,
                average_attn_weights=False
            )
            
            # 测试前向和反向
            f_passed, b_passed, ig_passed, pg_passed = check_forward_backward(
                [rm_x, rm_x, rm_x], [torch_x, torch_x, torch_x],
                rm_attn, torch_attn, rm_output, torch_output, "所有掩码组合"
            )
            
            self.assertTrue(f_passed, "前向传播不一致")
            self.assertTrue(b_passed, "反向传播不一致")
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("所有掩码组合", f_passed and b_passed)
                stats.add_result("所有掩码组合前向", f_passed)
                stats.add_result("所有掩码组合输入梯度", ig_passed)
                stats.add_result("所有掩码组合参数梯度", pg_passed)
        else:
            rm_x = rm.tensor(numpy_x, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            rm_attn_mask = rm.tensor(attn_mask_data)
            rm_key_padding_mask = rm.tensor(key_padding_mask_data)
            rm_output, rm_weights = rm_attn(
                rm_x, rm_x, rm_x,
                attn_mask=rm_attn_mask,
                key_padding_mask=rm_key_padding_mask,
                is_causal=True,
                need_weights=True,
                average_attn_weights=False
            )
            
            self.assertEqual(rm_output.shape, (seq_len, batch_size, embed_dim))
            self.assertEqual(rm_weights.shape, (batch_size, num_heads, seq_len, seq_len))
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("所有掩码组合形状", True)
    
    def test_large_num_heads(self):
        """测试大num_heads（小head_dim）"""
        embed_dim, num_heads = 64, 8  # head_dim = 8
        seq_len, batch_size = 5, 8
        
        np.random.seed(42)
        numpy_x = np.random.randn(seq_len, batch_size, embed_dim).astype(np.float32)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch_x = torch.tensor(numpy_x, requires_grad=True)
            torch_attn = torch_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            
            rm_x = rm.tensor(numpy_x, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            sync_parameters(rm_attn, torch_attn, embed_dim)
            
            rm_output, _ = rm_attn(rm_x, rm_x, rm_x)
            
            torch_output, _ = torch_attn(torch_x, torch_x, torch_x)
            
            # 测试前向和反向
            f_passed, b_passed, ig_passed, pg_passed = check_forward_backward(
                [rm_x, rm_x, rm_x], [torch_x, torch_x, torch_x],
                rm_attn, torch_attn, rm_output, torch_output, "大num_heads"
            )
            
            self.assertTrue(f_passed, "前向传播不一致")
            self.assertTrue(b_passed, "反向传播不一致")
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("大num_heads前向", f_passed)
                stats.add_result("大num_heads输入梯度", ig_passed)
                stats.add_result("大num_heads参数梯度", pg_passed)
        else:
            rm_x = rm.tensor(numpy_x, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            rm_output, _ = rm_attn(rm_x, rm_x, rm_x)
            
            self.assertEqual(rm_output.shape, (seq_len, batch_size, embed_dim))
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("大num_heads形状", True)
    
    def test_cross_attention_different_lengths(self):
        """测试交叉注意力场景：query/key/value完全不同且长度都不同"""
        embed_dim, num_heads = 64, 4
        tgt_len, src_len, batch_size = 3, 7, 8
        
        np.random.seed(42)
        numpy_q = np.random.randn(tgt_len, batch_size, embed_dim).astype(np.float32)
        numpy_k = np.random.randn(src_len, batch_size, embed_dim).astype(np.float32)
        numpy_v = np.random.randn(src_len, batch_size, embed_dim).astype(np.float32)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            torch_q = torch.tensor(numpy_q, requires_grad=True)
            torch_k = torch.tensor(numpy_k, requires_grad=True)
            torch_v = torch.tensor(numpy_v, requires_grad=True)
            
            torch_attn = torch_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            
            rm_q = rm.tensor(numpy_q, requires_grad=True)
            rm_k = rm.tensor(numpy_k, requires_grad=True)
            rm_v = rm.tensor(numpy_v, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            sync_parameters(rm_attn, torch_attn, embed_dim)
            
            rm_output, _ = rm_attn(rm_q, rm_k, rm_v)
            
            torch_output, _ = torch_attn(torch_q, torch_k, torch_v)
            
            # 测试前向和反向
            f_passed, b_passed, ig_passed, pg_passed = check_forward_backward(
                [rm_q, rm_k, rm_v], [torch_q, torch_k, torch_v],
                rm_attn, torch_attn, rm_output, torch_output, "交叉注意力不同长度"
            )
            
            self.assertTrue(f_passed, "前向传播不一致")
            self.assertTrue(b_passed, "反向传播不一致")
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("交叉注意力不同长度前向", f_passed)
                stats.add_result("交叉注意力不同长度输入梯度", ig_passed)
                stats.add_result("交叉注意力不同长度参数梯度", pg_passed)
        else:
            rm_q = rm.tensor(numpy_q, requires_grad=True)
            rm_k = rm.tensor(numpy_k, requires_grad=True)
            rm_v = rm.tensor(numpy_v, requires_grad=True)
            rm_attn = rm_nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            rm_output, _ = rm_attn(rm_q, rm_k, rm_v)
            
            self.assertEqual(rm_output.shape, (tgt_len, batch_size, embed_dim))
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result("交叉注意力不同长度形状", True)

if __name__ == '__main__':
    IS_RUNNING_AS_SCRIPT = True
    clear_screen()
    print("=" * 70)
    print("Riemann 与 PyTorch MultiheadAttention 一致性测试")
    print("=" * 70)
    
    unittest.main(exit=False, verbosity=0)
    
    stats.print_summary()
