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
    print("警告: 无法导入PyTorch，将只测试riemann的Transformer层")
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
def compare_values(rm_result, torch_result, atol=5e-4, rtol=2e-3):
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

def sync_encoder_layers(rm_encoder, torch_encoder, num_layers):
    """同步Encoder的所有层参数"""
    for i in range(num_layers):
        rm_layer = rm_encoder.layers[i]
        torch_layer = torch_encoder.layers[i]
        
        # 自注意力
        rm_layer.self_attn.in_proj_weight.data[:] = rm.tensor(torch_layer.self_attn.in_proj_weight.detach().numpy())
        if torch_layer.self_attn.in_proj_bias is not None:
            rm_layer.self_attn.in_proj_bias.data[:] = rm.tensor(torch_layer.self_attn.in_proj_bias.detach().numpy())
        rm_layer.self_attn.out_proj.weight.data[:] = rm.tensor(torch_layer.self_attn.out_proj.weight.detach().numpy())
        if torch_layer.self_attn.out_proj.bias is not None:
            rm_layer.self_attn.out_proj.bias.data[:] = rm.tensor(torch_layer.self_attn.out_proj.bias.detach().numpy())
        
        # FFN
        rm_layer.linear1.weight.data[:] = rm.tensor(torch_layer.linear1.weight.detach().numpy())
        if torch_layer.linear1.bias is not None:
            rm_layer.linear1.bias.data[:] = rm.tensor(torch_layer.linear1.bias.detach().numpy())
        rm_layer.linear2.weight.data[:] = rm.tensor(torch_layer.linear2.weight.detach().numpy())
        if torch_layer.linear2.bias is not None:
            rm_layer.linear2.bias.data[:] = rm.tensor(torch_layer.linear2.bias.detach().numpy())
        
        # LayerNorm
        rm_layer.norm1.weight.data[:] = rm.tensor(torch_layer.norm1.weight.detach().numpy())
        rm_layer.norm1.bias.data[:] = rm.tensor(torch_layer.norm1.bias.detach().numpy())
        rm_layer.norm2.weight.data[:] = rm.tensor(torch_layer.norm2.weight.detach().numpy())
        rm_layer.norm2.bias.data[:] = rm.tensor(torch_layer.norm2.bias.detach().numpy())
    
    # 最后的 norm
    if rm_encoder.norm is not None and torch_encoder.norm is not None:
        rm_encoder.norm.weight.data[:] = rm.tensor(torch_encoder.norm.weight.detach().numpy())
        rm_encoder.norm.bias.data[:] = rm.tensor(torch_encoder.norm.bias.detach().numpy())

def sync_decoder_layers(rm_decoder, torch_decoder, num_layers):
    """同步Decoder的所有层参数"""
    for i in range(num_layers):
        rm_layer = rm_decoder.layers[i]
        torch_layer = torch_decoder.layers[i]
        
        # 自注意力
        rm_layer.self_attn.in_proj_weight.data[:] = rm.tensor(torch_layer.self_attn.in_proj_weight.detach().numpy())
        if torch_layer.self_attn.in_proj_bias is not None:
            rm_layer.self_attn.in_proj_bias.data[:] = rm.tensor(torch_layer.self_attn.in_proj_bias.detach().numpy())
        rm_layer.self_attn.out_proj.weight.data[:] = rm.tensor(torch_layer.self_attn.out_proj.weight.detach().numpy())
        if torch_layer.self_attn.out_proj.bias is not None:
            rm_layer.self_attn.out_proj.bias.data[:] = rm.tensor(torch_layer.self_attn.out_proj.bias.detach().numpy())
        
        # 交叉注意力
        rm_layer.multihead_attn.in_proj_weight.data[:] = rm.tensor(torch_layer.multihead_attn.in_proj_weight.detach().numpy())
        if torch_layer.multihead_attn.in_proj_bias is not None:
            rm_layer.multihead_attn.in_proj_bias.data[:] = rm.tensor(torch_layer.multihead_attn.in_proj_bias.detach().numpy())
        rm_layer.multihead_attn.out_proj.weight.data[:] = rm.tensor(torch_layer.multihead_attn.out_proj.weight.detach().numpy())
        if torch_layer.multihead_attn.out_proj.bias is not None:
            rm_layer.multihead_attn.out_proj.bias.data[:] = rm.tensor(torch_layer.multihead_attn.out_proj.bias.detach().numpy())
        
        # FFN
        rm_layer.linear1.weight.data[:] = rm.tensor(torch_layer.linear1.weight.detach().numpy())
        if torch_layer.linear1.bias is not None:
            rm_layer.linear1.bias.data[:] = rm.tensor(torch_layer.linear1.bias.detach().numpy())
        rm_layer.linear2.weight.data[:] = rm.tensor(torch_layer.linear2.weight.detach().numpy())
        if torch_layer.linear2.bias is not None:
            rm_layer.linear2.bias.data[:] = rm.tensor(torch_layer.linear2.bias.detach().numpy())
        
        # LayerNorm
        rm_layer.norm1.weight.data[:] = rm.tensor(torch_layer.norm1.weight.detach().numpy())
        rm_layer.norm1.bias.data[:] = rm.tensor(torch_layer.norm1.bias.detach().numpy())
        rm_layer.norm2.weight.data[:] = rm.tensor(torch_layer.norm2.weight.detach().numpy())
        rm_layer.norm2.bias.data[:] = rm.tensor(torch_layer.norm2.bias.detach().numpy())
        rm_layer.norm3.weight.data[:] = rm.tensor(torch_layer.norm3.weight.detach().numpy())
        rm_layer.norm3.bias.data[:] = rm.tensor(torch_layer.norm3.bias.detach().numpy())
    
    # 最后的 norm
    if rm_decoder.norm is not None and torch_decoder.norm is not None:
        rm_decoder.norm.weight.data[:] = rm.tensor(torch_decoder.norm.weight.detach().numpy())
        rm_decoder.norm.bias.data[:] = rm.tensor(torch_decoder.norm.bias.detach().numpy())

class TestTransformerEnDecoder(unittest.TestCase):
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
    
    def _run_encoder_test_case(self, case_name, **kwargs):
        """运行TransformerEncoder的单个测试用例"""
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 4)
        dim_feedforward = kwargs.get('dim_feedforward', 512)
        activation = kwargs.get('activation', 'relu')
        norm_first = kwargs.get('norm_first', False)
        batch_first = kwargs.get('batch_first', False)
        batch_size = kwargs.get('batch_size', 4)
        seq_len = kwargs.get('seq_len', 5)
        num_layers = kwargs.get('num_layers', 2)
        use_norm = kwargs.get('use_norm', False)
        use_mask = kwargs.get('use_mask', False)
        use_key_padding_mask = kwargs.get('use_key_padding_mask', False)
        is_causal = kwargs.get('is_causal', False)
        
        start_time = time.time()
        
        try:
            # 确定输入形状
            if batch_first:
                src_shape = (batch_size, seq_len, d_model)
            else:
                src_shape = (seq_len, batch_size, d_model)
            
            # 创建输入数据
            numpy_src = np.random.randn(*src_shape).astype(np.float32)
            
            # Riemann 张量
            rm_src = rm.tensor(numpy_src, requires_grad=True)
            
            # PyTorch 张量
            torch_src = torch.tensor(numpy_src, requires_grad=True)
            
            # 创建掩码
            rm_mask = None
            torch_mask = None
            if use_mask:
                mask_shape = (seq_len, seq_len)
                mask_data = np.random.choice([0.0, -1e4], size=mask_shape, p=[0.8, 0.2]).astype(np.float32)
                rm_mask = rm.tensor(mask_data)
                torch_mask = torch.tensor(mask_data)
            
            rm_key_padding_mask = None
            torch_key_padding_mask = None
            if use_key_padding_mask:
                mask_shape = (batch_size, seq_len)
                mask_data = np.random.choice([False, True], size=mask_shape, p=[0.8, 0.2])
                rm_key_padding_mask = rm.tensor(mask_data)
                torch_key_padding_mask = torch.tensor(mask_data)
            
            # 创建模块（注意dropout设置为0.0，避免随机性）
            rm_encoder_layer = rm_nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=0.0, activation=activation,
                norm_first=norm_first, batch_first=batch_first
            )
            
            torch_encoder_layer = torch_nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=0.0, activation=activation,
                norm_first=norm_first, batch_first=batch_first
            )
            
            # 最后的 norm
            rm_norm = None
            torch_norm = None
            if use_norm:
                rm_norm = rm_nn.LayerNorm(d_model)
                torch_norm = torch_nn.LayerNorm(d_model)
            
            # 创建完整的 Encoder
            rm_encoder = rm_nn.TransformerEncoder(
                rm_encoder_layer, num_layers=num_layers, norm=rm_norm
            )
            
            # 设置 enable_nested_tensor=False 消除 PyTorch 的警告
            torch_encoder = torch_nn.TransformerEncoder(
                torch_encoder_layer, num_layers=num_layers, norm=torch_norm,
                enable_nested_tensor=False
            )
            
            # 同步参数
            sync_encoder_layers(rm_encoder, torch_encoder, num_layers)
            
            # 前向传播
            rm_output = rm_encoder(
                rm_src, mask=rm_mask,
                src_key_padding_mask=rm_key_padding_mask,
                is_causal=is_causal
            )
            
            # PyTorch: 处理is_causal时需要明确的mask
            if is_causal and torch_mask is None:
                np_causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * -1e4
                torch_mask = torch.tensor(np_causal_mask)
                rm_mask_actual = rm.tensor(np_causal_mask)
                rm_output = rm_encoder(
                    rm_src, mask=rm_mask_actual,
                    src_key_padding_mask=rm_key_padding_mask,
                    is_causal=False
                )
            
            torch_output = torch_encoder(
                torch_src, mask=torch_mask,
                src_key_padding_mask=torch_key_padding_mask,
                is_causal=is_causal if torch_mask is not None else False
            )
            
            # 比较前向输出
            passed_output = compare_values(rm_output, torch_output)
            
            # 反向传播
            rm_loss = rm_output.sum()
            rm_loss.backward()
            
            torch_loss = torch_output.sum()
            torch_loss.backward()
            
            # 比较输入梯度
            passed_input_grad = compare_values(rm_src.grad, torch_src.grad)
            
            # 比较第一层的参数梯度
            passed_param_grad = True
            passed_param_grad = passed_param_grad and compare_values(
                rm_encoder.layers[0].self_attn.in_proj_weight.grad, 
                torch_encoder.layers[0].self_attn.in_proj_weight.grad
            )
            passed_param_grad = passed_param_grad and compare_values(
                rm_encoder.layers[0].self_attn.out_proj.weight.grad, 
                torch_encoder.layers[0].self_attn.out_proj.weight.grad
            )
            passed_param_grad = passed_param_grad and compare_values(
                rm_encoder.layers[0].linear1.weight.grad, 
                torch_encoder.layers[0].linear1.weight.grad
            )
            passed_param_grad = passed_param_grad and compare_values(
                rm_encoder.layers[0].linear2.weight.grad, 
                torch_encoder.layers[0].linear2.weight.grad
            )
            
            passed = passed_output and passed_input_grad and passed_param_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(f"{case_name} - 输出", passed_output)
                stats.add_result(f"{case_name} - 输入梯度", passed_input_grad)
                stats.add_result(f"{case_name} - 参数梯度", passed_param_grad)
                
                status = "通过" if passed else "失败"
                status_color = Colors.OKGREEN if passed else Colors.FAIL
                print(f"测试用例: {case_name} - {status_color}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if passed_output:
                    rm_out_np = rm_output.detach().numpy() if rm_output.requires_grad else rm_output.numpy()
                    torch_out_np = torch_output.detach().numpy() if torch_output.requires_grad else torch_output.numpy()
                    print(f"  输出最大差异: {np.max(np.abs(rm_out_np - torch_out_np)):.2e}")
            
            self.assertTrue(passed_output, f"输出不一致: {case_name}")
            self.assertTrue(passed_input_grad, f"输入梯度不一致: {case_name}")
            self.assertTrue(passed_param_grad, f"参数梯度不一致: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def _run_decoder_test_case(self, case_name, **kwargs):
        """运行TransformerDecoder的单个测试用例"""
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 4)
        dim_feedforward = kwargs.get('dim_feedforward', 512)
        activation = kwargs.get('activation', 'relu')
        norm_first = kwargs.get('norm_first', False)
        batch_first = kwargs.get('batch_first', False)
        batch_size = kwargs.get('batch_size', 4)
        tgt_len = kwargs.get('tgt_len', 5)
        src_len = kwargs.get('src_len', 7)
        num_layers = kwargs.get('num_layers', 2)
        use_norm = kwargs.get('use_norm', False)
        use_tgt_mask = kwargs.get('use_tgt_mask', False)
        use_tgt_key_padding_mask = kwargs.get('use_tgt_key_padding_mask', False)
        use_memory_key_padding_mask = kwargs.get('use_memory_key_padding_mask', False)
        tgt_is_causal = kwargs.get('tgt_is_causal', False)
        memory_is_causal = kwargs.get('memory_is_causal', False)
        
        start_time = time.time()
        
        try:
            # 确定输入形状
            if batch_first:
                tgt_shape = (batch_size, tgt_len, d_model)
                memory_shape = (batch_size, src_len, d_model)
            else:
                tgt_shape = (tgt_len, batch_size, d_model)
                memory_shape = (src_len, batch_size, d_model)
            
            # 创建输入数据
            numpy_tgt = np.random.randn(*tgt_shape).astype(np.float32)
            numpy_memory = np.random.randn(*memory_shape).astype(np.float32)
            
            # Riemann 张量
            rm_tgt = rm.tensor(numpy_tgt, requires_grad=True)
            rm_memory = rm.tensor(numpy_memory, requires_grad=True)
            
            # PyTorch 张量
            torch_tgt = torch.tensor(numpy_tgt, requires_grad=True)
            torch_memory = torch.tensor(numpy_memory, requires_grad=True)
            
            # 创建掩码
            rm_tgt_mask = None
            torch_tgt_mask = None
            if use_tgt_mask:
                mask_shape = (tgt_len, tgt_len)
                mask_data = np.random.choice([0.0, -1e4], size=mask_shape, p=[0.8, 0.2]).astype(np.float32)
                rm_tgt_mask = rm.tensor(mask_data)
                torch_tgt_mask = torch.tensor(mask_data)
            
            rm_tgt_key_padding_mask = None
            torch_tgt_key_padding_mask = None
            if use_tgt_key_padding_mask:
                mask_shape = (batch_size, tgt_len)
                mask_data = np.random.choice([False, True], size=mask_shape, p=[0.8, 0.2])
                rm_tgt_key_padding_mask = rm.tensor(mask_data)
                torch_tgt_key_padding_mask = torch.tensor(mask_data)
            
            rm_memory_key_padding_mask = None
            torch_memory_key_padding_mask = None
            if use_memory_key_padding_mask:
                mask_shape = (batch_size, src_len)
                mask_data = np.random.choice([False, True], size=mask_shape, p=[0.8, 0.2])
                rm_memory_key_padding_mask = rm.tensor(mask_data)
                torch_memory_key_padding_mask = torch.tensor(mask_data)
            
            # 创建模块（注意dropout设置为0.0，避免随机性）
            rm_decoder_layer = rm_nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=0.0, activation=activation,
                norm_first=norm_first, batch_first=batch_first
            )
            
            torch_decoder_layer = torch_nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=0.0, activation=activation,
                norm_first=norm_first, batch_first=batch_first
            )
            
            # 最后的 norm
            rm_norm = None
            torch_norm = None
            if use_norm:
                rm_norm = rm_nn.LayerNorm(d_model)
                torch_norm = torch_nn.LayerNorm(d_model)
            
            # 创建完整的 Decoder
            rm_decoder = rm_nn.TransformerDecoder(
                rm_decoder_layer, num_layers=num_layers, norm=rm_norm
            )
            
            torch_decoder = torch_nn.TransformerDecoder(
                torch_decoder_layer, num_layers=num_layers, norm=torch_norm
            )
            
            # 同步参数
            sync_decoder_layers(rm_decoder, torch_decoder, num_layers)
            
            # 前向传播
            rm_output = rm_decoder(
                rm_tgt, rm_memory,
                tgt_mask=rm_tgt_mask,
                tgt_key_padding_mask=rm_tgt_key_padding_mask,
                memory_key_padding_mask=rm_memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal
            )
            
            # PyTorch: 处理tgt_is_causal时需要明确的mask
            if tgt_is_causal and torch_tgt_mask is None:
                np_causal_mask = np.triu(np.ones((tgt_len, tgt_len), dtype=np.float32), k=1) * -1e4
                torch_tgt_mask = torch.tensor(np_causal_mask)
                rm_tgt_mask_actual = rm.tensor(np_causal_mask)
                rm_output = rm_decoder(
                    rm_tgt, rm_memory,
                    tgt_mask=rm_tgt_mask_actual,
                    tgt_key_padding_mask=rm_tgt_key_padding_mask,
                    memory_key_padding_mask=rm_memory_key_padding_mask,
                    tgt_is_causal=False,
                    memory_is_causal=False
                )
            
            torch_output = torch_decoder(
                torch_tgt, torch_memory,
                tgt_mask=torch_tgt_mask,
                tgt_key_padding_mask=torch_tgt_key_padding_mask,
                memory_key_padding_mask=torch_memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal if torch_tgt_mask is not None else False,
                memory_is_causal=memory_is_causal
            )
            
            # 比较前向输出
            passed_output = compare_values(rm_output, torch_output)
            
            # 反向传播
            rm_loss = rm_output.sum()
            rm_loss.backward()
            
            torch_loss = torch_output.sum()
            torch_loss.backward()
            
            # 比较输入梯度
            passed_tgt_grad = compare_values(rm_tgt.grad, torch_tgt.grad)
            passed_memory_grad = compare_values(rm_memory.grad, torch_memory.grad)
            
            # 比较第一层的参数梯度
            passed_param_grad = True
            passed_param_grad = passed_param_grad and compare_values(
                rm_decoder.layers[0].self_attn.in_proj_weight.grad, 
                torch_decoder.layers[0].self_attn.in_proj_weight.grad
            )
            passed_param_grad = passed_param_grad and compare_values(
                rm_decoder.layers[0].multihead_attn.in_proj_weight.grad, 
                torch_decoder.layers[0].multihead_attn.in_proj_weight.grad
            )
            passed_param_grad = passed_param_grad and compare_values(
                rm_decoder.layers[0].linear1.weight.grad, 
                torch_decoder.layers[0].linear1.weight.grad
            )
            passed_param_grad = passed_param_grad and compare_values(
                rm_decoder.layers[0].linear2.weight.grad, 
                torch_decoder.layers[0].linear2.weight.grad
            )
            
            passed = passed_output and passed_tgt_grad and passed_memory_grad and passed_param_grad
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(f"{case_name} - 输出", passed_output)
                stats.add_result(f"{case_name} - tgt梯度", passed_tgt_grad)
                stats.add_result(f"{case_name} - memory梯度", passed_memory_grad)
                stats.add_result(f"{case_name} - 参数梯度", passed_param_grad)
                
                status = "通过" if passed else "失败"
                status_color = Colors.OKGREEN if passed else Colors.FAIL
                print(f"测试用例: {case_name} - {status_color}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if passed_output:
                    rm_out_np = rm_output.detach().numpy() if rm_output.requires_grad else rm_output.numpy()
                    torch_out_np = torch_output.detach().numpy() if torch_output.requires_grad else torch_output.numpy()
                    print(f"  输出最大差异: {np.max(np.abs(rm_out_np - torch_out_np)):.2e}")
            
            self.assertTrue(passed_output, f"输出不一致: {case_name}")
            self.assertTrue(passed_tgt_grad, f"tgt梯度不一致: {case_name}")
            self.assertTrue(passed_memory_grad, f"memory梯度不一致: {case_name}")
            self.assertTrue(passed_param_grad, f"参数梯度不一致: {case_name}")
            
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    # ==============================
    # TransformerEncoder 测试
    # ==============================
    def test_encoder_basic(self):
        """Encoder - 基础配置"""
        self._run_encoder_test_case("基础配置")
    
    def test_encoder_with_norm(self):
        """Encoder - 带最终norm"""
        self._run_encoder_test_case("带最终norm", use_norm=True)
    
    def test_encoder_norm_first(self):
        """Encoder - norm_first=True"""
        self._run_encoder_test_case("norm_first=True", norm_first=True)
    
    def test_encoder_batch_first(self):
        """Encoder - batch_first=True"""
        self._run_encoder_test_case("batch_first=True", batch_first=True)
    
    def test_encoder_gelu(self):
        """Encoder - activation='gelu'"""
        self._run_encoder_test_case("activation=gelu", activation='gelu')
    
    def test_encoder_large_dim(self):
        """Encoder - 大维度"""
        self._run_encoder_test_case("大维度", d_model=256, nhead=8, dim_feedforward=1024, num_layers=3)
    
    def test_encoder_with_mask(self):
        """Encoder - 带mask"""
        self._run_encoder_test_case("带mask", use_mask=True)
    
    def test_encoder_with_key_padding_mask(self):
        """Encoder - 带key_padding_mask"""
        self._run_encoder_test_case("带key_padding_mask", use_key_padding_mask=True)
    
    def test_encoder_is_causal(self):
        """Encoder - is_causal=True"""
        self._run_encoder_test_case("is_causal=True", is_causal=True)
    
    # ==============================
    # TransformerDecoder 测试
    # ==============================
    def test_decoder_basic(self):
        """Decoder - 基础配置"""
        self._run_decoder_test_case("基础配置")
    
    def test_decoder_with_norm(self):
        """Decoder - 带最终norm"""
        self._run_decoder_test_case("带最终norm", use_norm=True)
    
    def test_decoder_norm_first(self):
        """Decoder - norm_first=True"""
        self._run_decoder_test_case("norm_first=True", norm_first=True)
    
    def test_decoder_batch_first(self):
        """Decoder - batch_first=True"""
        self._run_decoder_test_case("batch_first=True", batch_first=True)
    
    def test_decoder_gelu(self):
        """Decoder - activation='gelu'"""
        self._run_decoder_test_case("activation=gelu", activation='gelu')
    
    def test_decoder_large_dim(self):
        """Decoder - 大维度"""
        self._run_decoder_test_case("大维度", d_model=256, nhead=8, dim_feedforward=1024, num_layers=3)
    
    def test_decoder_with_tgt_mask(self):
        """Decoder - 带tgt_mask"""
        self._run_decoder_test_case("带tgt_mask", use_tgt_mask=True)
    
    def test_decoder_with_tgt_key_padding_mask(self):
        """Decoder - 带tgt_key_padding_mask"""
        self._run_decoder_test_case("带tgt_key_padding_mask", use_tgt_key_padding_mask=True)
    
    def test_decoder_with_memory_key_padding_mask(self):
        """Decoder - 带memory_key_padding_mask"""
        self._run_decoder_test_case("带memory_key_padding_mask", use_memory_key_padding_mask=True)
    
    def test_decoder_is_causal(self):
        """Decoder - tgt_is_causal=True"""
        self._run_decoder_test_case("tgt_is_causal=True", tgt_is_causal=True)

if __name__ == '__main__':
    IS_RUNNING_AS_SCRIPT = True
    clear_screen()
    print("=" * 70)
    print("Riemann 与 PyTorch TransformerEncoder/Decoder一致性测试")
    print("=" * 70)
    
    unittest.main(exit=False, verbosity=0)
    
    stats.print_summary()
