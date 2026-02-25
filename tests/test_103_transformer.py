import unittest
import numpy as np
import time
import sys, os
import warnings

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
    print("警告: 无法导入PyTorch，将只测试riemann的Transformer")
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
    
    return np.allclose(rm_data, torch_data, atol=atol, rtol=rtol)

def sync_transformer(rm_transformer, torch_transformer, num_encoder_layers, num_decoder_layers):
    """同步Transformer的所有层参数"""
    
    # 同步Encoder
    rm_encoder = rm_transformer.encoder
    torch_encoder = torch_transformer.encoder
    
    for i in range(num_encoder_layers):
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
    
    # Encoder最后的norm
    if rm_encoder.norm is not None and torch_encoder.norm is not None:
        rm_encoder.norm.weight.data[:] = rm.tensor(torch_encoder.norm.weight.detach().numpy())
        rm_encoder.norm.bias.data[:] = rm.tensor(torch_encoder.norm.bias.detach().numpy())
    
    # 同步Decoder
    rm_decoder = rm_transformer.decoder
    torch_decoder = torch_transformer.decoder
    
    for i in range(num_decoder_layers):
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
    
    # Decoder最后的norm
    if rm_decoder.norm is not None and torch_decoder.norm is not None:
        rm_decoder.norm.weight.data[:] = rm.tensor(torch_decoder.norm.weight.detach().numpy())
        rm_decoder.norm.bias.data[:] = rm.tensor(torch_decoder.norm.bias.detach().numpy())

class TestTransformer(unittest.TestCase):
    def setUp(self):
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
        """运行Transformer的单个测试用例"""
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 2)
        num_encoder_layers = kwargs.get('num_encoder_layers', 2)
        num_decoder_layers = kwargs.get('num_decoder_layers', 2)
        dim_feedforward = kwargs.get('dim_feedforward', 512)
        activation = kwargs.get('activation', 'relu')
        norm_first = kwargs.get('norm_first', False)
        batch_first = kwargs.get('batch_first', False)
        batch_size = kwargs.get('batch_size', 2)
        src_len = kwargs.get('src_len', 10)
        tgt_len = kwargs.get('tgt_len', 8)
        use_tgt_mask = kwargs.get('use_tgt_mask', False)
        
        start_time = time.time()
        
        try:
            if batch_first:
                src_np = np.random.randn(batch_size, src_len, d_model).astype(np.float32)
                tgt_np = np.random.randn(batch_size, tgt_len, d_model).astype(np.float32)
            else:
                src_np = np.random.randn(src_len, batch_size, d_model).astype(np.float32)
                tgt_np = np.random.randn(tgt_len, batch_size, d_model).astype(np.float32)
            
            rm_src = rm.tensor(src_np)
            rm_tgt = rm.tensor(tgt_np)
            rm_src.requires_grad = True
            rm_tgt.requires_grad = True
            
            rm_transformer = rm_nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                activation=activation,
                layer_norm_eps=1e-05,
                batch_first=batch_first,
                norm_first=norm_first
            )
            
            rm_tgt_mask = None
            if use_tgt_mask:
                tgt_mask_np = np.where(
                    np.triu(np.ones((tgt_len, tgt_len), dtype=np.float32), k=1) == 1, -np.inf, 0.0
                ).astype(np.float32)
                rm_tgt_mask = rm.tensor(tgt_mask_np)
            
            if TORCH_AVAILABLE:
                torch_src = torch.tensor(src_np, requires_grad=True)
                torch_tgt = torch.tensor(tgt_np, requires_grad=True)
                
                # 手动构造 Transformer 的组件，避免警告
                # 1. 构造编码器层
                encoder_layer = torch_nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=batch_first,
                    norm_first=norm_first
                )
                
                # 2. 构造编码器，设置 enable_nested_tensor=False
                encoder_norm = torch_nn.LayerNorm(d_model, eps=1e-05)
                encoder = torch_nn.TransformerEncoder(
                    encoder_layer,
                    num_encoder_layers,
                    encoder_norm,
                    enable_nested_tensor=False
                )
                
                # 3. 构造解码器层
                decoder_layer = torch_nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=batch_first,
                    norm_first=norm_first
                )
                
                # 4. 构造解码器
                decoder_norm = torch_nn.LayerNorm(d_model, eps=1e-05)
                decoder = torch_nn.TransformerDecoder(
                    decoder_layer,
                    num_decoder_layers,
                    decoder_norm
                )
                
                # 5. 构造最终的 Transformer，使用自定义 encoder/decoder
                torch_transformer = torch_nn.Transformer(
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=batch_first,
                    norm_first=norm_first,
                    custom_encoder=encoder,
                    custom_decoder=decoder
                )
                
                sync_transformer(rm_transformer, torch_transformer, num_encoder_layers, num_decoder_layers)
                
                rm_transformer.eval()
                torch_transformer.eval()
                
                with torch.no_grad():
                    torch_tgt_mask = None
                    if use_tgt_mask:
                        torch_tgt_mask = torch.tensor(tgt_mask_np)
                    
                    torch_output = torch_transformer(
                        torch_src, torch_tgt,
                        tgt_mask=torch_tgt_mask
                    )
                
                rm_output = rm_transformer(
                    rm_src, rm_tgt,
                    tgt_mask=rm_tgt_mask
                )
                
                passed_output = compare_values(rm_output, torch_output)
                
                rm_src.grad = None
                rm_tgt.grad = None
                torch_src.grad = None
                torch_tgt.grad = None
                
                rm_transformer.train()
                torch_transformer.train()
                
                rm_output_train = rm_transformer(
                    rm_src, rm_tgt,
                    tgt_mask=rm_tgt_mask
                )
                
                torch_output_train = torch_transformer(
                    torch_src, torch_tgt,
                    tgt_mask=torch_tgt_mask
                )
                
                rm_loss = rm_output_train.sum()
                rm_loss.backward()
                
                torch_loss = torch_output_train.sum()
                torch_loss.backward()
                
                passed_input_grad_src = compare_values(rm_src.grad, torch_src.grad)
                passed_input_grad_tgt = compare_values(rm_tgt.grad, torch_tgt.grad)
                
                passed = passed_output and passed_input_grad_src and passed_input_grad_tgt
            else:
                rm_output = rm_transformer(
                    rm_src, rm_tgt,
                    tgt_mask=rm_tgt_mask
                )
                
                rm_loss = rm_output.sum()
                rm_loss.backward()
                
                passed_output = rm_output is not None
                passed_input_grad_src = rm_src.grad is not None
                passed_input_grad_tgt = rm_tgt.grad is not None
                
                passed = passed_output and passed_input_grad_src and passed_input_grad_tgt
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(f"{case_name}", passed)
                
                status = "通过" if passed else "失败"
                status_color = Colors.OKGREEN if passed else Colors.FAIL
                print(f"测试用例: {case_name} - {status_color}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                
                if TORCH_AVAILABLE and passed_output:
                    rm_out_np = rm_output.detach().numpy() if rm_output.requires_grad else rm_output.numpy()
                    torch_out_np = torch_output.detach().numpy() if torch_output.requires_grad else torch_output.numpy()
                    print(f"  输出最大差异: {np.max(np.abs(rm_out_np - torch_out_np)):.2e}")
            
            self.assertTrue(passed, f"测试失败: {case_name}")
            
        except Exception as e:
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(f"{case_name}", False)
                print(f"{Colors.FAIL}测试用例: {case_name} - 异常: {e}{Colors.ENDC}")
            raise
    
    def test_basic(self):
        self._run_test_case(
            "基本配置",
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            src_len=10,
            tgt_len=8,
            batch_size=2
        )
    
    def test_batch_first(self):
        self._run_test_case(
            "batch_first=True",
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            batch_first=True,
            src_len=10,
            tgt_len=8,
            batch_size=4
        )
    
    def test_norm_first(self):
        self._run_test_case(
            "norm_first=True",
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            norm_first=True,
            src_len=8,
            tgt_len=6,
            batch_size=2
        )
    
    def test_gelu_activation(self):
        self._run_test_case(
            "GELU激活",
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            activation="gelu",
            src_len=12,
            tgt_len=10,
            batch_size=2
        )
    
    def test_with_tgt_mask(self):
        self._run_test_case(
            "带tgt_mask",
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            use_tgt_mask=True,
            src_len=10,
            tgt_len=8,
            batch_size=2
        )
    
    def test_small_model(self):
        self._run_test_case(
            "小模型",
            d_model=64,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=128,
            src_len=5,
            tgt_len=5,
            batch_size=1
        )

def test_generate_square_subsequent_mask():
    """
    测试generate_square_subsequent_mask静态方法
    """
    print("\n" + "="*80)
    print(f"{Colors.HEADER}{Colors.BOLD}测试generate_square_subsequent_mask方法{Colors.ENDC}")
    print("="*80)
    stats.start_function("generate_square_subsequent_mask测试")
    
    sz_list = [3, 5, 10]
    
    for sz in sz_list:
        print(f"\n  测试 sz={sz}")
        
        rm_mask = rm_nn.Transformer.generate_square_subsequent_mask(sz)
        
        passed = rm_mask is not None and rm_mask.shape == (sz, sz)
        
        if TORCH_AVAILABLE:
            torch_mask = torch_nn.Transformer.generate_square_subsequent_mask(sz)
            passed = passed and compare_values(rm_mask, torch_mask, atol=1e-10, rtol=1e-10)
        
        stats.add_result(f"sz={sz}", passed)
        print(f"    结果: {Colors.OKGREEN if passed else Colors.FAIL}{'通过' if passed else '失败'}{Colors.ENDC}")
    
    stats.end_function()

# 主测试函数
def main():
    global IS_RUNNING_AS_SCRIPT
    IS_RUNNING_AS_SCRIPT = True
    
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("="*80)
    print("Riemann Transformer 测试套件")
    print("="*80)
    print(f"{Colors.ENDC}")
    
    if TORCH_AVAILABLE:
        print(f"{Colors.OKBLUE}PyTorch已加载，将进行数值一致性对比{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}PyTorch未加载，仅测试Riemann的基本功能{Colors.ENDC}")
    
    # 运行所有unittest测试
    test_cases = [
        ("test_basic", TestTransformer("test_basic")),
        ("test_batch_first", TestTransformer("test_batch_first")),
        ("test_norm_first", TestTransformer("test_norm_first")),
        ("test_gelu_activation", TestTransformer("test_gelu_activation")),
        ("test_with_tgt_mask", TestTransformer("test_with_tgt_mask")),
        ("test_small_model", TestTransformer("test_small_model")),
    ]
    
    for name, test in test_cases:
        stats.start_function(name)
        try:
            test.setUp()
            test._testMethodName = name
            test._run_test_case(
                name,
                **{
                    "test_basic": {"d_model":128,"nhead":2,"num_encoder_layers":2,"num_decoder_layers":2,"dim_feedforward":256},
                    "test_batch_first": {"d_model":128,"nhead":2,"num_encoder_layers":2,"num_decoder_layers":2,"dim_feedforward":256,"batch_first":True},
                    "test_norm_first": {"d_model":128,"nhead":2,"num_encoder_layers":2,"num_decoder_layers":2,"dim_feedforward":256,"norm_first":True},
                    "test_gelu_activation": {"d_model":128,"nhead":2,"num_encoder_layers":2,"num_decoder_layers":2,"dim_feedforward":256,"activation":"gelu"},
                    "test_with_tgt_mask": {"d_model":128,"nhead":2,"num_encoder_layers":2,"num_decoder_layers":2,"dim_feedforward":256,"use_tgt_mask":True},
                    "test_small_model": {"d_model":64,"nhead":2,"num_encoder_layers":1,"num_decoder_layers":1,"dim_feedforward":128},
                }[name]
            )
        except Exception as e:
            print(f"{Colors.FAIL}测试 {name} 异常: {e}{Colors.ENDC}")
        finally:
            test.tearDown()
    
    # 测试generate_square_subsequent_mask
    test_generate_square_subsequent_mask()
    
    # 打印统计摘要
    stats.print_summary()
    
    # 返回退出码
    return 0 if stats.passed_cases == stats.total_cases else 1

# 当作为脚本运行时执行main
if __name__ == "__main__":
    sys.exit(main())
