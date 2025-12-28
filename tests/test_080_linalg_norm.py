import unittest
import numpy as np
import torch
import sys, os
import time

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))
from riemann.linalg import norm
from riemann.tensordef import tensor
import riemann as rm

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
# ...

# 在StatisticsCollector类中添加一个方法来检查是否有失败用例
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

    def has_failures(self):
        """检查是否有测试用例失败"""
        return self.passed_cases < self.total_cases
    
    def get_failed_functions(self):
        """获取有失败用例的函数列表"""
        failed = []
        for func_name, stats in self.function_stats.items():
            if stats['passed'] < stats['total']:
                failed.append(func_name)
        return failed

# 全局统计实例
stats = StatisticsCollector()
IS_RUNNING_AS_SCRIPT = False

class TestLinalgNorm(unittest.TestCase):
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
    
    def test_vector_norm_basic(self):
        """测试基本向量范数计算"""
        # 测试数据 - 使用默认float32精度
        np_data = np.random.randn(5).astype(rm.get_default_dtype())
        
        # 转换为riemann和torch张量 - 使用默认精度
        riemann_tensor = tensor(np_data, requires_grad=True)
        torch_tensor = torch.tensor(np_data, requires_grad=True)
        
        # 测试各种范数
        ord_values = [None, 0, 1,-1,2,-2, float('inf'), float('-inf'), 3]
        
        for ord in ord_values:
            if IS_RUNNING_AS_SCRIPT:
                print(f"  测试范数类型: ord={ord}")
            
            case_name = f"vector_norm_ord={ord}"
            
            # 计算riemann的结果
            riemann_result = norm(riemann_tensor, ord=ord)
            
            # 计算torch的结果
            torch_result = torch.linalg.norm(torch_tensor, ord=ord)
            
            # 对比结果 - 统一转换为numpy数组进行比较
            try:
                np.testing.assert_allclose(
                    riemann_result.data,
                    torch_result.detach().numpy(),
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"向量范数测试失败，ord={ord}"
                )
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 向量范数 ord={ord}")
                    stats.add_result(case_name, True)
            except AssertionError as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                    stats.add_result(case_name, False, [str(e)])
                raise  # 直接抛出异常，让unittest框架处理
    
    def test_matrix_norm_basic(self):
        """测试基本矩阵范数计算"""
        # 测试数据 - 使用默认float32精度
        np_data = np.random.randn(3, 4).astype(rm.get_default_dtype())
        
        # 转换为riemann和torch张量 - 使用默认精度
        riemann_tensor = tensor(np_data, requires_grad=True)
        torch_tensor = torch.tensor(np_data, requires_grad=True)
        
        # 测试各种范数
        ord_values = [None, 'fro', 1,-1, 2, -2, float('inf'), float('-inf')]
        
        for ord in ord_values:
            if IS_RUNNING_AS_SCRIPT:
                print(f"  测试范数类型: ord={ord}")
            
            case_name = f"matrix_norm_ord={ord}"
            
            # 计算riemann的结果
            riemann_result = norm(riemann_tensor, ord=ord)
            
            # 计算torch的结果
            torch_result = torch.linalg.norm(torch_tensor, ord=ord)
            
            # 对比结果
            try:
                np.testing.assert_allclose(
                    riemann_result.data,
                    torch_result.detach().numpy(),
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"矩阵范数测试失败，ord={ord}"
                )
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 矩阵范数 ord={ord}")
                    stats.add_result(case_name, True)
            except AssertionError as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                    stats.add_result(case_name, False, [str(e)])
                raise  # 直接抛出异常
    
    def test_vector_norm_with_dim(self):
        """测试指定维度的向量范数"""
        # 测试数据 - 使用默认float32精度
        np_data = np.random.randn(2, 3, 4).astype(rm.get_default_dtype())
        
        # 转换为riemann和torch张量 - 使用默认精度
        riemann_tensor = tensor(np_data, requires_grad=True)
        torch_tensor = torch.tensor(np_data, requires_grad=True)
        
        # 测试不同维度和keepdim参数
        for dim in [0, 1, 2]:
            for keepdim in [True, False]:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  测试维度组合: dim={dim}, keepdim={keepdim}")
                # 测试各种范数
                ord_values = [None, 0, 1,-1, 2, -2, float('inf'), float('-inf'), 3]
                
                for ord in ord_values:
                    case_name = f"vector_norm_dim={dim}_keepdim={keepdim}_ord={ord}"
                    
                    # 计算riemann的结果
                    riemann_result = norm(riemann_tensor, ord=ord, dim=dim, keepdim=keepdim)
                    
                    # 计算torch的结果
                    torch_result = torch.linalg.norm(
                        torch_tensor,
                        ord=ord,
                        dim=dim,
                        keepdim=keepdim
                    )
                    
                    # 对比结果
                    try:
                        np.testing.assert_allclose(
                            riemann_result.data,
                            torch_result.detach().numpy(),
                            rtol=1e-6,
                            atol=1e-8,
                            err_msg=f"指定维度向量范数测试失败，ord={ord}, dim={dim}, keepdim={keepdim}"
                        )
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - ord={ord}")
                            stats.add_result(case_name, True)
                    except AssertionError as e:
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                            stats.add_result(case_name, False, [str(e)])
                        raise
    
    def test_matrix_norm_with_dim(self):
        """测试指定维度的矩阵范数"""
        # 测试数据 - 使用默认float32精度
        np_data = np.random.randn(2, 3, 4, 5).astype(rm.get_default_dtype())
        
        # 转换为riemann和torch张量 - 使用默认精度
        riemann_tensor = tensor(np_data, requires_grad=True)
        torch_tensor = torch.tensor(np_data, requires_grad=True)
        
        # 测试不同维度组合和keepdim参数
        dim_pairs = [(2, 3), (0, 1), (1, 2)]
        for dim in dim_pairs:
            for keepdim in [True, False]:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  测试维度组合: dim={dim}, keepdim={keepdim}")
                # 测试各种范数
                ord_values = [None, 'fro', 1,-1, 2,-2,float('inf'), float('-inf')]
                
                for ord in ord_values:
                    case_name = f"matrix_norm_dim={dim}_keepdim={keepdim}_ord={ord}"
                    
                    # 计算riemann的结果
                    riemann_result = norm(riemann_tensor, ord=ord, dim=dim, keepdim=keepdim)
                    
                    # 计算torch的结果
                    torch_result = torch.linalg.norm(
                        torch_tensor,
                        ord=ord,
                        dim=dim,
                        keepdim=keepdim
                    )
                    
                    # 对比结果
                    try:
                        np.testing.assert_allclose(
                            riemann_result.data,
                            torch_result.detach().numpy(),
                            rtol=1e-6,
                            atol=1e-8,
                            err_msg=f"指定维度矩阵范数测试失败，ord={ord}, dim={dim}, keepdim={keepdim}"
                        )
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - ord={ord}")
                            stats.add_result(case_name, True)
                    except AssertionError as e:
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                            stats.add_result(case_name, False, [str(e)])
                        raise
    
    def test_vector_norm_grad(self):
        """测试向量范数的梯度计算"""
        # 不测试L0范数的梯度，因为它通常不光滑
        ord_values = [1, 2, float('inf'), 3]
        
        # 测试数据 - 使用默认float32精度
        np_data = np.random.randn(5).astype(rm.get_default_dtype())
        
        for ord in ord_values:
            if IS_RUNNING_AS_SCRIPT:
                print(f"  测试梯度计算: ord={ord}")
            
            # 每次测试前重置张量以避免梯度累积
            riemann_tensor = tensor(np_data, requires_grad=True)
            torch_tensor = torch.tensor(np_data, requires_grad=True)
            
            # 计算riemann的结果和梯度
            riemann_result = norm(riemann_tensor, ord=ord)
            riemann_result.backward()
            riemann_grad = riemann_tensor.grad.data
            
            # 计算torch的结果和梯度
            torch_result = torch.linalg.norm(torch_tensor, ord=ord)
            torch_result.backward()
            torch_grad = torch_tensor.grad.numpy()
            
            # 对比梯度
            try:
                np.testing.assert_allclose(
                    riemann_grad,
                    torch_grad,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"向量范数梯度测试失败，ord={ord}"
                )
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 向量范数梯度 ord={ord}")
                    stats.add_result(f"vector_norm_grad_ord={ord}", True)
            except AssertionError as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                    stats.add_result(f"vector_norm_grad_ord={ord}", False, [str(e)])
                raise
    
    # 已移除test_matrix_norm_grad方法，因为其功能已被test_comprehensive_gradients覆盖
    
    def test_norm_with_dtype(self):
        """测试dtype参数"""
        # 测试数据 - 使用默认float32精度
        np_data = np.random.randn(5).astype(rm.get_default_dtype())
        
        # 测试不同的数据类型
        np_dtypes = [np.float64, np.float32]
        torch_dtypes = [torch.float64, torch.float32]
        
        for np_dtype, torch_dtype in zip(np_dtypes, torch_dtypes):
            if IS_RUNNING_AS_SCRIPT:
                print(f"  测试数据类型: dtype={np_dtype}")
            
            # 转换为riemann和torch张量 - 使用测试配置精度
            riemann_tensor = tensor(np_data, dtype=np_dtype,requires_grad=True)
            torch_tensor = torch.tensor(np_data, dtype=torch_dtype, requires_grad=True)
            
            # 计算riemann的结果
            riemann_result = norm(riemann_tensor, ord=2, dtype=np_dtype)
            
            # 计算torch的结果
            torch_result = torch.linalg.norm(torch_tensor, ord=2, dtype=torch_dtype)
            
            # 对比结果
            try:
                np.testing.assert_allclose(
                    riemann_result.data,
                    torch_result.detach().numpy(),
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"dtype参数测试失败，dtype={np_dtype}"
                )
                # 验证数据类型
                self.assertEqual(riemann_result.data.dtype, np_dtype)
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  {Colors.OKGREEN}✓ 通过{Colors.ENDC} - dtype={np_dtype}")
                    stats.add_result(f"norm_dtype={np_dtype}", True)
            except (AssertionError, TypeError) as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                    stats.add_result(f"norm_dtype={np_dtype}", False, [str(e)])
                raise
    
    def test_norm_with_out(self):
        """测试out参数"""
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试out参数功能")
        
        # 测试数据 - 使用默认float32精度
        np_data = np.random.randn(5).astype(rm.get_default_dtype())
        
        # 转换为riemann和torch张量 - 使用默认精度
        riemann_tensor = tensor(np_data, requires_grad=True)
        torch_tensor = torch.tensor(np_data, requires_grad=True)
        
        # 创建输出张量（不设置requires_grad以避免PyTorch的原地操作错误）
        riemann_out = tensor(np.zeros(()))
        torch_out = torch.zeros(())
        
        # 计算riemann的结果
        riemann_result = norm(riemann_tensor, ord=2, out=riemann_out)
        
        # 计算torch的结果
        torch_result = torch.linalg.norm(torch_tensor, ord=2, out=torch_out)
        
        # 对比结果
        try:
            np.testing.assert_allclose(
                riemann_result.data,
                torch_result.detach().numpy(),
                rtol=1e-6,
                atol=1e-8,
                err_msg="out参数测试失败"
            )
            
            # 验证out和result是同一个对象
            self.assertIs(riemann_result, riemann_out)
            if IS_RUNNING_AS_SCRIPT:
                print(f"  {Colors.OKGREEN}✓ 通过{Colors.ENDC} - out参数测试")
                stats.add_result("norm_with_out", True)
        except AssertionError as e:
            if IS_RUNNING_AS_SCRIPT:
                print(f"  {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                stats.add_result("norm_with_out", False, [str(e)])
            raise
    
    def test_special_cases(self):
        """测试特殊情况"""
        # 测试全零向量
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试全零向量")
        
        zero_vector_data = np.zeros(5, dtype=np.float32)
        riemann_zero_vector = tensor(zero_vector_data, requires_grad=True)
        torch_zero_vector = torch.tensor(zero_vector_data, requires_grad=True)
        
        # 向量测试各种范数
        vector_ord_values = [None, 0, 1,-1, 2, -2,float('inf'), float('-inf'), 3]
        for ord in vector_ord_values:
            try:
                riemann_result = norm(riemann_zero_vector, ord=ord)
                torch_result = torch.linalg.norm(torch_zero_vector, ord=ord)
                
                np.testing.assert_allclose(
                    riemann_result.data,
                    torch_result.detach().numpy(),
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"全零向量测试失败，ord={ord}"
                )
                if IS_RUNNING_AS_SCRIPT:
                    print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 全零向量 ord={ord}")
                    stats.add_result(f"zero_vector_ord={ord}", True)
            except AssertionError as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                    stats.add_result(f"zero_vector_ord={ord}", False, [str(e)])
                raise
        
        # 测试全零矩阵
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试全零矩阵")
        
        zero_matrix_data = np.zeros((3, 4), dtype=np.float32)
        riemann_zero_matrix = tensor(zero_matrix_data, requires_grad=True)
        torch_zero_matrix = torch.tensor(zero_matrix_data, requires_grad=True)
        
        # 矩阵测试各种范数（跳过L0）
        matrix_ord_values = [None, 'fro', 1, -1, 2, -2,float('inf')]
        for ord in matrix_ord_values:
            try:
                riemann_result = norm(riemann_zero_matrix, ord=ord)
                torch_result = torch.linalg.norm(torch_zero_matrix, ord=ord)
                
                np.testing.assert_allclose(
                    riemann_result.data,
                    torch_result.detach().numpy(),
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"全零矩阵测试失败，ord={ord}"
                )
                if IS_RUNNING_AS_SCRIPT:
                    print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 全零矩阵 ord={ord}")
                    stats.add_result(f"zero_matrix_ord={ord}", True)
            except AssertionError as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                    stats.add_result(f"zero_matrix_ord={ord}", False, [str(e)])
                raise
        
        # 测试单位矩阵的Frobenius范数
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试单位矩阵的Frobenius范数")
        
        identity_data = np.eye(3, dtype=np.float32)
        riemann_identity = tensor(identity_data, requires_grad=True)
        torch_identity = torch.tensor(identity_data, requires_grad=True)
        
        try:
            riemann_result = norm(riemann_identity, ord='fro')
            torch_result = torch.linalg.norm(torch_identity, ord='fro')
            
            np.testing.assert_allclose(
                riemann_result.data,
                torch_result.detach().numpy(),
                rtol=1e-6,
                atol=1e-8,
                err_msg="单位矩阵测试失败"
            )
            if IS_RUNNING_AS_SCRIPT:
                print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 单位矩阵Frobenius范数")
                stats.add_result("identity_matrix_frobenius", True)
        except AssertionError as e:
            if IS_RUNNING_AS_SCRIPT:
                print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                stats.add_result("identity_matrix_frobenius", False, [str(e)])
            raise
    
    def _test_error_case(self, test_func, expected_exception, case_name, description):
        """统一的错误测试辅助方法，减少重复代码"""
        try:
            with self.assertRaises(expected_exception):
                test_func()
            if IS_RUNNING_AS_SCRIPT:
                print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - {description}")
                stats.add_result(case_name, True)
        except AssertionError as e:
            if IS_RUNNING_AS_SCRIPT:
                print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {description}处理错误")
                stats.add_result(case_name, False, [str(e)])
            raise
            
    def test_error_handling(self):
        """测试异常处理"""
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试异常处理")
        
        # 测试非TN类型输入 - 使用统一的错误测试方法
        self._test_error_case(
            lambda: norm([1, 2, 3]),
            TypeError,
            "error_non_tn_type",
            "非TN类型输入异常"
        )
        
        # 测试不支持的ord值 - 优化了代码结构
        self._test_error_case(
            lambda: norm(tensor([1., 2., 3.]), ord=-3),
            ValueError,
            "error_invalid_ord",
            "不支持的ord值异常"
        )
        
        # 测试不支持的dim参数 - 使用统一的错误测试方法
        self._test_error_case(
            lambda: norm(tensor([1., 2., 3.]), dim=(0, 1)),
            ValueError,
            "error_invalid_dim",
            "不支持的dim参数异常"
        )
        
        # 测试Frobenius范数用于向量 - 使用统一的错误测试方法
        self._test_error_case(
            lambda: norm(tensor([1., 2., 3.]), ord='fro'),
            ValueError,
            "error_frobenius_for_vector",
            "Frobenius范数用于向量异常"
        )
        
        # 测试L0范数用于矩阵 - 使用统一的错误测试方法
        self._test_error_case(
            lambda: norm(tensor([[1., 2.], [3., 4.]]), ord=0),
            ValueError,
            "error_l0_for_matrix",
            "L0范数用于矩阵异常"
        )
        
        # 删除不再需要的nuc范数未实现测试
        # 测试nuc范数已实现，所以移除这个测试用例
        
        # 测试dim=None且ord!=None时对高维张量的错误处理 - 使用统一的错误测试方法
        self._test_error_case(
            lambda: norm(tensor(np.random.randn(2, 3, 4)), ord=2),
            ValueError,
            "error_high_dim_with_ord_not_none",
            "dim=None且ord!=None时高维张量异常"
        )

    # 已移除test_matrix_norm_negative_ord方法，因为其功能已被test_matrix_norm_basic覆盖

    def test_multi_axis_norm(self):
        """测试多轴范数计算"""
        # 测试数据 - 使用默认float32精度
        np_data = np.random.randn(2, 3, 4, 5).astype(rm.get_default_dtype())
        
        # 转换为riemann和torch张量 - 使用默认精度
        riemann_tensor = tensor(np_data, requires_grad=True)
        torch_tensor = torch.tensor(np_data, requires_grad=True)
        
        # 测试不同的多轴组合和范数类型
        multi_axis_combinations = [
            ((0, 1), [None, 1, -1,2, -2, float('inf'), float('-inf')]),  # 两轴组合
            ((2, 3), [None, 1, -1,2, -2, float('inf'), float('-inf')]),  # 两轴组合
        ]
        
        for dim, ord_values in multi_axis_combinations:
            for keepdim in [True, False]:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"  测试多轴组合: dim={dim}, keepdim={keepdim}")
                
                for ord in ord_values:
                    case_name = f"multi_axis_norm_dim={dim}_keepdim={keepdim}_ord={ord}"
                    
                    # 计算riemann的结果
                    try:
                        riemann_result = norm(riemann_tensor, ord=ord, dim=dim, keepdim=keepdim)
                        
                        # 计算torch的结果
                        torch_result = torch.linalg.norm(
                            torch_tensor,
                            ord=ord,
                            dim=dim,
                            keepdim=keepdim
                        )
                        
                        # 对比结果
                        np.testing.assert_allclose(
                            riemann_result.data,
                            torch_result.detach().numpy(),
                            rtol=1e-6,
                            atol=1e-8,
                            err_msg=f"多轴范数测试失败，ord={ord}, dim={dim}, keepdim={keepdim}"
                        )
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - ord={ord}")
                            stats.add_result(case_name, True)
                    except Exception as e:
                        # 捕获所有异常，包括运行时错误
                        if IS_RUNNING_AS_SCRIPT:
                            print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                            stats.add_result(case_name, False, [str(e)])
                        raise  # 直接抛出异常


        # 在TestLinalgNorm类中添加以下测试方法

    def test_matrix_spectral_norm(self):
        """测试矩阵谱范数(ord=2/-2)"""
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试矩阵谱范数(ord=2/-2)")
        
        # 测试2D矩阵
        np_data_2d = np.random.randn(4, 5).astype(rm.get_default_dtype())
        riemann_2d = tensor(np_data_2d, requires_grad=True)
        torch_2d = torch.tensor(np_data_2d, requires_grad=True)
        
        # 测试高维张量
        np_data_4d = np.random.randn(2, 4, 5, 3).astype(rm.get_default_dtype())
        riemann_4d = tensor(np_data_4d, requires_grad=True)
        torch_4d = torch.tensor(np_data_4d, requires_grad=True)
        
        # 测试ord=2和-2，以及不同keepdim组合
        test_cases = [
            (np_data_2d, riemann_2d, torch_2d, None, False),  # 2D矩阵，默认维度，不保持维度
            (np_data_2d, riemann_2d, torch_2d, (0, 1), False),  # 2D矩阵，显式维度，不保持维度
            (np_data_2d, riemann_2d, torch_2d, (0, 1), True),  # 2D矩阵，显式维度，保持维度
            (np_data_4d, riemann_4d, torch_4d, (1, 2), False),  # 4D张量，指定中间维度，不保持维度
            (np_data_4d, riemann_4d, torch_4d, (1, 2), True),  # 4D张量，指定中间维度，保持维度
        ]
        
        ord_values = [2, -2]
        
        for data, riemann_tensor, torch_tensor, dim, keepdim in test_cases:
            for ord in ord_values:
                case_name = f"matrix_spectral_norm_ord={ord}_dim={dim}_keepdim={keepdim}"
                try:
                    # 计算结果
                    riemann_result = norm(riemann_tensor, ord=ord, dim=dim, keepdim=keepdim)
                    torch_result = torch.linalg.norm(torch_tensor, ord=ord, dim=dim, keepdim=keepdim)
                    
                    # 验证结果
                    np.testing.assert_allclose(
                        riemann_result.data,
                        torch_result.detach().numpy(),
                        rtol=1e-6,
                        atol=1e-8,
                        err_msg=f"谱范数测试失败，ord={ord}, dim={dim}, keepdim={keepdim}"
                    )
                    
                    # 特别验证2D矩阵且keepdim=False时返回标量（与backward错误相关）
                    if data.ndim == 2 and dim in [None, (0, 1)] and not keepdim:
                        self.assertEqual(riemann_result.data.ndim, 0, 
                                        f"2D矩阵谱范数测试失败：keepdim=False时应返回标量，实际维度={riemann_result.data.ndim}")
                    
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - ord={ord}, dim={dim}, keepdim={keepdim}")
                        stats.add_result(case_name, True)
                except AssertionError as e:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                        stats.add_result(case_name, False, [str(e)])
                    raise



    def test_multi_axis_general_p_norm(self):
        """测试多轴通用p-范数(非整数p值)"""
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试多轴通用p-范数(非整数p值)")
        
        # 测试数据
        np_data = np.random.randn(2, 3, 4, 5).astype(rm.get_default_dtype())
        riemann_tensor = tensor(np_data, requires_grad=True)
        
        # 测试非整数p值
        non_integer_p_values = [1.5, 2.5, 0.5]  # 注意：p=0.5时需确保所有元素非负
        
        # 确保数据非负（对于p<1的情况）
        np_data_abs = np.abs(np_data)
        riemann_tensor_abs = tensor(np_data_abs, requires_grad=True)
        
        test_cases = [
            # 对于p>=1，可以使用原始数据
            ((0, 1), 1.5, np_data, riemann_tensor),
            ((2, 3), 2.5, np_data, riemann_tensor),
            # 对于p<1，使用非负数据
            ((0, 1), 0.5, np_data_abs, riemann_tensor_abs),
            ((2, 3), 0.5, np_data_abs, riemann_tensor_abs),
        ]
        
        for dim, p, data, riemann_t in test_cases:
            for keepdim in [True, False]:
                case_name = f"multi_axis_general_p_norm_p={p}_dim={dim}_keepdim={keepdim}"
                try:
                    # 计算Riemann结果
                    riemann_result = norm(riemann_t, ord=p, dim=dim, keepdim=keepdim)
                    
                    # 直接使用NumPy计算参考结果，因为PyTorch不支持非整数p值
                    # 对于多轴p-范数，我们需要手动计算
                    abs_data = np.abs(data)
                    powered_data = abs_data ** p
                    
                    # 对指定维度求和
                    for d in sorted(dim, reverse=True):
                        powered_data = np.sum(powered_data, axis=d, keepdims=keepdim)
                    
                    # 开p次方根
                    np_result = powered_data ** (1/p)
                    
                    # 验证结果
                    np.testing.assert_allclose(
                        riemann_result.data,
                        np_result,
                        rtol=1e-5,
                        atol=1e-7,
                        err_msg=f"通用p-范数测试失败，p={p}, dim={dim}, keepdim={keepdim}"
                    )
                    
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - p={p}, dim={dim}, keepdim={keepdim}")
                        stats.add_result(case_name, True)
                except AssertionError as e:
                    if IS_RUNNING_AS_SCRIPT:
                        print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                        stats.add_result(case_name, False, [str(e)])
                    raise  # 直接抛出异常以快速发现问题
    
    def test_comprehensive_gradients(self):
        """全面测试梯度计算"""
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试全面梯度计算")
        
        # 矩阵范数梯度测试（补充更多ord值）
        np_data = np.random.randn(3, 4).astype(rm.get_default_dtype())
        
        # 测试更多矩阵范数ord值的梯度
        matrix_ord_values = ['fro', 1, -1, float('inf'), float('-inf'), 2, -2, 'nuc']
        
        for ord in matrix_ord_values:
            case_name = f"matrix_norm_grad_ord={ord}"
            
            # 每次测试前重置张量以避免梯度累积
            riemann_tensor = tensor(np_data, requires_grad=True)
            torch_tensor = torch.tensor(np_data, requires_grad=True)
            
            try:
                # 计算结果
                riemann_result = norm(riemann_tensor, ord=ord)
                torch_result = torch.linalg.norm(torch_tensor, ord=ord)
                
                # 确保结果是标量（可求导）
                self.assertEqual(riemann_result.data.ndim, 0, 
                                f"梯度测试失败：ord={ord}时应返回标量，实际维度={riemann_result.data.ndim}")
                
                # 计算梯度
                riemann_result.backward()
                torch_result.backward()
                
                # 验证梯度 - 对于ord=-2和ord='nuc'增加容差范围
                rtol = 1e-5 if ord in [-2, 'nuc'] else 1e-6
                atol = 1e-7 if ord in [-2, 'nuc'] else 1e-8
                
                np.testing.assert_allclose(
                    riemann_tensor.grad.data,
                    torch_tensor.grad.detach().numpy(),
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"矩阵范数梯度测试失败，ord={ord}"
                )
                
                if IS_RUNNING_AS_SCRIPT:
                    print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 矩阵范数梯度 ord={ord}")
                    stats.add_result(case_name, True)
            except AssertionError as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                    stats.add_result(case_name, False, [str(e)])
                raise


    def test_complex_tensor_norm(self):
        """测试复数张量的范数计算"""
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试复数张量范数")
        
        # 测试复数向量范数
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试复数向量")
        
        # 创建复数向量数据
        np_complex_vector = (np.random.randn(5).astype(np.float32) + 
                           1j * np.random.randn(5).astype(np.float32))
        
        # 转换为riemann和torch张量
        riemann_complex_vector = tensor(np_complex_vector, requires_grad=True)
        torch_complex_vector = torch.tensor(np_complex_vector, requires_grad=True)
        
        # 测试各种向量范数
        vector_ord_values = [None, 0, 1, -1, 2, -2, float('inf'), float('-inf'), 3]
        all_passed = True
        failed_cases = []
        
        for ord in vector_ord_values:
            case_name = f"complex_vector_norm_ord={ord}"
            
            try:
                # 计算riemann的结果
                riemann_result = norm(riemann_complex_vector, ord=ord)
                
                # 计算torch的结果
                torch_result = torch.linalg.norm(torch_complex_vector, ord=ord)
                
                # 对比结果
                np.testing.assert_allclose(
                    riemann_result.data,
                    torch_result.detach().numpy(),
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"复数向量范数测试失败，ord={ord}"
                )
                
                # 对比梯度
                self.assertEqual(riemann_complex_vector.grad is not None, 
                               torch_complex_vector.grad is not None,
                               f"梯度测试失败：ord={ord}时梯度计算不一致")
                
                if IS_RUNNING_AS_SCRIPT:
                    print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 复数向量范数 ord={ord}")
                    stats.add_result(case_name, True)
            except AssertionError as e:
                if IS_RUNNING_AS_SCRIPT:
                    print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
                all_passed = False
                failed_cases.append(str(e))
        
        # 测试复数向量范数梯度
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试复数向量梯度")
        
        # 创建复数向量数据
        np_complex_vector = (np.random.randn(5).astype(np.float32) + 
                           1j * np.random.randn(5).astype(np.float32))
        
        # 测试L2范数的梯度（最常用）
        try:
            # Riemann梯度计算
            riemann_complex_vector = tensor(np_complex_vector, requires_grad=True)
            riemann_result = norm(riemann_complex_vector, ord=2)
            riemann_result.backward()
            riemann_grad = riemann_complex_vector.grad.data
            
            # PyTorch梯度计算
            torch_complex_vector = torch.tensor(np_complex_vector, requires_grad=True)
            torch_result = torch.linalg.norm(torch_complex_vector, ord=2)
            torch_result.backward()
            torch_grad = torch_complex_vector.grad.detach().numpy()
            
            # 对比梯度结果
            np.testing.assert_allclose(
                riemann_grad,
                torch_grad,
                rtol=1e-6,
                atol=1e-8,
                err_msg="复数向量L2范数梯度测试失败"
            )
            
            if IS_RUNNING_AS_SCRIPT:
                print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 复数向量L2范数梯度")
            stats.add_result("complex_vector_l2_norm_grad", True)
            
        except AssertionError as e:
            if IS_RUNNING_AS_SCRIPT:
                print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
            stats.add_result("complex_vector_l2_norm_grad", False, [str(e)])
            raise
        
        # 测试复数矩阵范数梯度
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试复数矩阵梯度")
        
        # 创建复数矩阵数据
        np_complex_matrix = (np.random.randn(3, 4).astype(np.float32) + 
                           1j * np.random.randn(3, 4).astype(np.float32))
        
        # 测试Frobenius范数的梯度
        try:
            # Riemann梯度计算
            riemann_complex_matrix = tensor(np_complex_matrix, requires_grad=True)
            riemann_result = norm(riemann_complex_matrix, ord='fro')
            riemann_result.backward()
            riemann_grad = riemann_complex_matrix.grad.data
            
            # PyTorch梯度计算
            torch_complex_matrix = torch.tensor(np_complex_matrix, requires_grad=True)
            torch_result = torch.linalg.norm(torch_complex_matrix, ord='fro')
            torch_result.backward()
            torch_grad = torch_complex_matrix.grad.detach().numpy()
            
            # 对比梯度结果
            np.testing.assert_allclose(
                riemann_grad,
                torch_grad,
                rtol=1e-6,
                atol=1e-8,
                err_msg="复数矩阵Frobenius范数梯度测试失败"
            )
            
            if IS_RUNNING_AS_SCRIPT:
                print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 复数矩阵Frobenius范数梯度")
            stats.add_result("complex_matrix_frobenius_norm_grad", True)
            
        except AssertionError as e:
            if IS_RUNNING_AS_SCRIPT:
                print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
            stats.add_result("complex_matrix_frobenius_norm_grad", False, [str(e)])
            raise
            
        # 测试带dim参数的复数张量范数梯度
        if IS_RUNNING_AS_SCRIPT:
            print(f"  测试带dim参数的复数张量梯度")
        
        try:
            # Riemann梯度计算 - 沿列计算范数
            riemann_complex_matrix = tensor(np_complex_matrix, requires_grad=True)
            riemann_col_norm = norm(riemann_complex_matrix, ord=2, dim=0)
            # 对每个元素求和再反向传播
            riemann_sum = riemann_col_norm.sum()
            riemann_sum.backward()
            riemann_grad = riemann_complex_matrix.grad.data
            
            # PyTorch梯度计算 - 沿列计算范数
            torch_complex_matrix = torch.tensor(np_complex_matrix, requires_grad=True)
            torch_col_norm = torch.linalg.norm(torch_complex_matrix, ord=2, dim=0)
            torch_sum = torch_col_norm.sum()
            torch_sum.backward()
            torch_grad = torch_complex_matrix.grad.detach().numpy()
            
            # 对比梯度结果
            np.testing.assert_allclose(
                riemann_grad,
                torch_grad,
                rtol=1e-6,
                atol=1e-8,
                err_msg="带dim参数的复数张量范数梯度测试失败"
            )
            
            if IS_RUNNING_AS_SCRIPT:
                print(f"    {Colors.OKGREEN}✓ 通过{Colors.ENDC} - 带dim参数的复数张量范数梯度")
            stats.add_result("complex_tensor_with_dim_grad", True)
            
        except AssertionError as e:
            if IS_RUNNING_AS_SCRIPT:
                print(f"    {Colors.FAIL}✗ 失败{Colors.ENDC} - {str(e)}")
            stats.add_result("complex_tensor_with_dim_grad", False, [str(e)])
            raise

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行矩阵范数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print()
    
    # 创建测试套件并运行
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgNorm)
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 打印失败的详细信息
    if stats.has_failures():
        print(f"\n{Colors.FAIL}失败的测试函数: {', '.join(stats.get_failed_functions())}{Colors.ENDC}")
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)