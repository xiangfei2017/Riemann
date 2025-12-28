import numpy as np
import torch
import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.riemann import tensor, set_grad_enabled
from src.riemann.dtype import float32, float64, int32, int64, bool_ as riemann_bool, complex64, complex128


class Colors:
    """用于美化输出的颜色类"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class StatisticsCollector:
    """用于收集测试统计信息的类"""
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.failures = []
        self.skipped_tests = 0
    
    def add_success(self):
        self.total_tests += 1
        self.passed_tests += 1
    
    def add_failure(self, test_name, error_msg):
        self.total_tests += 1
        self.failed_tests += 1
        self.failures.append((test_name, error_msg))
    
    def add_skip(self):
        self.total_tests += 1
        self.skipped_tests += 1
    
    def get_summary(self):
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        summary = f"\n{'='*50}\n"
        summary += f"Test Summary: {self.passed_tests} passed, {self.failed_tests} failed, {self.skipped_tests} skipped"
        summary += f"\nSuccess Rate: {success_rate:.2f}%"
        
        if self.failures:
            summary += f"\n\nFailed Tests:"
            for i, (test_name, error_msg) in enumerate(self.failures, 1):
                summary += f"\n{i}. {test_name}: {error_msg}"
        
        summary += f"\n{'='*50}\n"
        return summary


statistics = StatisticsCollector()

# 检测是否作为脚本直接运行
IS_RUNNING_AS_SCRIPT = False

def compare_values(riemann_result, torch_result, test_name, rtol=1e-05, atol=1e-08):
    """比较riemann和torch的结果是否一致"""
    try:
        # 比较数据类型
        riemann_dtype = riemann_result.dtype
        torch_dtype = torch_result.dtype
        
        # 特殊处理布尔类型
        if np.issubdtype(riemann_dtype, np.bool_) and torch_dtype == torch.bool:
            dtype_match = True
        else:
            dtype_match = str(riemann_dtype).lower() == str(torch_dtype).lower().replace('torch.', '')
        
        # 比较值
        np.testing.assert_allclose(
            riemann_result.data,
            torch_result.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Values do not match in {test_name}"
        )
        
        return True, None
    except AssertionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def print_test_result(test_name, success, error_msg=None, color=True):
    """打印测试结果"""
    if success:
        status = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if color else "PASS"
        statistics.add_success()
    else:
        status = f"{Colors.FAIL}FAIL{Colors.ENDC}" if color else "FAIL"
        statistics.add_failure(test_name, error_msg)
    
    # 直接输出测试结果，移除多余的换行符
    print(f"{test_name}: {status}")
    if not success and error_msg:
        print(f"  Error: {error_msg}")


class TestOperatorFunctions(unittest.TestCase):
    """测试tensordef中的各种运算符函数"""
    
    # 类属性，用于标记第一个测试是否已经运行
    _test_started = False
    
    def setUp(self):
        """每个测试用例执行前的设置"""
        set_grad_enabled(True)
        
        # 仅在独立运行脚本时，为非第一个测试方法打印空行
        if IS_RUNNING_AS_SCRIPT:
            # 获取当前正在运行的方法名（简化版本）
            if not TestOperatorFunctions._test_started:
                # 标记第一个测试已运行
                TestOperatorFunctions._test_started = True
            else:
                # 不是第一个测试方法，则打印空行
                print()
    
    def tearDown(self):
        """每个测试用例执行后的清理"""
        pass
    
    def test_arithmetic_operators(self):
        """测试算术运算符：+、-、*、/、取负号、取正号、幂运算"""
        # 测试不同数据类型
        dtypes = [
            (float32, torch.float32),
            (float64, torch.float64),
            (int32, torch.int32),
            (int64, torch.int64)
        ]
        
        for riemann_dtype, torch_dtype in dtypes:
            # 跳过不支持梯度的类型
            requires_grad = np.issubdtype(riemann_dtype, np.floating)
            
            # 创建测试数据
            data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=riemann_dtype)
            data2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=riemann_dtype)
            
            # 创建张量
            x = tensor(data1, requires_grad=requires_grad)
            y = tensor(data2, requires_grad=requires_grad)
            
            # 创建对应的PyTorch张量
            tx = torch.tensor(data1, dtype=torch_dtype, requires_grad=requires_grad)
            ty = torch.tensor(data2, dtype=torch_dtype, requires_grad=requires_grad)
            
            # 测试加法
            add_result = x + y
            tadd_result = tx + ty
            success, error = compare_values(add_result, tadd_result, f"Addition {riemann_dtype}")
            print_test_result(f"Addition {riemann_dtype}", success, error)
            
            # 测试减法
            sub_result = x - y
            tsub_result = tx - ty
            success, error = compare_values(sub_result, tsub_result, f"Subtraction {riemann_dtype}")
            print_test_result(f"Subtraction {riemann_dtype}", success, error)
            
            # 测试乘法
            mul_result = x * y
            tmul_result = tx * ty
            success, error = compare_values(mul_result, tmul_result, f"Multiplication {riemann_dtype}")
            print_test_result(f"Multiplication {riemann_dtype}", success, error)
            
            # 测试除法（跳过整数除法的精确比较，因为行为可能不同）
            if not np.issubdtype(riemann_dtype, np.integer):
                div_result = x / y
                tdiv_result = tx / ty
                success, error = compare_values(div_result, tdiv_result, f"Division {riemann_dtype}")
                print_test_result(f"Division {riemann_dtype}", success, error)
            
            # 测试取负号
            neg_result = -x
            tneg_result = -tx
            success, error = compare_values(neg_result, tneg_result, f"Negation {riemann_dtype}")
            print_test_result(f"Negation {riemann_dtype}", success, error)
            
            # 测试取正号
            pos_result = +x
            tpos_result = +tx
            success, error = compare_values(pos_result, tpos_result, f"Positive {riemann_dtype}")
            print_test_result(f"Positive {riemann_dtype}", success, error)
            
            # 测试幂运算（pow函数）
            pow_result = x ** y
            tpow_result = tx ** ty
            success, error = compare_values(pow_result, tpow_result, f"Power {riemann_dtype}")
            print_test_result(f"Power {riemann_dtype}", success, error)
            
            # 测试梯度（仅对浮点类型）
            if requires_grad:
                # 加法梯度测试
                add_result = x + y
                add_result.sum().backward()
                
                tadd_result = tx + ty
                tadd_result.sum().backward()
                
                # 比较梯度
                success_x_grad, error_x_grad = compare_values(x.grad, tx.grad, f"Addition gradient x {riemann_dtype}")
                success_y_grad, error_y_grad = compare_values(y.grad, ty.grad, f"Addition gradient y {riemann_dtype}")
                
                print_test_result(f"Addition gradient x {riemann_dtype}", success_x_grad, error_x_grad)
                print_test_result(f"Addition gradient y {riemann_dtype}", success_y_grad, error_y_grad)
                
                # 乘法梯度测试
                # 重置梯度
                x.grad = None
                y.grad = None
                
                mul_result = x * y
                mul_result.sum().backward()
                
                # 重置PyTorch梯度
                tx.grad = None
                ty.grad = None
                
                tmul_result = tx * ty
                tmul_result.sum().backward()
                
                # 比较梯度
                success_x_grad, error_x_grad = compare_values(x.grad, tx.grad, f"Multiplication gradient x {riemann_dtype}")
                success_y_grad, error_y_grad = compare_values(y.grad, ty.grad, f"Multiplication gradient y {riemann_dtype}")
                
                print_test_result(f"Multiplication gradient x {riemann_dtype}", success_x_grad, error_x_grad)
                print_test_result(f"Multiplication gradient y {riemann_dtype}", success_y_grad, error_y_grad)
                
                # 幂运算梯度测试
                # 重置梯度
                x.grad = None
                y.grad = None
                
                pow_result = x ** y
                pow_result.sum().backward()
                
                # 重置PyTorch梯度
                tx.grad = None
                ty.grad = None
                
                tpow_result = tx ** ty
                tpow_result.sum().backward()
                
                # 比较梯度
                success_x_grad, error_x_grad = compare_values(x.grad, tx.grad, f"Power gradient x {riemann_dtype}")
                success_y_grad, error_y_grad = compare_values(y.grad, ty.grad, f"Power gradient y {riemann_dtype}")
                
                print_test_result(f"Power gradient x {riemann_dtype}", success_x_grad, error_x_grad)
                print_test_result(f"Power gradient y {riemann_dtype}", success_y_grad, error_y_grad)
        # 输出分隔空行
        print()

    def test_comparison_operators(self):
        """测试比较运算符：>、>=、<、<=、==、!="""
        # 测试不同数据类型
        dtypes = [
            (float32, torch.float32),
            (float64, torch.float64),
            (int32, torch.int32),
            (int64, torch.int64),
            (riemann_bool, torch.bool)
        ]
        
        for riemann_dtype, torch_dtype in dtypes:
            # 创建测试数据
            data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=riemann_dtype)
            data2 = np.array([[2.0, 2.0], [2.0, 4.0]], dtype=riemann_dtype)
            
            # 创建张量
            x = tensor(data1)
            y = tensor(data2)
            
            # 创建对应的PyTorch张量
            tx = torch.tensor(data1, dtype=torch_dtype)
            ty = torch.tensor(data2, dtype=torch_dtype)
            
            # 测试大于
            gt_result = x > y
            tgt_result = tx > ty
            success, error = compare_values(gt_result, tgt_result, f"Greater than {riemann_dtype}")
            print_test_result(f"Greater than {riemann_dtype}", success, error)
            
            # 测试大于等于
            ge_result = x >= y
            tge_result = tx >= ty
            success, error = compare_values(ge_result, tge_result, f"Greater or equal {riemann_dtype}")
            print_test_result(f"Greater or equal {riemann_dtype}", success, error)
            
            # 测试小于
            lt_result = x < y
            tlt_result = tx < ty
            success, error = compare_values(lt_result, tlt_result, f"Less than {riemann_dtype}")
            print_test_result(f"Less than {riemann_dtype}", success, error)
            
            # 测试小于等于
            le_result = x <= y
            tle_result = tx <= ty
            success, error = compare_values(le_result, tle_result, f"Less or equal {riemann_dtype}")
            print_test_result(f"Less or equal {riemann_dtype}", success, error)
            
            # 测试等于
            eq_result = x == y
            teq_result = tx == ty
            success, error = compare_values(eq_result, teq_result, f"Equal {riemann_dtype}")
            print_test_result(f"Equal {riemann_dtype}", success, error)
            
            # 测试不等于
            ne_result = x != y
            tne_result = tx != ty
            success, error = compare_values(ne_result, tne_result, f"Not equal {riemann_dtype}")
            print_test_result(f"Not equal {riemann_dtype}", success, error)
        # 输出分隔空行
        print()

    def test_complex_arithmetic_operators(self):
        """测试复数类型的算术运算符：+、-、*、/、power"""
        # 测试不同复数数据类型
        dtypes = [
            (complex64, torch.complex64),
            (complex128, torch.complex128)
        ]
        
        for riemann_dtype, torch_dtype in dtypes:
            requires_grad = True
            
            # 创建测试数据（复数数据）
            data1 = np.array([[1.0+2.0j, 3.0+4.0j], [5.0+6.0j, 7.0+8.0j]], dtype=riemann_dtype)
            data2 = np.array([[2.0+1.0j, 4.0+3.0j], [6.0+5.0j, 8.0+7.0j]], dtype=riemann_dtype)
            
            # 创建张量
            x = tensor(data1, requires_grad=requires_grad)
            y = tensor(data2, requires_grad=requires_grad)
            
            # 创建对应的PyTorch张量
            tx = torch.tensor(data1, dtype=torch_dtype, requires_grad=requires_grad)
            ty = torch.tensor(data2, dtype=torch_dtype, requires_grad=requires_grad)
            
            # 测试加法
            add_result = x + y
            tadd_result = tx + ty
            success, error = compare_values(add_result, tadd_result, f"Complex Addition {riemann_dtype}")
            print_test_result(f"Complex Addition {riemann_dtype}", success, error)
            
            # 测试减法
            sub_result = x - y
            tsub_result = tx - ty
            success, error = compare_values(sub_result, tsub_result, f"Complex Subtraction {riemann_dtype}")
            print_test_result(f"Complex Subtraction {riemann_dtype}", success, error)
            
            # 测试乘法
            mul_result = x * y
            tmul_result = tx * ty
            success, error = compare_values(mul_result, tmul_result, f"Complex Multiplication {riemann_dtype}")
            print_test_result(f"Complex Multiplication {riemann_dtype}", success, error)
            
            # 测试除法
            div_result = x / y
            tdiv_result = tx / ty
            success, error = compare_values(div_result, tdiv_result, f"Complex Division {riemann_dtype}")
            print_test_result(f"Complex Division {riemann_dtype}", success, error)
            
            # 测试幂运算
            power_result = x ** 2.0  # 复数的平方
            tpower_result = tx ** 2.0
            success, error = compare_values(power_result, tpower_result, f"Complex Power {riemann_dtype}")
            print_test_result(f"Complex Power {riemann_dtype}", success, error)
            
            # 测试梯度 - 使用实部进行反向传播
            if requires_grad:
                # 加法梯度测试
                x.grad = None
                y.grad = None
                add_result = x + y
                # 对实部求和后反向传播
                add_result.real.sum().backward()
                
                tx.grad = None
                ty.grad = None
                tadd_result = tx + ty
                tadd_result.real.sum().backward()
                
                # 比较梯度
                success_x_grad, error_x_grad = compare_values(x.grad, tx.grad, f"Complex Addition gradient x {riemann_dtype}")
                success_y_grad, error_y_grad = compare_values(y.grad, ty.grad, f"Complex Addition gradient y {riemann_dtype}")
                
                print_test_result(f"Complex Addition gradient x {riemann_dtype}", success_x_grad, error_x_grad)
                print_test_result(f"Complex Addition gradient y {riemann_dtype}", success_y_grad, error_y_grad)
                
                # 减法梯度测试
                x.grad = None
                y.grad = None
                sub_result = x - y
                # 对实部求和后反向传播
                sub_result.real.sum().backward()
                
                tx.grad = None
                ty.grad = None
                tsub_result = tx - ty
                tsub_result.real.sum().backward()
                
                # 比较梯度
                success_x_grad, error_x_grad = compare_values(x.grad, tx.grad, f"Complex Subtraction gradient x {riemann_dtype}")
                success_y_grad, error_y_grad = compare_values(y.grad, ty.grad, f"Complex Subtraction gradient y {riemann_dtype}")
                
                print_test_result(f"Complex Subtraction gradient x {riemann_dtype}", success_x_grad, error_x_grad)
                print_test_result(f"Complex Subtraction gradient y {riemann_dtype}", success_y_grad, error_y_grad)
                
                # 乘法梯度测试
                x.grad = None
                y.grad = None
                mul_result = x * y
                # 对实部求和后反向传播
                mul_result.real.sum().backward()
                
                tx.grad = None
                ty.grad = None
                tmul_result = tx * ty
                tmul_result.real.sum().backward()
                
                # 比较梯度
                success_x_grad, error_x_grad = compare_values(x.grad, tx.grad, f"Complex Multiplication gradient x {riemann_dtype}")
                success_y_grad, error_y_grad = compare_values(y.grad, ty.grad, f"Complex Multiplication gradient y {riemann_dtype}")
                
                print_test_result(f"Complex Multiplication gradient x {riemann_dtype}", success_x_grad, error_x_grad)
                print_test_result(f"Complex Multiplication gradient y {riemann_dtype}", success_y_grad, error_y_grad)
                
                # 除法梯度测试
                x.grad = None
                y.grad = None
                div_result = x / y
                # 对实部求和后反向传播
                div_result.real.sum().backward()
                
                tx.grad = None
                ty.grad = None
                tdiv_result = tx / ty
                tdiv_result.real.sum().backward()
                
                # 比较梯度
                success_x_grad, error_x_grad = compare_values(x.grad, tx.grad, f"Complex Division gradient x {riemann_dtype}")
                success_y_grad, error_y_grad = compare_values(y.grad, ty.grad, f"Complex Division gradient y {riemann_dtype}")
                
                print_test_result(f"Complex Division gradient x {riemann_dtype}", success_x_grad, error_x_grad)
                print_test_result(f"Complex Division gradient y {riemann_dtype}", success_y_grad, error_y_grad)
                
                # 幂运算梯度测试 - 对x的梯度
                x.grad = None
                y.grad = None
                power_result = x ** 2.0
                # 对实部求和后反向传播
                power_result.real.sum().backward()
                
                tx.grad = None
                ty.grad = None
                tpower_result = tx ** 2.0
                tpower_result.real.sum().backward()
                
                # 比较梯度
                success_x_grad, error_x_grad = compare_values(x.grad, tx.grad, f"Complex Power gradient x {riemann_dtype}")
                print_test_result(f"Complex Power gradient x {riemann_dtype}", success_x_grad, error_x_grad)
                
                # 幂运算梯度测试 - 对y的梯度
                # 使用不同的测试数据来更清晰地测试对y的梯度
                x2 = tensor(np.array([[1.0+1.0j, 2.0+2.0j]], dtype=riemann_dtype), requires_grad=True)
                y2 = tensor(np.array([[2.0, 3.0]], dtype=riemann_dtype), requires_grad=True)
                
                tx2 = torch.tensor(np.array([[1.0+1.0j, 2.0+2.0j]], dtype=riemann_dtype), 
                                  dtype=torch_dtype, requires_grad=True)
                ty2 = torch.tensor(np.array([[2.0, 3.0]], dtype=riemann_dtype), 
                                  dtype=torch_dtype, requires_grad=True)
                
                # 重置梯度
                x2.grad = None
                y2.grad = None
                power_result2 = x2 ** y2
                # 对实部求和后反向传播
                power_result2.real.sum().backward()
                
                # 重置PyTorch梯度
                tx2.grad = None
                ty2.grad = None
                tpower_result2 = tx2 ** ty2
                tpower_result2.real.sum().backward()
                
                # 比较对y的梯度
                success_y_grad, error_y_grad = compare_values(y2.grad, ty2.grad, f"Complex Power gradient y {riemann_dtype}")
                print_test_result(f"Complex Power gradient y {riemann_dtype}", success_y_grad, error_y_grad)
        
        # 输出分隔空行
        print()
        
    def test_bitwise_operators(self):
        """测试位运算符"""
        # 测试不同整数数据类型
        dtypes = [
            (int32, torch.int32),
            (int64, torch.int64)
        ]
        
        for riemann_dtype, torch_dtype in dtypes:
            # 创建测试数据
            data1 = np.array([[1, 2], [3, 4]], dtype=riemann_dtype)
            data2 = np.array([[1, 3], [2, 4]], dtype=riemann_dtype)
            
            # 创建张量
            x = tensor(data1)
            y = tensor(data2)
            
            # 创建对应的PyTorch张量
            tx = torch.tensor(data1, dtype=torch_dtype)
            ty = torch.tensor(data2, dtype=torch_dtype)
            
            # 测试按位与
            and_result = x & y
            tand_result = tx & ty
            success, error = compare_values(and_result, tand_result, f"Bitwise AND {riemann_dtype}")
            print_test_result(f"Bitwise AND {riemann_dtype}", success, error)
            
            # 测试按位或
            or_result = x | y
            tor_result = tx | ty
            success, error = compare_values(or_result, tor_result, f"Bitwise OR {riemann_dtype}")
            print_test_result(f"Bitwise OR {riemann_dtype}", success, error)
            
            # 测试按位异或
            xor_result = x ^ y
            txor_result = tx ^ ty
            success, error = compare_values(xor_result, txor_result, f"Bitwise XOR {riemann_dtype}")
            print_test_result(f"Bitwise XOR {riemann_dtype}", success, error)
            
            # 测试按位取反
            not_result = ~x
            tnot_result = ~tx
            success, error = compare_values(not_result, tnot_result, f"Bitwise NOT {riemann_dtype}")
            print_test_result(f"Bitwise NOT {riemann_dtype}", success, error)
            
            # 测试左移
            shift_amount = 1
            lshift_result = x << shift_amount
            tlshift_result = tx << shift_amount
            success, error = compare_values(lshift_result, tlshift_result, f"Left shift {riemann_dtype}")
            print_test_result(f"Left shift {riemann_dtype}", success, error)
            
            # 测试右移
            rshift_result = x >> shift_amount
            trshift_result = tx >> shift_amount
            success, error = compare_values(rshift_result, trshift_result, f"Right shift {riemann_dtype}")
            print_test_result(f"Right shift {riemann_dtype}", success, error)
            
        # 输出分隔空行
        print()
        
    def test_type_conversion_functions(self):
        """测试类型转换函数"""
        # 创建测试数据
        data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        
        # 创建张量
        x = tensor(data)
        
        # 测试转换为float32
        float32_result = x.type(float32)
        self.assertEqual(float32_result.dtype, float32)
        print_test_result("Type conversion to float32", True)
        
        # 测试转换为float64
        float64_result = x.type(float64)
        self.assertEqual(float64_result.dtype, float64)
        print_test_result("Type conversion to float64", True)
        
        # 测试转换为int32
        int32_result = x.type(int32)
        self.assertEqual(int32_result.dtype, int32)
        print_test_result("Type conversion to int32", True)
        
        # 测试转换为int64
        int64_result = x.type(int64)
        self.assertEqual(int64_result.dtype, int64)
        print_test_result("Type conversion to int64", True)
        
        # 测试float()方法
        float_result = x.float()
        self.assertEqual(float_result.dtype, float32)
        print_test_result("float() method", True)
        
        # 测试double()方法
        double_result = x.double()
        self.assertEqual(double_result.dtype, float64)
        print_test_result("double() method", True)
        
        # 测试type_as方法
        y = tensor(np.array([1, 2]), dtype=int32)        
        type_as_result = x.type_as(y)
        self.assertEqual(type_as_result.dtype, int32)
        print_test_result("type_as() method", True)
        
        # 测试to方法
        to_result = x.to(int64)
        self.assertEqual(to_result.dtype, int64)
        print_test_result("to() method", True)
        
        # 输出分隔空行
        print()
        
    def test_any_all_functions(self):
        """测试any和all函数"""
        # 创建测试数据 - 全False以确保all()返回False
        bool_data = np.array([[False, False], [False, False]], dtype=bool)
        
        # 创建张量
        x = tensor(bool_data)
        tx = torch.tensor(bool_data, dtype=torch.bool)
        
        # 测试all()函数
        all_result = x.all()
        tall_result = tx.all()
        success, error = compare_values(all_result, tall_result, "all() function")
        print_test_result("all() function", success, error)
        
        # 测试all(dim=0)函数
        all_dim0_result = x.all(dim=0)
        tall_dim0_result = tx.all(dim=0)
        success, error = compare_values(all_dim0_result, tall_dim0_result, "all(dim=0) function")
        print_test_result("all(dim=0) function", success, error)
        
        # 测试all(dim=1)函数
        all_dim1_result = x.all(dim=1)
        tall_dim1_result = tx.all(dim=1)
        success, error = compare_values(all_dim1_result, tall_dim1_result, "all(dim=1) function")
        print_test_result("all(dim=1) function", success, error)
        
        # 测试all(keepdim=True)函数
        all_keepdim_result = x.all(dim=0, keepdim=True)
        tall_keepdim_result = tx.all(dim=0, keepdim=True)
        success, error = compare_values(all_keepdim_result, tall_keepdim_result, "all(keepdim=True) function")
        print_test_result("all(keepdim=True) function", success, error)
        
        # 创建有True值的数据测试any()
        bool_data_any = np.array([[False, True], [False, False]], dtype=bool)
        x_any = tensor(bool_data_any)
        tx_any = torch.tensor(bool_data_any, dtype=torch.bool)
        
        # 测试any()函数
        any_result = x_any.any()
        tany_result = tx_any.any()
        success, error = compare_values(any_result, tany_result, "any() function")
        print_test_result("any() function", success, error)
        
        # 测试any(dim=0)函数
        any_dim0_result = x_any.any(dim=0)
        tany_dim0_result = tx_any.any(dim=0)
        success, error = compare_values(any_dim0_result, tany_dim0_result, "any(dim=0) function")
        print_test_result("any(dim=0) function", success, error)
        
        # 测试any(dim=1)函数
        any_dim1_result = x_any.any(dim=1)
        tany_dim1_result = tx_any.any(dim=1)
        success, error = compare_values(any_dim1_result, tany_dim1_result, "any(dim=1) function")
        print_test_result("any(dim=1) function", success, error)
        
        # 测试any(keepdim=True)函数
        any_keepdim_result = x_any.any(dim=0, keepdim=True)
        tany_keepdim_result = tx_any.any(dim=0, keepdim=True)
        success, error = compare_values(any_keepdim_result, tany_keepdim_result, "any(keepdim=True) function")
        print_test_result("any(keepdim=True) function", success, error)
        
        # 输出分隔空行
        print()
        
    def test_comparison_operators_only(self):
        """测试元素级比较运算符"""
        # 创建测试数据
        data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        data2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        data3 = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
        
        # 创建张量
        x = tensor(data1)
        y = tensor(data2)
        z = tensor(data3)
        
        # 创建对应的PyTorch张量
        tx = torch.tensor(data1, dtype=torch.float32)
        ty = torch.tensor(data2, dtype=torch.float32)
        tz = torch.tensor(data3, dtype=torch.float32)
        
        # 测试等于操作符
        try:
            eq_result = (x == y)
            teq_result = (tx == ty)
            success_eq, error_eq = compare_values(eq_result, teq_result, "Equal operator (equal tensors)")
            print_test_result("Equal operator (equal tensors)", success_eq, error_eq)
            
            eq_result = (x == z)
            teq_result = (tx == tz)
            success_eq, error_eq = compare_values(eq_result, teq_result, "Equal operator (unequal tensors)")
            print_test_result("Equal operator (unequal tensors)", success_eq, error_eq)
            
        except Exception as e:
            print_test_result("Equal operator", False, str(e))
        
        # 测试不等于操作符
        try:
            ne_result = (x != z)
            tne_result = (tx != tz)
            success, error = compare_values(ne_result, tne_result, "Not equal operator")
            print_test_result("Not equal operator", success, error)
        except Exception as e:
            print_test_result("Not equal operator", False, str(e))
        
        # 输出分隔空行
        print()
        
    def test_scalar_operations(self):
        """测试与标量的操作"""
        # 创建测试数据
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        scalar = 2.0
        
        # 创建张量
        x = tensor(data, requires_grad=True)
        tx = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        
        # 测试标量加法
        add_scalar_result = x + scalar
        tadd_scalar_result = tx + scalar
        success, error = compare_values(add_scalar_result, tadd_scalar_result, "Addition with scalar")
        print_test_result("Addition with scalar", success, error)
        
        # 测试标量减法
        sub_scalar_result = x - scalar
        tsub_scalar_result = tx - scalar
        success, error = compare_values(sub_scalar_result, tsub_scalar_result, "Subtraction with scalar")
        print_test_result("Subtraction with scalar", success, error)
        
        # 测试标量乘法
        mul_scalar_result = x * scalar
        tmul_scalar_result = tx * scalar
        success, error = compare_values(mul_scalar_result, tmul_scalar_result, "Multiplication with scalar")
        print_test_result("Multiplication with scalar", success, error)
        
        # 测试标量除法
        div_scalar_result = x / scalar
        tdiv_scalar_result = tx / scalar
        success, error = compare_values(div_scalar_result, tdiv_scalar_result, "Division with scalar")
        print_test_result("Division with scalar", success, error)
        
        # 测试标量梯度
        mul_scalar_result.sum().backward()
        tmul_scalar_result.sum().backward()
        
        success, error = compare_values(x.grad, tx.grad, "Gradient with scalar multiplication")
        print_test_result("Gradient with scalar multiplication", success, error)
        
        # 输出分隔空行
        print()
        
    def test_broadcasting_operations(self):
        """测试广播操作"""
        # 创建测试数据
        data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        data2 = np.array([[1.0], [2.0]], dtype=np.float32)
        
        # 创建张量
        x = tensor(data1, requires_grad=True)
        y = tensor(data2, requires_grad=True)
        
        # 创建对应的PyTorch张量
        tx = torch.tensor(data1, dtype=torch.float32, requires_grad=True)
        ty = torch.tensor(data2, dtype=torch.float32, requires_grad=True)
        
        # 测试广播加法
        add_broadcast_result = x + y
        tadd_broadcast_result = tx + ty
        success, error = compare_values(add_broadcast_result, tadd_broadcast_result, "Broadcast addition")
        print_test_result("Broadcast addition", success, error)
        
        # 测试广播乘法
        mul_broadcast_result = x * y
        tmul_broadcast_result = tx * ty
        success, error = compare_values(mul_broadcast_result, tmul_broadcast_result, "Broadcast multiplication")
        print_test_result("Broadcast multiplication", success, error)
        
        # 测试广播梯度
        add_broadcast_result.sum().backward()
        tadd_broadcast_result.sum().backward()
        
        success_x_grad, error_x_grad = compare_values(x.grad, tx.grad, "Broadcast addition gradient x")
        success_y_grad, error_y_grad = compare_values(y.grad, ty.grad, "Broadcast addition gradient y")
        
        print_test_result("Broadcast addition gradient x", success_x_grad, error_x_grad)
        print_test_result("Broadcast addition gradient y", success_y_grad, error_y_grad)
        
        # 输出分隔空行
        print()
        
def clear_screen():
    """清除终端屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
def main():
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行运算符函数测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: True{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestOperatorFunctions)
    
    # 运行测试，禁用默认输出
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    print(statistics.get_summary())
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()
