import os
import sys
import time
import numpy as np
import unittest

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import riemann as rm
from riemann import optim

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# 定义各种优化测试函数
PI = rm.tensor(np.pi,dtype=rm.get_default_dtype())
E = rm.tensor(np.e,dtype=rm.get_default_dtype())

# 1. Rosenbrock函数（香蕉函数）
def rosenbrock_2d(x, y):
    """Rosenbrock函数（香蕉函数），极小值在(1, 1)处，值为0"""
    return 100. * (y - x**2.)**2. + (1. - x)**2.

# 2. Himmelblau函数
def himmelblau(x, y):
    """Himmelblau函数，有4个全局最小值点，值均为0"""
    return (x**2. + y - 11.)**2. + (x + y**2. - 7.)**2.

# 3. Rastrigin函数（二维版本）
def rastrigin_2d(x, y):
    """Rastrigin函数（二维版本），极小值在(0, 0)处，值为0"""
    A = 10.
    return A * 2. + (x**2. - A * rm.cos(2. * PI * x)) + (y**2. - A * rm.cos(2. * PI * y))

# 4. Sphere函数（二维版本）
def sphere_2d(x, y):
    """Sphere函数（二维版本），极小值在(0, 0)处，值为0"""
    return x**2. + y**2.

# 5. Ackley函数（二维版本）
def ackley_2d(x, y):
    """Ackley函数（二维版本），极小值在(0, 0)处，值为0"""
    a = 20.
    b = 0.2
    c = 2. * PI
    term1 = -a * rm.exp(-b * rm.sqrt(0.5 * (x**2. + y**2.)))
    term2 = -rm.exp(0.5 * (rm.cos(c * x) + rm.cos(c * y)))
    return term1 + term2 + a + E

# 6. Beale函数
def beale_2d(x, y):
    """Beale函数，极小值在(3, 0.5)处，值为0"""
    term1 = (1.5 - x + x*y)**2.
    term2 = (2.25 - x + x*y**2.)**2.
    term3 = (2.625 - x + x*y**3.)**2.
    return term1 + term2 + term3

# 定义测试函数及其配置
TEST_FUNCTIONS = {
    'Rosenbrock': {
        'func': rosenbrock_2d,
        'start_x': -1.2,
        'start_y': 1.0,
        'expected_min': (1.0, 1.0),
        'description': 'f(x,y) = 100*(y-x^2)^2 + (1-x)^2'
    },
    'Himmelblau': {
        'func': himmelblau,
        'start_x': 0.0,
        'start_y': 0.0,
        'expected_min': [(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)],
        'description': 'f(x,y) = (x² + y - 11)² + (x + y² - 7)²'
    },
    'Rastrigin': {
        'func': rastrigin_2d,
        'start_x': 1.0,
        'start_y': 1.0,
        'expected_min': (0.0, 0.0),
        'description': 'f(x,y) = 20 + (x² - 10·cos(2πx)) + (y² - 10·cos(2πy))'
    },
    'Sphere': {
        'func': sphere_2d,
        'start_x': 1.0,
        'start_y': -1.0,
        'expected_min': (0.0, 0.0),
        'description': 'f(x,y) = x² + y²'
    },
    'Ackley': {
        'func': ackley_2d,
        'start_x': 1.0,
        'start_y': 1.0,
        'expected_min': (0.0, 0.0),
        'description': '复杂指数余弦函数'
    },
    'Beale': {
        'func': beale_2d,
        'start_x': 0.0,
        'start_y': 0.0,
        'expected_min': (3.0, 0.5),
        'description': 'f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²'
    }
}

# 优化器参数配置
OPTIMIZER_CONFIGS = {
    'GD': {
        'class': optim.GD,
        'params': {'lr': 0.001},
        'max_iter': 1000,
        'tolerance': 1e-6
    },
    'SGD': {
        'class': optim.SGD,
        'params': {'lr': 0.0005, 'momentum': 0.9},
        'max_iter': 1000,
        'tolerance': 1e-6
    },
    'Adam': {
        'class': optim.Adam,
        'params': {'lr': 0.05, 'betas': (0.9, 0.999), 'eps': 1e-8},
        'max_iter': 1000,
        'tolerance': 1e-6
    },
    'Adagrad': {
        'class': optim.Adagrad,
        'params': {'lr': 0.3, 'initial_accumulator_value': 0.1, 'eps': 1e-7},
        'max_iter': 1000,
        'tolerance': 1e-6
    },
    'LBFGS': {
        'class': optim.LBFGS,
        'params': {'lr': 1.0, 'max_iter': 50, 'max_eval': 50, 
                  'tolerance_grad': 1e-4, 'tolerance_change': 1e-9, 'history_size': 20},
        'max_iter': 3,
        'tolerance': 1e-6
    }
}

# 优化器测试基类
class OptimizerTestBase:
    def run_optimizer_test(self, optimizer_name, optimizer_config, test_function_name, test_func_config):
        """运行单个优化器在单个测试函数上的测试"""
        func = test_func_config['func']
        start_x = test_func_config['start_x']
        start_y = test_func_config['start_y']
        
        # 初始化参数
        x = rm.tensor(start_x, requires_grad=True)
        y = rm.tensor(start_y, requires_grad=True)
        params = [x, y]
        # print(f'x.dtype={x.dtype},y.dtype={y.dtype}')
        
        # 创建优化器
        optimizer = optimizer_config['class'](params, **optimizer_config['params'])
        max_iterations = optimizer_config['max_iter']
        tolerance = optimizer_config['tolerance']
        
        start_time = time.time()
        prev_loss = float('inf')
        best_loss = float('inf')
        best_x = x.item()
        best_y = y.item()
        
        # print(f'prev_loss.dtype={prev_loss.dtype},best_loss.dtype={best_loss.dtype}')

        try:
            # 特殊处理LBFGS优化器
            if optimizer_name == 'LBFGS':
                # LBFGS需要闭包函数
                def closure():
                    optimizer.zero_grad(True)
                    loss = func(x, y)
                    loss.backward()
                    return loss
                
                # LBFGS的外部迭代
                for i in range(max_iterations):
                    loss = optimizer.step(closure)
                    current_loss = loss.item()
                    x_val = x.item()
                    y_val = y.item()
                    
                    # 更新最佳值
                    if current_loss < best_loss and np.isfinite(current_loss):
                        best_loss = current_loss
                        best_x = x_val
                        best_y = y_val
                    
                    # 检查是否溢出
                    if not np.isfinite(current_loss) or not np.isfinite(x_val) or not np.isfinite(y_val):
                        break
                    
                    # 检查收敛
                    if current_loss < tolerance:
                        break
                    
                    # 检查参数范围
                    if abs(x_val) > 100 or abs(y_val) > 100:
                        break
            else:
                # 其他优化器
                for i in range(max_iterations):
                    # 前向传播计算损失
                    loss = func(x, y)
                    # print(f'loss.dtype={loss.dtype}')
                    current_loss = loss.item()
                    
                    # 更新最佳值
                    if current_loss < best_loss and np.isfinite(current_loss):
                        best_loss = current_loss
                        best_x = x.item()
                        best_y = y.item()
                    
                    # 检查数值稳定性
                    if not np.isfinite(current_loss):
                        break
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 更新参数
                    optimizer.step()
                    
                    # 检查参数范围
                    if abs(x.item()) > 100 or abs(y.item()) > 100:
                        break
                    
                    # 检查收敛
                    if abs(prev_loss - current_loss) < tolerance:
                        break
                    
                    prev_loss = current_loss
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            return {
                'function': test_function_name,
                'optimizer': optimizer_name,
                'final_loss': best_loss,
                'final_params': (best_x, best_y),
                'time_taken': time_taken,
                'success': np.isfinite(best_loss) and (abs(best_x) < 100 and abs(best_y) < 100)
            }
            
        except Exception as e:
            end_time = time.time()
            time_taken = end_time - start_time
            return {
                'function': test_function_name,
                'optimizer': optimizer_name,
                'final_loss': float('inf'),
                'final_params': (float('nan'), float('nan')),
                'time_taken': time_taken,
                'success': False,
                'error': str(e)
            }

# 测试类
class TestOptimizers(unittest.TestCase, OptimizerTestBase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)
    
    def test_optimizer_gd(self):
        """测试GD优化器在所有测试函数上的性能"""
        print(f"\n=== 测试GD优化器 ===")
        results = []
        
        for func_name, func_config in TEST_FUNCTIONS.items():
            # print(f'funcname={func_name}')
            result = self.run_optimizer_test('GD', OPTIMIZER_CONFIGS['GD'], func_name, func_config)
            results.append(result)
        
        # 打印表格形式的结果汇总
        self._print_optimizer_results_table('GD', results)
        
        # 验证至少有一个测试成功
        self.assertTrue(any(r['success'] for r in results), "GD优化器所有测试都失败")
        
    def _get_str_display_length(self, s):
        """计算字符串的显示宽度，考虑中文字符宽度为英文的两倍"""
        length = 0
        for char in s:
            # 中文字符的Unicode范围
            if '\u4e00' <= char <= '\u9fff':
                length += 2
            else:
                length += 1
        return length
    
    def _pad_string_for_display(self, s, target_width):
        """根据目标显示宽度填充字符串，考虑中文字符宽度"""
        current_width = self._get_str_display_length(s)
        padding_needed = target_width - current_width
        if padding_needed > 0:
            return s + ' ' * padding_needed
        return s
    
    def _print_optimizer_results_table(self, optimizer_name, results):
        """以表格形式打印优化器在所有测试函数上的结果，考虑中文字符宽度"""
        # 定义固定列宽（以显示宽度为单位）
        col_widths = [
            15,  # 测试函数名称
            8,   # 状态
            20,  # 最终损失值
            35,  # 最终参数 (x, y)
            12   # 耗时 (秒)
        ]
        
        # 打印标题和分隔线
        print("\n" + "-" * sum(col_widths) + "-" * 4)  # +4 是因为有4个空格分隔符
        print(f"{optimizer_name}优化器在所有测试函数上的性能结果:")
        print("-" * sum(col_widths) + "-" * 4)
        
        # 打印表头
        headers = ["测试函数", "状态", "最终损失值", "最终参数 (x, y)", "耗时 (秒)"]
        header_line = ""
        for i, header in enumerate(headers):
            header_line += self._pad_string_for_display(header, col_widths[i])
            if i < len(headers) - 1:
                header_line += " "  # 列之间的分隔符
        print(header_line)
        print("-" * sum(col_widths) + "-" * 4)
        
        # 打印每行数据
        for result in results:
            func_name = result['function']
            status = "成功" if result['success'] else "失败"
            loss = result['final_loss']
            x, y = result['final_params']
            time_taken = result['time_taken']
            
            # 格式化输出
            if np.isfinite(loss) and np.isfinite(x) and np.isfinite(y):
                loss_str = f"{loss:.10f}"
                params_str = f"({x:.8f}, {y:.8f})"
            else:
                loss_str = "NaN"
                params_str = "(NaN, NaN)"
            
            # 构建行
            row_parts = [
                self._pad_string_for_display(func_name, col_widths[0]),
                self._pad_string_for_display(status, col_widths[1]),
                self._pad_string_for_display(loss_str, col_widths[2]),
                self._pad_string_for_display(params_str, col_widths[3]),
                self._pad_string_for_display(f"{time_taken:.4f}", col_widths[4])
            ]
            row = " ".join(row_parts)
            print(row)
        
        print("-" * sum(col_widths) + "-" * 4)
        # 添加一个空行作为测试用例之间的分隔
        print()
    
    def test_optimizer_sgd(self):
        """测试SGD优化器在所有测试函数上的性能"""
        print(f"\n=== 测试SGD优化器 ===")
        results = []
        
        for func_name, func_config in TEST_FUNCTIONS.items():
            result = self.run_optimizer_test('SGD', OPTIMIZER_CONFIGS['SGD'], func_name, func_config)
            results.append(result)
        
        # 打印表格形式的结果汇总
        self._print_optimizer_results_table('SGD', results)
        
        # 验证至少有一个测试成功
        self.assertTrue(any(r['success'] for r in results), "SGD优化器所有测试都失败")
    
    def test_optimizer_adam(self):
        """测试Adam优化器在所有测试函数上的性能"""
        print(f"\n=== 测试Adam优化器 ===")
        results = []
        
        for func_name, func_config in TEST_FUNCTIONS.items():
            result = self.run_optimizer_test('Adam', OPTIMIZER_CONFIGS['Adam'], func_name, func_config)
            results.append(result)
        
        # 打印表格形式的结果汇总
        self._print_optimizer_results_table('Adam', results)
        
        # 验证至少有一个测试成功
        self.assertTrue(any(r['success'] for r in results), "Adam优化器所有测试都失败")
    
    def test_optimizer_adagrad(self):
        """测试Adagrad优化器在所有测试函数上的性能"""
        print(f"\n=== 测试Adagrad优化器 ===")
        results = []
        
        for func_name, func_config in TEST_FUNCTIONS.items():
            result = self.run_optimizer_test('Adagrad', OPTIMIZER_CONFIGS['Adagrad'], func_name, func_config)
            results.append(result)
        
        # 打印表格形式的结果汇总
        self._print_optimizer_results_table('Adagrad', results)
        
        # 验证至少有一个测试成功
        self.assertTrue(any(r['success'] for r in results), "Adagrad优化器所有测试都失败")
    
    def test_optimizer_lbfgs(self):
        """测试LBFGS优化器在所有测试函数上的性能"""
        print(f"\n=== 测试LBFGS优化器 ===")
        results = []
        
        for func_name, func_config in TEST_FUNCTIONS.items():
            result = self.run_optimizer_test('LBFGS', OPTIMIZER_CONFIGS['LBFGS'], func_name, func_config)
            results.append(result)
        
        # 打印表格形式的结果汇总
        self._print_optimizer_results_table('LBFGS', results)
        
        # 验证至少有一个测试成功
        self.assertTrue(any(r['success'] for r in results), "LBFGS优化器所有测试都失败")

# 独立脚本运行功能
def run_all_tests():
    """运行所有测试并显示综合结果"""
    clear_screen()
    print("=" * 80)
    print("Riemann优化器性能测试")
    print("=" * 80)
    
    # 打印测试函数信息
    print("\n测试函数信息:")
    for i, (name, config) in enumerate(TEST_FUNCTIONS.items(), 1):
        print(f"{i}. {name}: {config['description']}")
    print("\n")

    # 创建测试用例
    suite = unittest.TestSuite()
    suite.addTest(TestOptimizers('test_optimizer_gd'))
    suite.addTest(TestOptimizers('test_optimizer_sgd'))
    suite.addTest(TestOptimizers('test_optimizer_adam'))
    suite.addTest(TestOptimizers('test_optimizer_adagrad'))
    suite.addTest(TestOptimizers('test_optimizer_lbfgs'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # rm.set_default_dtype(rm.float64)
    success = run_all_tests()
    sys.exit(0 if success else 1)