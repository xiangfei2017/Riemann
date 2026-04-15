import sys
import os
import time
import numpy as np
import unittest
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import riemann as rm
from riemann.utils.data import Dataset, DataLoader
from riemann.vision.datasets import MNIST

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

# 用于测试的数据集类
class SimpleTestDataset(Dataset):
    """简单的测试数据集"""
    def __init__(self, size=1000, transform=None):
        self.data = [(i, rm.tensor([i, i*2]), rm.tensor(i*3)) for i in range(size)]
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # 模拟耗时操作，便于测试多进程效果
        time.sleep(0.001)
        
        item = self.data[index]
        if self.transform:
            item = (item[0], self.transform(item[1]), item[2])
        return item

# 辅助函数
def get_data_path():
    """获取数据目录路径"""
    current_dir = Path(__file__).resolve().parent.parent
    return str(current_dir / "data")

def clear_screen():
    """跨平台清屏函数"""
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

# 测试类
class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """类级别的设置，只在测试类运行前执行一次"""
        # 创建简单测试数据集
        cls.simple_dataset = SimpleTestDataset(size=100)
        
        # 尝试加载MNIST测试数据集
        cls.mnist_dataset = None
        try:
            data_root = get_data_path()
            # 使用MNIST测试集
            cls.mnist_dataset = MNIST(root=data_root, train=False, download=True)
            print(f"MNIST测试集加载成功，共 {len(cls.mnist_dataset)} 个样本")
        except Exception as e:
            print(f"警告: MNIST测试数据集加载失败: {e}，部分测试将被跳过")
    
    def setUp(self):
        """每个测试用例开始前执行"""
        # 获取当前测试方法名
        self.current_test_name = self._testMethodName
        stats.start_function(self.current_test_name)
    
    def tearDown(self):
        """每个测试用例结束后执行"""
        stats.end_function()
    
    def _test_dataloader_params(self, dataset, batch_size=32, shuffle=False, num_workers=0, 
                               drop_last=False, max_iterations=1):
        """通用的DataLoader测试函数"""
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last
        )
        
        # 验证数据加载器的基本功能
        start_time = time.time()
        batches = []
        
        for i, batch in enumerate(dataloader):
            self.assertTrue(isinstance(batch, tuple) or isinstance(batch, list))
            batches.append(batch)
            if i >= max_iterations - 1:
                break
        
        load_time = time.time() - start_time
        return len(batches), load_time
    
    def test_dataloader_single_process(self):
        """测试单进程数据加载"""
        print("测试单进程数据加载...")
        
        # 测试不同的batch_size
        for batch_size in [1, 10, 32]:
            with self.subTest(batch_size=batch_size):
                batch_count, _ = self._test_dataloader_params(
                    self.simple_dataset,
                    batch_size=batch_size,
                    num_workers=0
                )
                expected_batches = len(self.simple_dataset) // batch_size
                self.assertEqual(batch_count, min(expected_batches, 1))
                stats.add_result(f"batch_size={batch_size}", True)
    
    def test_dataloader_shuffle(self):
        """测试数据打乱功能"""
        print("测试数据打乱功能...")
        
        # 不打乱顺序
        dataloader_no_shuffle = DataLoader(self.simple_dataset, batch_size=10, shuffle=False, num_workers=0)
        first_batch_no_shuffle = next(iter(dataloader_no_shuffle))
        
        # 打乱顺序
        dataloader_shuffle = DataLoader(self.simple_dataset, batch_size=10, shuffle=True, num_workers=0)
        first_batch_shuffle = next(iter(dataloader_shuffle))
        
        # 至少有一个元素应该不同（概率上几乎总是成立）
        all_same = True
        for no_shuffle, shuffle in zip(first_batch_no_shuffle, first_batch_shuffle):
            if isinstance(no_shuffle, rm.TN) and isinstance(shuffle, rm.TN):
                if not (no_shuffle == shuffle).all().item():
                    all_same = False
                    break
            elif no_shuffle != shuffle:
                all_same = False
                break
        
        passed = not all_same
        self.assertFalse(all_same, "打乱功能似乎没有生效")
        stats.add_result("shuffle功能", passed)
    
    def test_dataloader_drop_last(self):
        """测试丢弃最后不完整批次功能"""
        print("测试丢弃最后不完整批次功能...")
        
        total_size = 100
        batch_size = 15
        
        # 不丢弃最后批次
        dataloader_keep = DataLoader(
            self.simple_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        # 丢弃最后批次
        dataloader_drop = DataLoader(
            self.simple_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )
        
        # 计算批次数量
        keep_batches = list(dataloader_keep)
        drop_batches = list(dataloader_drop)
        
        expected_keep = (total_size + batch_size - 1) // batch_size
        expected_drop = total_size // batch_size
        
        self.assertEqual(len(keep_batches), expected_keep)
        self.assertEqual(len(drop_batches), expected_drop)
        
        stats.add_result(f"drop_last=False批次数={len(keep_batches)}", len(keep_batches) == expected_keep)
        stats.add_result(f"drop_last=True批次数={len(drop_batches)}", len(drop_batches) == expected_drop)
        
        # 验证最后批次的大小
        if keep_batches:
            batch_element = keep_batches[-1][0]
            if hasattr(batch_element, 'shape') and len(batch_element.shape) > 0:
                last_batch_size = batch_element.shape[0]
            elif isinstance(batch_element, (list, tuple)):
                last_batch_size = len(batch_element)
            else:
                print("警告: 无法确定批次大小，跳过此检查")
                return
                
            expected_last_size = total_size % batch_size
            self.assertEqual(last_batch_size, expected_last_size if expected_last_size > 0 else batch_size)
    
    def test_dataloader_multi_process(self):
        """测试多进程数据加载"""
        print("测试多进程数据加载...")
        
        # 测试不同数量的worker
        for num_workers in [1, 2, min(4, os.cpu_count() or 2)]:
            with self.subTest(num_workers=num_workers):
                print(f"  测试 {num_workers} 个工作进程...")
                try:
                    batch_count, load_time = self._test_dataloader_params(
                        self.simple_dataset,
                        batch_size=32,
                        num_workers=num_workers,
                        max_iterations=3
                    )
                    print(f"    加载完成，耗时: {load_time:.4f}秒")
                    self.assertEqual(batch_count, 3)
                    stats.add_result(f"num_workers={num_workers}", True)
                except Exception as e:
                    print(f"    多进程测试失败: {str(e)}")
                    stats.add_result(f"num_workers={num_workers}", False, str(e))
    
    def test_dataloader_performance_comparison(self):
        """比较单进程和多进程性能差异"""
        print("比较单进程和多进程性能差异...")
        
        # 只测试3个批次以便快速完成
        _, single_time = self._test_dataloader_params(
            self.simple_dataset,
            batch_size=32,
            num_workers=0,
            max_iterations=3
        )
        
        stats.add_result(f"单进程耗时={single_time:.4f}s", True)
        
        # 测试2个worker
        if os.cpu_count() >= 2:
            try:
                _, multi_time = self._test_dataloader_params(
                    self.simple_dataset,
                    batch_size=32,
                    num_workers=2,
                    max_iterations=3
                )
                print(f"单进程耗时: {single_time:.4f}秒")
                print(f"多进程(2)耗时: {multi_time:.4f}秒")
                stats.add_result(f"多进程(2)耗时={multi_time:.4f}s", True)
            except Exception as e:
                print(f"多进程性能测试失败: {str(e)}")
                stats.add_result("多进程性能测试", False, str(e))
    
    def test_dataloader_mnist(self):
        """测试MNIST数据集加载（使用riemann的MNIST数据集类）"""
        if not self.mnist_dataset or len(self.mnist_dataset) == 0:
            self.skipTest("MNIST测试数据集不可用")
            return
        
        print("测试MNIST数据集加载...")
        
        # 创建带transform的MNIST数据集，将PIL Image转换为tensor
        def pil_to_tensor(pil_img):
            """将PIL Image转换为riemann tensor"""
            import numpy as np
            np_array = np.array(pil_img, dtype=np.float32) / 255.0
            return rm.tensor(np_array).reshape(1, 28, 28)  # (1, 28, 28)
        
        data_root = get_data_path()
        mnist_dataset_transformed = MNIST(
            root=data_root, 
            train=False, 
            download=False,
            transform=pil_to_tensor
        )
        
        # 基本加载测试
        dataloader = DataLoader(mnist_dataset_transformed, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        
        # 验证批次结构
        self.assertEqual(len(batch), 2)  # 应该包含image, label
        images, labels = batch
        
        # 验证数据类型和形状
        self.assertEqual(images.shape, (2, 1, 28, 28))  # 2张图像，1通道，28x28
        self.assertEqual(len(labels), 2)
        
        stats.add_result(f"MNIST批次图像形状={images.shape}", images.shape == (2, 1, 28, 28))
        stats.add_result(f"MNIST批次标签数量={len(labels)}", len(labels) == 2)
        
        # 测试不同batch_size
        for batch_size in [4, 16, 32]:
            with self.subTest(batch_size=batch_size):
                dataloader = DataLoader(
                    mnist_dataset_transformed, 
                    batch_size=batch_size, 
                    shuffle=False, 
                    num_workers=0
                )
                batch = next(iter(dataloader))
                images, labels = batch
                self.assertEqual(images.shape[0], batch_size)
                stats.add_result(f"batch_size={batch_size}", images.shape[0] == batch_size)

# 独立脚本运行功能
def run_all_tests():
    """运行所有测试并显示结果"""
    clear_screen()
    print("=" * 80)
    print("Riemann DataLoader 测试")
    print("=" * 80)
    print("测试内容：")
    print("1. 单进程数据加载测试")
    print("2. 数据打乱功能测试")
    print("3. 丢弃最后批次功能测试")
    print("4. 多进程数据加载测试")
    print("5. 单进程和多进程性能比较")
    print("6. MNIST数据集加载测试（使用riemann的MNIST数据集类）")
    print("\n")
    
    # 创建测试用例
    suite = unittest.TestSuite()
    suite.addTest(TestDataLoader('test_dataloader_single_process'))
    suite.addTest(TestDataLoader('test_dataloader_shuffle'))
    suite.addTest(TestDataLoader('test_dataloader_drop_last'))
    suite.addTest(TestDataLoader('test_dataloader_multi_process'))
    suite.addTest(TestDataLoader('test_dataloader_performance_comparison'))
    suite.addTest(TestDataLoader('test_dataloader_mnist'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
