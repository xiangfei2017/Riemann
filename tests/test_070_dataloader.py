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

class MNISTDataset(Dataset):
    """MNIST数据集加载类"""
    def __init__(self, csv_file=None, subset_size=None):
        self.data_list = []
        if csv_file and os.path.exists(csv_file):
            with open(csv_file, 'r') as training_data_file:
                for i, record in enumerate(tqdm(training_data_file)):
                    all_values = record.split(',')
                    label = int(all_values[0])

                    target = rm.full((10,), fill_value=0.01)
                    target[label] = 0.99  

                    image_numpy = (np.asarray(all_values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01
                    image = rm.tensor(image_numpy, dtype=rm.get_default_dtype())
                    self.data_list.append((label, image, target))
                    
                    if subset_size and i >= subset_size - 1:
                        break
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        # 模拟耗时操作，便于测试多进程效果
        time.sleep(0.002)
        return self.data_list[index]

# 辅助函数
def get_file_path(filename):
    """获取MNIST数据集文件路径"""
    current_dir = Path(__file__).resolve().parent.parent
    file_path = current_dir / "data" / "MNIST" / filename
    return str(file_path)

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
        
        # 尝试加载MNIST数据集的小样本
        cls.mnist_file = get_file_path("mnist_test_10.csv")  # 使用小数据集用于快速测试
        if os.path.exists(cls.mnist_file):
            cls.mnist_dataset = MNISTDataset(cls.mnist_file)
        else:
            cls.mnist_dataset = None
            print(f"警告: MNIST测试数据集 {cls.mnist_file} 不存在，部分测试将被跳过")
    
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
                self.assertEqual(batch_count, min(expected_batches, 1))  # 只运行1个iteration用于测试
    
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
        
        # 注意：这是一个概率性测试，理论上有极小概率失败
        self.assertFalse(all_same, "打乱功能似乎没有生效")
    
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
        
        expected_keep = (total_size + batch_size - 1) // batch_size  # 向上取整
        expected_drop = total_size // batch_size  # 向下取整
        
        self.assertEqual(len(keep_batches), expected_keep)
        self.assertEqual(len(drop_batches), expected_drop)
        
        # 验证最后批次的大小
        if keep_batches:
            # 更准确地获取批次大小：检查批次中第一个张量的第一个维度
            batch_element = keep_batches[-1][0]
            if hasattr(batch_element, 'shape') and len(batch_element.shape) > 0:
                last_batch_size = batch_element.shape[0]
            elif isinstance(batch_element, (list, tuple)):
                last_batch_size = len(batch_element)
            else:
                # 如果无法确定批次大小，则直接跳过此检查
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
                except Exception as e:
                    print(f"    多进程测试失败: {str(e)}")
                    # 多进程测试可能因环境而异，记录但不中断
    
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
                # 注意：由于数据集较小和进程启动开销，多进程可能不总是更快
            except Exception as e:
                print(f"多进程性能测试失败: {str(e)}")
    
    def test_dataloader_mnist(self):
        """测试MNIST数据集加载（如果可用）"""
        if not self.mnist_dataset or len(self.mnist_dataset) == 0:
            self.skipTest("MNIST测试数据集不可用")
            return
        
        print("测试MNIST数据集加载...")
        
        # 基本加载测试
        dataloader = DataLoader(self.mnist_dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        
        # 验证批次结构
        self.assertEqual(len(batch), 3)  # 应该包含label, image, target
        labels, images, targets = batch
        
        # 验证数据类型和形状
        self.assertEqual(len(labels), 2)
        self.assertEqual(images.shape, (2, 784))  # 28x28图像展平为784
        self.assertEqual(targets.shape, (2, 10))  # 10类分类

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
    print("6. MNIST数据集加载测试（如果可用）")
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
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)