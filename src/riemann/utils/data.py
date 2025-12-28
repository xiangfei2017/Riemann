# BSD 3-Clause License
#
# Copyright (c) 2025, Fei Xiang
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Riemann Library Data Utilities Module: Dataset and Data Loading

This module provides data loading and dataset management utilities for the Riemann
machine learning framework. It implements abstract dataset interfaces and concrete
implementations for handling tensor data efficiently, supporting both single-process
and multi-process data loading scenarios.

Main features:
    - Dataset abstract base class: Defines the standard interface for all datasets
    - TensorDataset: Simple dataset implementation for tensor data storage
    - DataLoader: Efficient data loading with support for batching, shuffling,
      and multi-process loading
    - Multi-process data loading: Parallel data preprocessing and loading
      for improved performance
    - Memory-efficient iteration: Lazy loading of data to minimize memory usage
    - Flexible sampling strategies: Support for various data sampling methods

Using this module, you can efficiently load and preprocess data for training
machine learning models, with support for large-scale datasets and parallel
processing capabilities.
"""

import numpy as np
from ..autograd import *
import multiprocessing
from queue import Empty
import atexit,sys
import platform

class Dataset:
    """
    抽象数据集基类，定义了所有数据集必须实现的标准接口。
    
    Dataset 类是 Riemann 框架中数据加载的核心抽象，所有具体的数据集实现
    都应该继承此类并实现其抽象方法。它提供了与 PyTorch Dataset 类似的接口，
    使得数据加载过程可以与 DataLoader 无缝配合。
    
    使用方法:
        class MyDataset(Dataset):
            def __len__(self):
                return dataset_size
                
            def __getitem__(self, idx):
                return data_item_at_index(idx)
    """
    
    def __len__(self):
        """
        返回数据集中的样本数量。
        
        这是一个抽象方法，子类必须实现此方法以返回数据集的大小。
        DataLoader 使用此方法来确定迭代次数和批次索引范围。
        
        返回:
            int: 数据集中的样本总数
            
        异常:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError
        
    def __getitem__(self, index):
        """
        根据给定索引获取数据集中的单个样本。
        
        这是一个抽象方法，子类必须实现此方法以返回指定索引处的样本。
        返回的样本可以是单个张量、张量元组、字典或其他数据结构，
        具体取决于数据集的设计和 collate_fn 的处理能力。
        
        参数:
            index (int): 要获取的样本索引，范围应在 [0, len(dataset)) 内
            
        返回:
            样本数据，可以是张量、张量元组、字典或其他数据结构
            
        异常:
            NotImplementedError: 如果子类未实现此方法
            IndexError: 如果索引超出有效范围
        """
        raise NotImplementedError

class TensorDataset(Dataset):
    """
    简单的张量数据集实现，将多个张量的第一个维度作为数据集维度。
    
    TensorDataset 是 Dataset 的一个便捷实现，适用于已经加载到内存中的
    张量数据。它将多个张量的第一个维度（通常是样本维度）对齐，
    每次索引返回所有张量在相同索引处的数据组成的元组。
    
    使用场景:
        - 已加载到内存的小型数据集
        - 特征和标签分别存储在不同张量中的监督学习数据
        - 快速原型开发和实验
    
    示例:
        features = rm.randn(1000, 10)  # 1000个样本，每个10个特征
        labels = rm.randint(0, 2, (1000,))  # 1000个标签
        dataset = TensorDataset(features, labels)
        feature, label = dataset[0]  # 获取第一个样本的特征和标签
    """
    
    def __init__(self, *tensors):
        """
        初始化张量数据集。
        
        创建一个 TensorDataset，将所有输入张量的第一个维度作为数据集维度。
        所有张量的第一个维度大小必须相同，以确保索引的一致性。
        
        参数:
            *tensors (TN): 可变数量的张量，所有张量的第一个维度大小必须相同
            
        异常:
            AssertionError: 如果任何张量的第一个维度大小与第一个张量不同
        """
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors), "所有张量的第一个维度大小必须相同"
        self.tensors = tensors

    def __len__(self):
        """
        返回数据集的大小，即张量的第一个维度大小。
        
        返回:
            int: 数据集中的样本数量，等于张量的第一个维度大小
        """
        return self.tensors[0].shape[0]

    def __getitem__(self, idx):
        """
        获取指定索引处的样本数据。
        
        返回所有张量在指定索引处的数据组成的元组，元组的顺序与
        初始化时传入的张量顺序相同。
        
        参数:
            idx (int): 要获取的样本索引
            
        返回:
            tuple: 包含所有张量在指定索引处数据的元组
            
        异常:
            IndexError: 如果索引超出有效范围 [0, len(dataset))
        """
        return tuple(t[idx] for t in self.tensors)


def default_collate(batch):
    """
    默认的批次处理函数，将一批样本数据转换为适合模型输入的张量格式。
    仿照 PyTorch 的 default_collate 设计，支持标量、NumPy 数组、列表、元组、字典以及自定义 TN 类型的递归处理。
    
    参数:
        batch (list): 一个批次的样本列表，每个样本可以是各种数据类型。
        
    返回:
        根据输入类型组合成的批次数据。标量、NumPy 数组、TN 张量会被堆叠，列表、元组、字典会递归处理。
        
    异常:
        TypeError: 当遇到无法处理的类型时抛出。
    """
    # 检查批次是否为空
    if not batch:
        raise ValueError("Batch cannot be empty")
    
    elem = batch[0]
    elem_type = type(elem)
    
    # 1. 处理标量 (int, float)
    if isinstance(elem, (int, float)):
        # 在 PyTorch 中，标量会被转换为 torch.Tensor。
        # 这里为了保持一致性，我们将标量转换为 TN 张量。
        # 注意：这与原代码（返回 np.array）不同，更贴近 PyTorch 行为。
        return tensor(batch)  # 修改点1：标量转为 TN 张量而非 NumPy 数组
    
    # 2. 处理 NumPy 数组
    elif isinstance(elem, np.ndarray):
        # 尝试堆叠 NumPy 数组。如果数组形状不一致，np.stack 会抛出异常。
        try:
            stacked = np.stack(batch, axis=0)
            return tensor(stacked)  # 修改点2：NumPy 数组也转为 TN 张量
        except ValueError:
            # 如果堆叠失败（例如变长序列），fallback 到列表或自定义处理
            # 在实际应用中，你可能需要更复杂的逻辑，如填充（padding）
            return batch  # 或者抛出异常，取决于你的需求
    
    # 3. 处理自定义 TN 张量
    elif isinstance(elem, TN):
        # rm.stack 应类似于 np.stack 或 torch.stack，在第0维（批次维）堆叠张量。
        return stack(batch, dim=0)
            
    # 4. 处理序列 (列表 list 或元组 tuple)
    elif isinstance(elem, (list, tuple)):
        # 递归处理序列中的每个元素。
        # 使用 zip(*batch) 进行转置，将不同样本的相同位置元素分组。
        # 例如：batch = [(a1, b1), (a2, b2)] -> transposed = [(a1, a2), (b1, b2)]
        transposed = zip(*batch)
        # 对每一组元素递归调用 default_collate
        return elem_type([default_collate(samples) for samples in transposed])  # 修改点4：使用递归处理嵌套结构
    
    # 5. 处理字典 (dict)
    elif isinstance(elem, dict):
        # 确保批次中所有字典的键相同
        if not all(set(elem.keys()) == set(b.keys()) for b in batch):
            raise ValueError("All dictionaries in the batch must have the same keys")
        # 对字典中的每个键对应的值列表递归调用 default_collate
        return {key: default_collate([d[key] for d in batch]) for key in elem.keys()}
    
    # 6. 处理其他不支持的类型（如字符串 str）
    else:
        # 原代码直接返回 batch（列表），但 PyTorch 对无法处理的类型会抛出 TypeError。
        # 修改点5：明确抛出异常，提醒用户需要自定义 collate_fn
        raise TypeError(f"default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {elem_type}")

class DataLoader:
    """
    高效的数据加载器，支持批次处理、数据洗牌和多进程加载。
    
    DataLoader 是 Riemann 框架中数据加载的核心组件，提供了与 PyTorch DataLoader
    类似的接口和功能。它支持单进程和多进程数据加载，可以自动处理批次形成、
    数据洗牌和最后不完整批次的处理，是训练机器学习模型的重要工具。
    
    主要特性:
        - 批次处理：自动将单个样本组合成批次
        - 数据洗牌：支持每个 epoch 开始时的数据随机化
        - 多进程加载：使用多个工作进程并行加载数据，提高 I/O 密集型任务的效率
        - 自定义批次处理：支持自定义 collate_fn 处理复杂的数据结构
        - 内存管理：自动处理工作进程的生命周期和资源清理
    
    使用示例:
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        for batch_features, batch_labels in loader:
            # 训练代码
            pass
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False, 
                 num_workers=0, collate_fn=None, drop_last=False):
        """
        初始化数据加载器。
        
        创建一个 DataLoader 实例，配置数据加载的各种参数。支持单进程和多进程
        数据加载模式，多进程模式下会创建工作进程来并行处理数据加载任务。
        
        参数:
            dataset (Dataset): 要加载的数据集
            batch_size (int, optional): 每个批次的大小，默认为1
            shuffle (bool, optional): 是否在每个 epoch 开始时洗牌数据，默认为False
            num_workers (int, optional): 数据加载的工作进程数，0表示主进程加载，默认为0
            collate_fn (callable, optional): 批次处理函数，用于将样本列表组合成批次，
                                         默认使用 default_collate
            drop_last (bool, optional): 如果数据集大小不能被批次大小整除，是否丢弃
                                     最后一个不完整的批次，默认为False
                                     
        属性:
            dataset (Dataset): 要加载的数据集
            batch_size (int): 批次大小
            shuffle (bool): 是否洗牌数据
            num_workers (int): 工作进程数
            collate_fn (callable): 批次处理函数
            drop_last (bool): 是否丢弃最后不完整批次
            workers (list): 工作进程列表
            sample_iter: 样本索引迭代器
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn or default_collate  # 使用模块级函数而非实例方法
        self.drop_last = drop_last
        
        # 多进程相关属性
        self.workers = []
        self.sample_iter = None
        
        # 注册清理函数，确保进程正确退出
        atexit.register(self._shutdown_workers)
        
        # 设置多进程启动方法，避免在多线程环境中的fork死锁警告
        # 在Linux/macOS上使用spawn模式，Windows上保持默认
        if platform.system() != 'Windows' and num_workers > 0:
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                # 如果start_method已经设置，忽略错误
                pass

    # DataLoader设计为可迭代对象，需要添加__len__方法
    def __len__(self):
        """
        返回数据加载器的批次数目。
        
        根据 drop_last 参数的设置，计算数据加载器在一个 epoch 中会产生的
        批次数量。如果 drop_last 为 True，则只包含完整的批次；否则包含
        所有批次，包括可能不完整的最后一个批次。
        
        返回:
            int: 一个 epoch 中的批次数量
        """
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    # 全局打乱索引的索引迭代器，len(self.dataset)特别大时，内存消耗的
    def _get_indices(self):
        """
        获取数据集的索引迭代器，支持洗牌和 drop_last 处理。
        
        创建一个包含数据集索引的迭代器，根据 shuffle 参数决定是否打乱顺序，
        根据 drop_last 参数决定是否截断最后不完整的批次。
        
        返回:
            iterator: 数据集索引的迭代器
        """
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        
        # 处理最后不完整批次
        if self.drop_last:
            indices = indices[:len(indices)//self.batch_size*self.batch_size]
        return iter(indices)

    def __iter__(self):
        """
        返回数据加载器的迭代器。
        
        每次调用此方法都会创建一个新的迭代器，根据 num_workers 参数
        决定使用单进程还是多进程模式。在多进程模式下，会创建工作进程
        并设置任务队列和结果队列。
        
        返回:
            iterator: 数据批次迭代器
        """
        # 每次迭代时重新获取索引迭代器
        self.sample_iter = self._get_indices()
        
        if self.num_workers == 0:
            return self._single_process_iter()
        else:
            return self._multi_process_iter()

    def _single_process_iter(self):
        """
        单进程数据加载迭代器。
        
        在主进程中直接加载数据并形成批次，适用于小数据集或调试场景。
        这种方式简单直接，但可能成为训练过程的瓶颈，特别是在数据加载
        需要大量计算或 I/O 操作时。
        
        生成:
            batch: 经过 collate_fn 处理的数据批次
        """
        batch = []
        for idx in self.sample_iter:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield self.collate_fn([self.dataset[i] for i in batch])
                batch = []
        
        # 处理最后不完整的批次
        if not self.drop_last and len(batch) > 0:
            yield self.collate_fn([self.dataset[i] for i in batch])

    def _multi_process_iter(self):
        """
        多进程数据加载迭代器。
        
        创建多个工作进程并行加载数据，使用任务队列分发索引批次，
        使用结果队列收集处理后的数据。这种方法适用于大数据集或
        数据加载需要大量计算的场景，可以显著提高数据加载效率。
        
        生成:
            batch: 经过 collate_fn 处理的数据批次
        """
        # 创建进程安全的任务队列和结果队列
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue(maxsize=self.num_workers*2)
        
        # 为每个worker分配任务
        batches = []
        batch = []
        for idx in self.sample_iter:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        
        if not self.drop_last and len(batch) > 0:
            batches.append(batch)
        
        # 将批次任务放入队列
        for batch_indices in batches:
            self.task_queue.put(batch_indices)
        
        # 添加结束标记
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        # 创建工作进程
        self.workers = [
            multiprocessing.Process(
                target=self._worker_loop,
                args=(self.task_queue, self.result_queue, self.dataset, self.collate_fn)
            ) for _ in range(self.num_workers)
        ]
        
        for w in self.workers:
            w.daemon = True  # 设置守护进程，主进程退出时自动终止
            w.start()
        
        # 从结果队列获取数据
        completed_workers = 0
        while completed_workers < self.num_workers:
            try:
                result = self.result_queue.get(timeout=30)
                if result is None:  # 工作进程结束信号
                    completed_workers += 1
                else:
                    yield result
            except Empty:
                # 检查工作进程是否都还存活
                if all(not w.is_alive() for w in self.workers):
                    break
        
        self._shutdown_workers()

    @staticmethod
    def _worker_loop(task_queue, result_queue, dataset, collate_fn):
        """
        工作进程的主循环函数。
        
        此函数在每个工作进程中运行，从任务队列获取索引批次，
        从数据集加载对应的数据，使用 collate_fn 处理后放入结果队列。
        当收到 None 任务时，表示工作结束，发送结束信号并退出。
        
        参数:
            task_queue (Queue): 任务队列，包含索引批次或 None 结束信号
            result_queue (Queue): 结果队列，用于存放处理后的批次数据
            dataset (Dataset): 要加载的数据集
            collate_fn (callable): 批次处理函数
        """
        try:
            while True:
                batch_indices = task_queue.get()
                if batch_indices is None:  # 结束信号
                    result_queue.put(None)  # 通知主进程本worker结束
                    break
                
                try:
                    # 加载数据并处理
                    batch_data = [dataset[i] for i in batch_indices]
                    processed_batch = collate_fn(batch_data)
                    result_queue.put(processed_batch)
                except Exception as e:
                    # 捕获处理异常，避免工作进程崩溃
                    print(f"Worker error: {e}", file=sys.stderr)
                    result_queue.put(None)  # 发送空结果表示错误
        finally:
            result_queue.put(None)  # 确保最终发送结束信号

    def _shutdown_workers(self):
        """
        安全关闭所有工作进程。
        
        终止所有仍在运行的工作进程，等待它们结束，并关闭相关队列。
        此方法会在析构函数和程序退出时被调用，确保资源被正确释放。
        """
        if hasattr(self, 'workers'):
            for w in self.workers:
                if w.is_alive():
                    w.terminate()  # 强制终止可能卡住的进程
                if hasattr(w, '_started') and w._started: # 检查进程是否已启动
                    w.join(timeout=1.0)  # 等待进程结束，设置超时

        # 关闭队列 (如果使用了多进程队列，且队列是在类中创建的)
        if hasattr(self, 'task_queue'):
            self.task_queue.close()
        if hasattr(self, 'result_queue'):
            self.result_queue.close()

    def __del__(self):
        """
        析构函数，确保资源被正确清理。
        
        当 DataLoader 实例被垃圾回收时，自动调用 _shutdown_workers
        方法清理工作进程和队列，防止资源泄漏。
        """
        self._shutdown_workers()
        