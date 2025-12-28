#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试优化后的Optimizer类功能

测试内容:
1. 基础SGD优化器功能
2. Dampening和Nesterov支持
3. Closure参数支持
4. 状态管理和序列化
5. 错误处理和边界检查
6. 类型提示验证
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from riemann import tensor, Parameter
from riemann.optim import SGD, Adam
from riemann.nn import Module
import unittest


class SimpleModel(Module):
    """简单的线性模型用于测试"""
    
    def __init__(self):
        super().__init__()
        self.weight = Parameter(tensor(np.random.randn(10, 5).astype(np.float32) * 0.1))
        self.bias = Parameter(tensor(np.zeros(5).astype(np.float32)))
    
    def forward(self, x):
        # 使用矩阵乘法，需要建立计算图
        from riemann import matmul
        return matmul(x, self.weight) + self.bias


class TestOptimizedOptimizer(unittest.TestCase):
    """测试优化后的Optimizer类"""
    
    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)
        self.model = SimpleModel()
        self.x = tensor(np.random.randn(3, 10).astype(np.float32))
        self.y = tensor(np.random.randn(3, 5).astype(np.float32))
    
    def test_sgd_basic_functionality(self):
        """测试SGD基础功能"""
        # 创建SGD优化器
        optimizer = SGD(self.model.parameters(), lr=0.01)
        
        # 检查参数组设置
        self.assertEqual(len(optimizer.param_groups), 1)
        group = optimizer.param_groups[0]
        self.assertEqual(group['lr'], 0.01)
        self.assertEqual(group['momentum'], 0.0)
        self.assertEqual(group['weight_decay'], 0.0)
        self.assertEqual(group['dampening'], 0.0)
        self.assertEqual(group['nesterov'], False)
        
        # 检查状态初始化
        self.assertIsInstance(optimizer.state, dict)
    
    def test_sgd_momentum_dampening(self):
        """测试SGD动量和抑制功能"""
        # 测试带动量的SGD
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9, dampening=0.1)
        
        group = optimizer.param_groups[0]
        self.assertEqual(group['momentum'], 0.9)
        self.assertEqual(group['dampening'], 0.1)
        
        # 执行一步优化
        output = self.model(self.x)
        diff = output - self.y
        loss = (diff * diff).mean()
        loss.backward()
        
        optimizer.step()
        
        # 检查动量状态是否正确初始化
        for param in self.model.parameters():
            param_id = id(param)
            if param_id in optimizer.state:
                self.assertIn('velocity', optimizer.state[param_id])
                self.assertEqual(optimizer.state[param_id]['velocity'].shape, param.data.shape)
    
    def test_sgd_nesterov(self):
        """测试SGD Nesterov动量"""
        # 测试Nesterov动量
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        
        group = optimizer.param_groups[0]
        self.assertTrue(group['nesterov'])
        
        # 执行一步优化
        output = self.model(self.x)
        diff = output - self.y
        loss = (diff * diff).mean()
        loss.backward()
        
        initial_weight = self.model.weight.data.copy()
        optimizer.step()
        
        # 检查参数是否更新
        self.assertFalse(np.allclose(initial_weight, self.model.weight.data))
    
    def test_closure_support(self):
        """测试closure参数支持"""
        optimizer = SGD(self.model.parameters(), lr=0.01)
        
        def closure():
            output = self.model(self.x)
            diff = output - self.y
            loss = (diff * diff).mean()
            loss.backward()
            return loss.data.item()
        
        # 测试带closure的step
        loss_value = optimizer.step(closure)
        self.assertIsNotNone(loss_value)
        self.assertIsInstance(loss_value, float)
    
    def test_state_dict_and_load_state_dict(self):
        """测试状态字典的保存和加载"""
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        # 执行几步优化以产生状态
        for _ in range(3):
            output = self.model(self.x)
            diff = output - self.y
            loss = (diff * diff).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 保存状态
        state_dict = optimizer.state_dict()
        
        # 检查状态字典结构
        self.assertIn('state', state_dict)
        self.assertIn('param_groups', state_dict)
        
        # 创建新的优化器并加载状态
        new_optimizer = SGD(self.model.parameters(), lr=0.001)  # 不同的学习率
        new_optimizer.load_state_dict(state_dict)
        
        # 检查加载后的状态
        self.assertEqual(len(new_optimizer.param_groups), len(optimizer.param_groups))
        self.assertEqual(new_optimizer.param_groups[0]['lr'], 0.01)  # 应该被覆盖
    
    def test_zero_grad_functionality(self):
        """测试梯度清零功能"""
        optimizer = SGD(self.model.parameters(), lr=0.01)
        
        # 计算梯度
        output = self.model(self.x)
        diff = output - self.y
        loss = (diff * diff).mean()
        loss.backward()
        
        # 检查梯度存在
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 检查梯度被清零
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertTrue(np.allclose(param.grad.data, 0))
        
        # 测试set_to_none=True
        output = self.model(self.x)
        diff = output - self.y
        loss = (diff * diff).mean()
        loss.backward()
        
        optimizer.zero_grad(set_to_none=True)
        
        for param in self.model.parameters():
            self.assertIsNone(param.grad)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试负学习率
        with self.assertRaises(ValueError):
            SGD(self.model.parameters(), lr=-0.01)
        
        # 测试负动量
        with self.assertRaises(ValueError):
            SGD(self.model.parameters(), lr=0.01, momentum=-0.1)
        
        # 测试负权重衰减
        with self.assertRaises(ValueError):
            SGD(self.model.parameters(), lr=0.01, weight_decay=-0.01)
        
        # 测试负抑制系数
        with self.assertRaises(ValueError):
            SGD(self.model.parameters(), lr=0.01, dampening=-0.01)
        
        # 测试无效的Nesterov配置
        with self.assertRaises(ValueError):
            SGD(self.model.parameters(), lr=0.01, nesterov=True)  # 没有动量
        
        with self.assertRaises(ValueError):
            SGD(self.model.parameters(), lr=0.01, momentum=0.9, dampening=0.1, nesterov=True)  # 有抑制
    
    def test_empty_parameter_list(self):
        """测试空参数列表检查"""
        with self.assertRaises(ValueError):
            SGD([], lr=0.01)
    
    def test_parameter_group_management(self):
        """测试参数组管理"""
        # 创建多个参数组
        param_groups = [
            {'params': [self.model.weight], 'lr': 0.01},
            {'params': [self.model.bias], 'lr': 0.001}
        ]
        
        optimizer = SGD(param_groups)
        
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        self.assertEqual(optimizer.param_groups[1]['lr'], 0.001)
    
    def test_add_param_group(self):
        """测试添加参数组"""
        optimizer = SGD([self.model.weight], lr=0.01)
        
        # 添加新的参数组
        optimizer.add_param_group({'params': [self.model.bias], 'lr': 0.001})
        
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertIs(optimizer.param_groups[1]['params'][0], self.model.bias)
    
    def test_repr_functionality(self):
        """测试字符串表示"""
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        repr_str = repr(optimizer)
        
        self.assertIn('SGD', repr_str)
        self.assertIn('Parameter Group', repr_str)
        self.assertIn('lr: 0.01', repr_str)
        self.assertIn('momentum: 0.9', repr_str)
    
    def test_gradient_shape_validation(self):
        """测试梯度形状验证"""
        optimizer = SGD(self.model.parameters(), lr=0.01)
        
        # 手动设置错误形状的梯度
        self.model.weight.grad = tensor(np.random.randn(5, 10).astype(np.float32))  # 错误形状
        
        with self.assertRaises(ValueError):
            optimizer.step()


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)