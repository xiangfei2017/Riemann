"""
演示Riemann的自动微分和反向传播功能

本示例展示了如何使用Riemann的backward()函数进行自动微分和反向传播。
包括标量输出和向量输出的梯度计算，以及如何使用梯度进行参数更新。

主要功能：
1. 标量输出的反向传播
2. 向量输出的反向传播
3. 梯度累积和清除
4. 使用梯度进行简单的参数更新
"""

import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))  # 添加父目录到路径

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

# 导入riemann模块
import riemann as rm
from riemann.autograd import grad

# 打印标题
print('===== Riemann 自动微分库 - backward函数使用演示 =====\n')

# 示例1: 基本的backward使用 - 标量输出
print('1、基本的backward使用 - 标量输出')
print('f(x) = x^2')
print('x = 3.0')

def simple_func(x):
    return x ** 2.

# 创建输入张量并设置requires_grad=True
x = rm.tensor(3.0, requires_grad=True)
print(f'输入x: {x}, requires_grad: {x.requires_grad}')

# 前向传播
y = simple_func(x)
print(f'输出y = f(x): {y}')

# 反向传播 - 对标量输出可以直接调用backward()
y.backward()

# 查看梯度结果
print(f'反向传播后，x的梯度 dx/dy: {x.grad}')
print('说明: 对于f(x)=x²，导数df/dx=2x，当x=3时，梯度值为6\n')

# 示例2: 向量输出的反向传播 - 需要提供gradient参数
print('2、向量输出的反向传播 - 需要提供gradient参数')
print('f(x) = x^2')
print('x = [1.0, 2.0, 3.0]')

# 创建向量输入
ex_x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f'输入ex_x: {ex_x}')

# 前向传播
ex_y = ex_x ** 2.
print(f'输出ex_y: {ex_y}')

# 向量输出需要提供gradient参数（与输出形状相同的张量）
grad_output = rm.tensor([1.0, 1.0, 1.0])
ex_y.backward(gradient=grad_output)

# 查看梯度结果
print(f'反向传播后，ex_x的梯度: {ex_x.grad}')
print('说明: 对于向量输出，需要提供gradient参数来指定每个输出分量的梯度\n')

# 示例3: 复杂计算图的反向传播
print('3、复杂计算图的反向传播')
print('f(a, b, c) = a*b + b*c + c*a')
print('a = 1.0, b = 2.0, c = 3.0')

# 创建多个输入
a = rm.tensor(1.0, requires_grad=True)
b = rm.tensor(2.0, requires_grad=True)
c = rm.tensor(3.0, requires_grad=True)
print(f'输入a: {a}, b: {b}, c: {c}')

# 构建复杂计算图
ab = a * b
bc = b * c
ca = c * a
z = ab + bc + ca
print(f'计算结果z: {z}')

# 反向传播
z.backward()

# 查看各输入的梯度
print(f'dz/da: {a.grad}')  # 应为 b + c = 5
print(f'dz/db: {b.grad}')  # 应为 a + c = 4
print(f'dz/dc: {c.grad}')  # 应为 b + a = 3
print('说明: backward自动计算并累积所有路径的梯度贡献\n')

# 示例4: 使用create_graph参数进行高阶导数计算
print('4、使用create_graph参数进行高阶导数计算')
print('f(x) = x^3')
print('x = 2.0')

# 创建输入
h_x = rm.tensor(2.0, requires_grad=True)
print(f'输入h_x: {h_x}')

# 一阶导数
h_y = h_x ** 3.
print(f'f(x) = {h_y}')
h_y.backward(create_graph=True)
print(f'一阶导数 df/dx = {h_x.grad}')  # 应为 3x² = 12

# 二阶导数（需要清除之前的梯度）
h_x.grad = None  # 清除一阶导数
# 重新计算一阶导数，但这次保存计算图
h_y = h_x ** 3.
grad_output = rm.tensor(1.0)
dy_dx = grad(h_y, h_x, grad_outputs=grad_output, create_graph=True)[0]
print(f'dy_dx = {dy_dx}')

# 对一阶导数再求导（得到二阶导数）
dy_dx.backward()
print(f'二阶导数 d²f/dx² = {h_x.grad}')  # 应为 6x = 12
print('说明: create_graph=True允许对梯度结果再次求导，用于计算高阶导数\n')

# 示例5: 中间节点梯度的保存（使用retains_grad=True）
print('5、中间节点梯度的保存（使用retains_grad=True）')
print('f(x) = sin(x^2)')
print('x = 1.0')

# 创建输入
mid_x = rm.tensor(1.0, requires_grad=True)
print(f'输入mid_x: {mid_x}')

# 创建中间变量并设置retains_grad=True
mid_y = mid_x ** 2.
mid_y.retains_grad = True  # 保存中间变量的梯度
print(f'中间变量mid_y = x²: {mid_y}, retains_grad: {mid_y.retains_grad}')

# 最终输出
mid_z = rm.sin(mid_y)
print(f'最终输出mid_z = sin(y): {mid_z}')

# 反向传播
mid_z.backward()

# 查看梯度
print(f'dz/dx: {mid_x.grad}')  # 应为 2x*cos(x²)
print(f'dz/dy: {mid_y.grad}')  # 应为 cos(y)
print('说明: 通过设置retains_grad=True，可以保存中间节点的梯度值\n')

# 示例6: 梯度的累积与清零
print('6、梯度的累积与清零')
print('多次前向传播和反向传播')

# 创建输入
grad_x = rm.tensor(2.0, requires_grad=True)
print(f'初始输入grad_x: {grad_x}')

# 第一次前向传播和反向传播
y1 = grad_x ** 2.
y1.backward()
print(f'第一次反向传播后，grad_x.grad = {grad_x.grad}')  # 应为 4

# 第二次前向传播和反向传播（注意梯度会累积）
y2 = grad_x ** 3.
y2.backward()
print(f'第二次反向传播后，grad_x.grad = {grad_x.grad}')  # 应为 4 + 12 = 16

# 清零梯度
grad_x.grad = None
print(f'梯度清零后，grad_x.grad = {grad_x.grad}')

# 重新计算
y3 = grad_x ** 2.
y3.backward()
print(f'重新计算后，grad_x.grad = {grad_x.grad}')  # 应为 4
print('说明: 梯度默认会累积，在多次反向传播前需要手动清零\n')

# 示例7: 矩阵和张量运算的反向传播
print('7、矩阵和张量运算的反向传播')
print('矩阵乘法示例')

# 创建矩阵输入
mat_a = rm.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
mat_b = rm.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
print(f'矩阵A:\n{mat_a}')
print(f'矩阵B:\n{mat_b}')

# 矩阵乘法
mat_c = mat_a @ mat_b
print(f'矩阵乘积C = A@B:\n{mat_c}')

# 创建与输出形状相同的梯度
grad_c = rm.tensor([[1.0, 0.0], [0.0, 1.0]])
mat_c.backward(gradient=grad_c)

# 查看梯度
print(f'dC/dA:\n{mat_a.grad}')  # 应为 grad_c @ B^T
print(f'dC/dB:\n{mat_b.grad}')  # 应为 A^T @ grad_c
print('说明: backward支持复杂的张量运算梯度计算\n')

# 总结
print('===== backward函数使用总结 =====')
print('1. backward()函数用于执行自动微分的反向传播，计算并存储梯度')
print('2. 对标量输出，可以直接调用backward()')
print('3. 对非标量输出，必须提供gradient参数，形状与输出相同')
print('4. 梯度默认会累积，多次反向传播前需要手动清零(grad=None)')
print('5. 设置create_graph=True可以保存梯度的计算图，用于高阶导数计算')
print('6. 中间节点通过设置retains_grad=True可以保存其梯度值')
print('7. backward支持复杂的计算图和各种张量运算')
print('8. 梯度结果存储在各节点的.grad属性中')