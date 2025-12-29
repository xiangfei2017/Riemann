"""
演示Riemann的梯度计算功能

本示例展示了如何使用Riemann的grad()函数计算函数的梯度。
包括单输入单输出、多输入单输出、多输入多输出等情况的梯度计算。

主要功能：
1. 单输入单输出函数的梯度计算
2. 多输入单输出函数的梯度计算
3. 高阶导数计算
4. 偏导数计算
5. 使用梯度进行优化
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
print('===== Riemann 自动微分库 - grad函数使用演示 =====\n')

# 示例1: 基本的grad使用 - 单输入单输出
print('1、基本的grad使用 - 单输入单输出')
print('f(x) = x^2')
print('x = 3.0')

def simple_func(x):
    return x ** 2.

# 创建输入张量并设置requires_grad=True
x = rm.tensor(3.0, requires_grad=True)
print(f'输入x: {x}')

# 计算函数值
y = simple_func(x)
print(f'函数值y = f(x): {y}')

# 使用grad函数计算梯度
gradient = grad(y, x)
print(f'使用grad计算的梯度df/dx: {gradient}')
print('说明: grad函数返回梯度张量，而不是像backward那样存储在.grad属性中\n')

# 示例2: 多输入的grad使用
print('2、多输入的grad使用')
print('f(x, y) = x*y + x^2')
print('x = 2.0, y = 3.0')

def multi_input_func(x, y):
    return x * y + x ** 2.

# 创建多个输入
x_multi = rm.tensor(2.0, requires_grad=True)
y_multi = rm.tensor(3.0, requires_grad=True)
print(f'输入x: {x_multi}, y: {y_multi}')

# 计算函数值
z_multi = multi_input_func(x_multi, y_multi)
print(f'函数值z = f(x, y): {z_multi}')

# 计算对x和y的梯度
grad_x, grad_y = grad(z_multi, [x_multi, y_multi])
print(f'dz/dx = {grad_x}')  # 应为 y + 2x = 7
print(f'dz/dy = {grad_y}')  # 应为 x = 2
print('说明: 对于多输入，grad返回与输入列表对应的梯度元组\n')

# 示例3: 向量输入的梯度计算
print('3、向量输入的梯度计算')
print('f(x) = (x^2).sum()')
print('x = [1.0, 2.0, 3.0]')

def vector_func(x):
    return (x ** 2.).sum()

# 创建向量输入
x_vec = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f'输入x: {x_vec}')

# 计算函数值
y_vec = vector_func(x_vec)
print(f'函数值y = f(x): {y_vec}')

# 计算梯度
grad_vec = grad(y_vec, x_vec)
print(f'梯度df/dx: {grad_vec}')  # 应为 [2, 4, 6]
print('说明: grad函数支持向量和张量输入的梯度计算\n')

# 示例4: 使用grad_outputs参数
print('4、使用grad_outputs参数')
print('对于非标量输出，需要提供grad_outputs参数')
print('f(x) = x^2')
print('x = [1.0, 2.0, 3.0]')

# 创建输入
x_grad_out = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f'输入x: {x_grad_out}')

# 非标量输出
y_grad_out = x_grad_out ** 2.
print(f'非标量输出y: {y_grad_out}')

# 提供grad_outputs参数
grad_outputs = rm.tensor([1.0, 2.0, 3.0])
grad_result = grad(y_grad_out, x_grad_out, grad_outputs=grad_outputs)
print(f'使用grad_outputs计算的梯度: {grad_result}')
print('说明: grad_outputs相当于链式法则中的上游梯度\n')

# 示例5: 使用create_graph参数进行高阶导数计算
print('5、使用create_graph参数进行高阶导数计算')
print('f(x) = x^3')
print('x = 2.0')

# 创建输入
h_x = rm.tensor(2.0, requires_grad=True)
print(f'输入x: {h_x}')

# 一阶导数
h_y = h_x ** 3.
print(f'函数值f(x) = {h_y}')

# 使用create_graph=True计算一阶导数
dy_dx = grad(h_y, h_x, create_graph=True)[0]
print(f'一阶导数df/dx = {dy_dx}')  # 应为 3x² = 12

# 对一阶导数再求导，得到二阶导数
d2y_dx2 = grad(dy_dx, h_x)[0]
print(f'二阶导数d²f/dx² = {d2y_dx2}')  # 应为 6x = 12
print('说明: create_graph=True允许对梯度结果再次求导\n')

# 示例6: 使用strict参数
print('6、使用strict参数')
print('测试当输出与某些输入无关时的行为')
print('f(x, y) = x^2')
print('x = 2.0, y = 3.0')

def independent_func(x, y):
    # 注意：这个函数只依赖于x，与y无关
    return x ** 2.

# 创建输入
x_indep = rm.tensor(2.0, requires_grad=True)
y_indep = rm.tensor(3.0, requires_grad=True)
print(f'输入x: {x_indep}, y: {y_indep}')

# 计算函数值
z_indep = independent_func(x_indep, y_indep)
print(f'函数值z = f(x, y): {z_indep}')

# 默认strict=False，对无关输入返回None
grad_x_indep, grad_y_indep = grad(z_indep, [x_indep, y_indep], allow_unused=True)
print(f'allow_unused=True时，dz/dx = {grad_x_indep}, dz/dy = {grad_y_indep}')

# strict=True，对无关输入会抛出错误
try:
    grad(z_indep, [x_indep, y_indep], allow_unused=False)
    print('错误: 应该抛出异常但没有')
except RuntimeError as e:
    print(f'allow_unused=False时，正确抛出异常: {str(e)}')
print('说明: allow_unused参数控制当输出与输入无关时的行为\n')

# 示例7: 复杂计算图中的梯度计算
print('7、复杂计算图中的梯度计算')
print('f(a, b) = sin(a*b) + cos(a)')
print('a = 1.0, b = 2.0')

def complex_func(a, b):
    prod = a * b
    sin_term = rm.sin(prod)
    cos_term = rm.cos(a)
    return sin_term + cos_term

# 创建输入
a_complex = rm.tensor(1.0, requires_grad=True)
b_complex = rm.tensor(2.0, requires_grad=True)
print(f'输入a: {a_complex}, b: {b_complex}')

# 计算函数值
z_complex = complex_func(a_complex, b_complex)
print(f'函数值z = f(a, b): {z_complex}')

# 计算梯度
grad_a_complex, grad_b_complex = grad(z_complex, [a_complex, b_complex])
print(f'dz/da = {grad_a_complex}')  # 应为 b*cos(a*b) - sin(a)
print(f'dz/db = {grad_b_complex}')  # 应为 a*cos(a*b)
print('说明: grad函数可以处理复杂的计算图和嵌套运算\n')

# 示例8: 矩阵运算的梯度计算
print('8、矩阵运算的梯度计算')
print('矩阵乘法和求和操作')

# 创建矩阵输入
mat_a = rm.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
mat_b = rm.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
print(f'矩阵A:\n{mat_a}')
print(f'矩阵B:\n{mat_b}')

# 矩阵运算
mat_product = mat_a @ mat_b
mat_sum = mat_product.sum()
print(f'矩阵乘积和: {mat_sum}')

# 计算梯度
grad_a, grad_b = grad(mat_sum, [mat_a, mat_b])
print(f'dsum/dA:\n{grad_a}')  # 应为 B^T
print(f'dsum/dB:\n{grad_b}')  # 应为 A^T
print('说明: grad函数支持矩阵和张量运算的梯度计算\n')

# 总结
print('===== grad函数使用总结 =====')
print('1. grad函数计算输出对输入的梯度，返回梯度张量而不是存储在.grad属性中')
print('2. 支持单个或多个输入，返回与输入列表对应的梯度元组')
print('3. 对标量输出，可以不提供grad_outputs参数')
print('4. 对非标量输出，必须提供与输出形状相同的grad_outputs参数')
print('5. create_graph=True允许对梯度结果再次求导，用于高阶导数计算')
print('6. allow_unused参数控制当输出与输入无关时的行为(True返回None，False抛出异常)')
print('7. 支持复杂的计算图、嵌套运算以及各种张量操作')
print('8. 是构建更高级自动微分功能(jacobian, hessian, jvp, vjp等)的基础')