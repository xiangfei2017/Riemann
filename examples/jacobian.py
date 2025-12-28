import sys,os,time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))  # 添加父目录到路径

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

# import torch
# from torch.autograd.functional import jacobian

import riemann as rm
from riemann.autograd.functional import jacobian, jvp, vjp

# 输入为单个 Tensor，输出也为单个 Tensor
print('1、输入为单个 Tensor，输出也为单个 Tensor')
def simple_func(x):
    return x ** 2.  # 一个简单的向量函数

x_input = rm.tensor([1.0, 2.0, 3.0] , requires_grad=True)
J = jacobian(simple_func, x_input)
print(J)
# 输出应为：
# tensor([[2., 0., .],
#         [0., 4., 0.],
#         [0., 0., 6.]])

# 输入为多个 Tensor (以元组形式传入)
print('2、输入为多个 Tensor (以元组形式传入)')
def multi_input_func(x,y):
    return x * y + y ** 2.

x_in = rm.tensor(2.0, requires_grad=True)
y_in = rm.tensor(3.0, requires_grad=True)
J_multi = jacobian(multi_input_func, (x_in, y_in))
print(J_multi)
# 输出是一个元组，包含两个雅可比矩阵（每个输入对应一个）：
# (tensor(3.), tensor(8.))

# 定义一个函数，返回两个输出
print('3、单输入，返回两个输出')
def multi_output_func(x):
    output1 = x ** 2.  # 第一个输出
    output2 = rm.sin(x)  # 第二个输出
    return output1, output2  # 以元组形式返回多个输出

x_input = rm.tensor([1.0, 2.0], requires_grad=True)

# 计算雅可比矩阵
J_multi = jacobian(multi_output_func, x_input) # J_multi 是一个元组

print("第一个输出 (x^2) 的雅可比矩阵:")
print(J_multi[0])  # 输出关于 x^2 的雅可比矩阵
#  tensor([[2., 0.],
#          [0., 4.]])

print("\n第二个输出 (sin(x)) 的雅可比矩阵:")
print(J_multi[1])  # 输出关于 sin(x) 的雅可比矩阵
#  tensor([[ 0.5403,  0.0000],
#          [ 0.0000, -0.4161]])

print('4、两个输入张量 (x, y)，并返回两个输出张量')
# 定义一个函数 func，它接受两个输入张量 (x, y)，并返回两个输出张量。
# 输出1: x 的平方和 y 的立方之和
# 输出2: x 和 y 的乘积
def func(x,y):
    out1 = x ** 2. + y ** 3.  # 第一个输出
    out2 = x * y             # 第二个输出
    return out1, out2        # 返回一个包含两个输出张量的元组

# 创建输入张量
x_in = rm.tensor([1.0, 2.0], requires_grad=True)  # 第一个输入，形状 (2,)
y_in = rm.tensor([3.0, 4.0], requires_grad=True)  # 第二个输入，形状 (2,)

# 将输入组合成一个元组传递给 jacobian 函数
inputs = (x_in, y_in)

# 计算雅可比矩阵
J = jacobian(func, inputs) # J 是一个嵌套元组

print("雅可比矩阵结果 J 的类型:", type(J))
print("J 的长度 (输出数量):", len(J))
print("J[0] 的长度 (第一个输出对输入的导数数量):", len(J[0]))
print("J[1] 的长度 (第二个输出对输入的导数数量):", len(J[1]))

# 打印详细的雅可比矩阵
print("\n第一个输出 out1 关于第一个输入 x 的雅可比矩阵 (J[0][0]):")
print(J[0][0])
print("\n第一个输出 out1 关于第二个输入 y 的雅可比矩阵 (J[0][1]):")
print(J[0][1])
print("\n第二个输出 out2 关于第一个输入 x 的雅可比矩阵 (J[1][0]):")
print(J[1][0])
print("\n第二个输出 out2 关于第二个输入 y 的雅可比矩阵 (J[1][1]):")
print(J[1][1])

# 示例5: JVP计算 - 雅可比矩阵与向量乘积
print('\n5、JVP计算 - 雅可比矩阵与向量乘积的效率对比')
def func_jvp(x):
    return x ** 3.  # 一个简单的立方函数

# 创建输入张量和向量v
x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
v = rm.tensor([0.1, 0.2, 0.3], requires_grad=True)

print(f"输入张量x: {x}")
print(f"输入向量v: {v}")
print(f"x形状: {x.shape}, v形状: {v.shape}")

# 方法1: 直接计算雅可比矩阵，然后使用riemann的@运算符与向量v相乘
start_time = time.time()
J = jacobian(func_jvp, x)
# 使用riemann的@运算符进行矩阵乘法，需要将v重塑为列向量
v_col = v.reshape(-1, 1)  # 重塑为列向量
jvp_manual = (J @ v_col).reshape(-1)  # 矩阵乘法后再重塑回一维向量
manual_time = time.time() - start_time

print(f"\n方法1 (直接计算):")
print(f"雅可比矩阵J:")
print(J)
print(f"J @ v 结果: {jvp_manual}")
print(f"计算耗时: {manual_time*1000:.4f} ms")

# 方法2: 使用jvp函数直接计算
start_time = time.time()
_, jvp_result = jvp(func_jvp, x, v)
jvp_time = time.time() - start_time

print(f"\n方法2 (使用jvp函数):")
print(f"jvp结果: {jvp_result}")
print(f"计算耗时: {jvp_time*1000:.4f} ms")
print(f"速度提升: {manual_time/jvp_time:.2f}x")
print(f"结果一致性验证: {(jvp_manual - jvp_result).abs().max() < 1e-6}")

# 示例6: VJP计算 - 向量与雅可比矩阵乘积
print('\n6、VJP计算 - 向量与雅可比矩阵乘积的效率对比')
def func_vjp(x):
    return x ** 3.  # 一个简单的立方函数

# 创建输入张量和向量v（与输出形状匹配）
x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
v = rm.tensor([0.1, 0.2, 0.3], requires_grad=True)

print(f"输入张量x: {x}")
print(f"输入向量v: {v}")
print(f"x形状: {x.shape}, v形状: {v.shape}")

# 方法1: 直接计算雅可比矩阵，然后使用riemann的@运算符计算v @ J
start_time = time.time()
J = jacobian(func_vjp, x)
# 使用riemann的@运算符进行矩阵乘法，需要将v重塑为行向量
v_row = v.reshape(1, -1)  # 重塑为行向量
vjp_manual = (v_row @ J).reshape(-1)  # 矩阵乘法后再重塑回一维向量
manual_time = time.time() - start_time

print(f"\n方法1 (直接计算):")
print(f"雅可比矩阵J:")
print(J)
print(f"v @ J 结果: {vjp_manual}")
print(f"计算耗时: {manual_time*1000:.4f} ms")

# 方法2: 使用vjp函数直接计算
start_time = time.time()
_, vjp_result = vjp(func_vjp, x, v)
vjp_time = time.time() - start_time

print(f"\n方法2 (使用vjp函数):")
print(f"vjp结果: {vjp_result}")
print(f"计算耗时: {vjp_time*1000:.4f} ms")
print(f"速度提升: {manual_time/vjp_time:.2f}x")
print(f"结果一致性验证: {(vjp_manual - vjp_result).abs().max() < 1e-6}")
