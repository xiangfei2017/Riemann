import sys,os,time
# 添加time模块以进行效率对比
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))  # 添加父目录到路径

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

# 导入riemann模块
import riemann as rm
from riemann.autograd.functional import hessian, hvp, vhp
from riemann.autograd import grad

# import torch as rm
# from torch.autograd.functional import hessian
# from torch.autograd import grad

# 打印标题
print('===== Riemann 自动微分库 - Hessian矩阵计算示例 =====\n')

# 示例1: 单变量函数 f(x) = x^2
print('1、单变量函数 f(x) = x^2')
print('f(x) = x²')
print('x=3')
def simple_func(x):
    return x ** 2.  # 一个简单的标量函数

x_input = rm.tensor(3.0, requires_grad=True)
H = hessian(simple_func, x_input)
print('Hessian矩阵结果:')
print(H)
# 输出应为: tensor(2.0)
print('说明: 对于f(x)=x²，二阶导数为2，所以Hessian矩阵是标量2.0\n')

# 示例2: 多变量函数 f(x) = x[0]^2 + x[1]^2
print('2、多变量函数 f(x) = x[0]^2 + x[1]^2')
print('f(x,y) = x²+y²')
print('x=1, y=2')
def multi_var_func(x):
    return x[0] ** 2. + x[1] ** 2.  # 二元二次函数

x_input = rm.tensor([1.0, 2.0], requires_grad=True)
H = hessian(multi_var_func, x_input)
print('Hessian矩阵结果:')
print(H)
# 输出应为: 
# tensor([[2., 0.],
#         [0., 2.]])
print('说明: 对于f(x,y)=x²+y²，Hessian矩阵是对角矩阵，对角线元素为2\n')

# 示例3: 输入为多个张量 (以元组形式传入)
print('3、输入为多个张量 (以元组形式传入)')
print('f(x,y) = x²y + y³')
print('x=2, y=3')
def two_input_func(x, y):
    return x ** 2. * y + y ** 3.  # 两个输入变量的函数

x_in = rm.tensor(2.0, requires_grad=True)
y_in = rm.tensor(3.0, requires_grad=True)
H_multi = hessian(two_input_func, (x_in, y_in))
print('Hessian矩阵结果 (是一个元组，包含四个分量):')
print('Hessian of f w.r.t (x,x) ∂²f/∂x² :', H_multi[0][0])
print('Hessian of f w.r.t (x,y) ∂²f/∂x∂y:', H_multi[0][1])
print('Hessian of f w.r.t (y,x) ∂²f/∂y∂x:', H_multi[1][0])
print('Hessian of f w.r.t (y,y) ∂²f/∂y² :', H_multi[1][1])
# 输出应为:
# Hessian of f w.r.t (x,x): tensor(6.)
# Hessian of f w.r.t (x,y): tensor(4.)
# Hessian of f w.r.t (y,x): tensor(4.)
# Hessian of f w.r.t (y,y): tensor(18.)
print('说明: 对于多输入函数，Hessian返回的是一个嵌套元组，包含所有二阶混合偏导数\n')

# 示例4: 具有交叉项的多变量函数
print('4、具有交叉项的多变量函数 f(x) = x[0]^2 + x[1]^2 + x[0]*x[1]')
print('f(x,y) = x²+y²+x·y')
print('x=1, y=1')
def cross_term_func(x):
    return x[0] ** 2. + x[1] ** 2. + x[0] * x[1]  # 包含交叉项的函数

x_input = rm.tensor([1.0, 1.0], requires_grad=True)
H = hessian(cross_term_func, x_input)
print('Hessian矩阵结果:')
print(H)
# 输出应为: 
# tensor([[2., 1.],
#         [1., 2.]])
print('说明: 由于存在交叉项x[0]*x[1]，Hessian矩阵的非对角线元素不为零\n')

# 示例5: 使用create_graph参数保留计算图
print('5、使用create_graph=True参数保留计算图')
print('f(x) = x³')
print('x=2')
def cubic_func(x):
    return x ** 3.  # 三次函数

x_in = rm.tensor(2.0, requires_grad=True)
# 创建计算图，以便可以对Hessian矩阵进一步求导
H_with_graph = hessian(cubic_func, x_in, create_graph=True)
print('带计算图的Hessian矩阵:')
print(H_with_graph)
print('Hessian矩阵的requires_grad属性:', H_with_graph.requires_grad)

# 尝试对Hessian矩阵再求导
# 定义一个函数，将Hessian矩阵作为输入
print('\n对Hessian矩阵再求导:')
def hessian_func(h):
    return h * 2.  # 对Hessian矩阵进行简单运算

# 计算Hessian的Hessian（对于标量输入，这相当于三阶导数）
third_derivative = grad(hessian_func(H_with_graph), x_in)
print('三阶导数 (Hessian的导数):')
print(third_derivative)
# 输出应为: tensor(12.0)
print('说明: 对于f(x)=x³，三阶导数为6x，得到12\n')

# 示例6: 高阶张量输入
print('6、高阶张量输入 - 形状为(2,2)的矩阵')
print('f(x) = (x[0,0] + x[1,1])²')
print('x=[[1,2],[3,4]]')
def matrix_func(x):
    # 计算矩阵的迹（对角线元素之和）的平方
    return (x[0,0] + x[1,1]) ** 2.

# 创建一个2x2的矩阵输入
x_matrix = rm.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
H_matrix = hessian(matrix_func, x_matrix)
print('Hessian矩阵形状:', H_matrix.shape)
print('Hessian矩阵结果:')
print(H_matrix)
print('说明: 对于形状为(2,2)的输入，Hessian矩阵形状为(2,2,2,2)\n')

# 示例7: HVP计算 - Hessian矩阵与向量乘积
print('\n7、HVP计算 - Hessian矩阵与向量乘积的效率对比')
def func_hvp(x):
    return (x ** 2.).sum()  # 一个简单的二次函数，输出标量

# 创建输入张量和向量v
x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
v = rm.tensor([0.1, 0.2, 0.3], requires_grad=True)

print(f"输入张量x: {x}")
print(f"输入向量v: {v}")
print(f"x形状: {x.shape}, v形状: {v.shape}")

# 方法1: 直接计算Hessian矩阵，然后使用riemann的@运算符与向量v相乘
start_time = time.time()
H = hessian(func_hvp, x)
# 使用riemann的@运算符进行矩阵乘法，需要将v重塑为列向量
v_col = v.reshape(-1, 1)  # 重塑为列向量
hvp_manual = (H @ v_col).reshape(-1)  # 矩阵乘法后再重塑回一维向量
manual_time = time.time() - start_time

print(f"\n方法1 (直接计算):")
print(f"Hessian矩阵H:")
print(H)
print(f"H @ v 结果: {hvp_manual}")
print(f"计算耗时: {manual_time*1000:.4f} ms")

# 方法2: 使用hvp函数直接计算
start_time = time.time()
_,hvp_result = hvp(func_hvp, x, v)
hvp_time = time.time() - start_time

print(f"\n方法2 (使用hvp函数):")
print(f"hvp结果: {hvp_result}")
print(f"计算耗时: {hvp_time*1000:.4f} ms")
print(f"速度提升: {manual_time/hvp_time:.2f}x")
print(f"结果一致性验证: {(hvp_manual - hvp_result).abs().max() < 1e-6}")

# 示例8: VHP计算 - 向量与Hessian矩阵乘积
print('\n8、VHP计算 - 向量与Hessian矩阵乘积的效率对比')
def func_vhp(x):
    return (x ** 2.).sum()  # 一个简单的二次函数，输出标量

# 创建输入张量和向量v
x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
v = rm.tensor([0.1, 0.2, 0.3], requires_grad=True)

print(f"输入张量x: {x}")
print(f"输入向量v: {v}")
print(f"x形状: {x.shape}, v形状: {v.shape}")

# 方法1: 直接计算Hessian矩阵，然后使用riemann的@运算符计算v @ H
start_time = time.time()
H = hessian(func_vhp, x)
# 使用riemann的@运算符进行矩阵乘法，需要将v重塑为行向量
v_row = v.reshape(1, -1)  # 重塑为行向量
vhp_manual = (v_row @ H).reshape(-1)  # 矩阵乘法后再重塑回一维向量
manual_time = time.time() - start_time

print(f"\n方法1 (直接计算):")
print(f"Hessian矩阵H:")
print(H)
print(f"v @ H 结果: {vhp_manual}")
print(f"计算耗时: {manual_time*1000:.4f} ms")

# 方法2: 使用vhp函数直接计算
start_time = time.time()
_,vhp_result = vhp(func_vhp, x, v)
vhp_time = time.time() - start_time

print(f"\n方法2 (使用vhp函数):")
print(f"vhp结果: {vhp_result}")
print(f"计算耗时: {vhp_time*1000:.4f} ms")
print(f"速度提升: {manual_time/vhp_time:.2f}x")
print(f"结果一致性验证: {(vhp_manual - vhp_result).abs().max() < 1e-6}")

# 更新总结部分
print('\n===== Hessian矩阵计算总结 =====')
print('1. Hessian矩阵是函数二阶导数的矩阵，对于标量函数f(x)，H[i,j] = ∂²f/∂x_i∂x_j')
print('2. riemann的hessian函数支持单输入、多输入、高阶张量输入等多种情况')
print('3. 对于多输入函数，返回嵌套元组，包含所有二阶偏导数组合')
print('4. 可通过create_graph参数保留计算图，支持高阶导数计算')
print('5. hessian函数要求被求导函数必须返回标量值')
print('6. HVP (Hessian-Vector Product) 和 VHP (Vector-Hessian Product) 可以高效计算，无需显式构建完整Hessian矩阵')
print('7. 由于Hessian矩阵的对称性，HVP和VHP的结果在数值上是相同的')