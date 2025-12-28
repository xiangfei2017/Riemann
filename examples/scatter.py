import sys,os
# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

import riemann as rm
# import torch

print("示例1：scatter")
x = rm.zeros(2, 4,requires_grad=True)

index = rm.tensor([[2, 1], 
                      [0, 3]])
src = rm.tensor([[5.0, 6.0], 
                    [7.0, 8.0]],requires_grad=True)

output = x.scatter(dim=1, index=index, src=src)
print(output)

output.sum().backward()
print(f'x.grad: \n{x.grad}')
print(f'src.grad: \n{src.grad}')

print("\n示例2：scatter_")

# 假设有4个样本，其类别标签分别为1, 3, 0, 2（共4个类别）
labels = rm.tensor([1, 3, 0, 2])
# 初始化一个4x4的全零张量，用于存放one-hot编码
one_hot = rm.zeros(4, 4)
# 沿维度1（即列方向）进行scatter操作，将对应标签位置的值置为1
one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

print("原始标签：", labels)
print("生成的one-hot编码：\n", one_hot)

print("\n示例3：scatter_add_")
# 初始化一个大小为5的一维目标张量
target_add0 = rm.zeros(5,requires_grad=True)
target_add = target_add0.clone()
target_add.retain_grad()

# 源张量
src_vals_add = rm.tensor([10., 20., 30., 40.],requires_grad=True)
# 索引张量，注意索引1出现了两次
indices_add = rm.tensor([1, 1, 3, 4], dtype=rm.long)

target_add.scatter_add_(0, indices_add, src_vals_add)
print("使用scatter_add_累加后的张量：", target_add)

target_add.sum().backward()
print(f'target_add.grad: \n{target_add.grad}')
print(f'src_vals_add.grad: \n{src_vals_add.grad}')

