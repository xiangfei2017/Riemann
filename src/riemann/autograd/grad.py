# BSD 3-Clause License
#
# Copyright (c) 2025, Fei Xiang
# All rights reserved.
#
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
#
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
Riemann Library Gradient Computation Module

This module provides core gradient computation functionality for tensor differentiation, 
including backpropagation implementation and gradient calculation utilities.

Core components:
- `backward`: Wrapper method that delegates to TN.backward() for reverse-mode autodiff
- `grad`: Computes gradients of outputs w.r.t. inputs, returning gradient tuples directly
- `gradcheck`: Numerical gradient verification via finite differences
- `Function`: Base class for custom autograd.Function-style forward/backward definitions

All functions support Riemann Tensor (TN) type and automatic differentiation.
"""

from typing import Callable, Any
import numpy as np
from ..tensordef import *
from ..gradmode import set_grad_enabled


def backward(self:TN, gradient: TN|None = None, retain_graph: bool = False, create_graph: bool = False):
        """执行反向模式自动微分（反向传播）。

        该函数从当前张量开始，通过计算图向后传播梯度，为所有叶子节点
        或设置了retains_grad=True的中间节点计算并存储梯度。

        参数：
            self (TN): 用于触发反向传播的张量
            retain_graph (bool, 可选): 该参数用于与PyTorch兼容，
                                      riemann反向传播不依赖此参数，
                                      无论为True还是False，riemann都支持多次反向传播调用
            gradient (TN | None, 可选): 输出张量self的梯度，默认为None
            create_graph (bool, 可选): 是否在梯度计算过程中创建计算图，
                                      设置为True以进行高阶导数计算，默认为False

        返回：
            None: 此函数不返回值，梯度结果直接存储在每个节点的.grad属性中

        异常：
            RuntimeError: 当当前张量是叶子节点、当前张量不是标量，
                         或当前张量不需要梯度时抛出
        """
        return self.backward(gradient=gradient, create_graph=create_graph)

def grad(outputs:TN,
         inputs:TN|list[TN]|tuple[TN],
         retain_graph: bool = False,
         grad_outputs:TN|None=None,
         create_graph : bool = False,
         allow_unused: bool = False)->tuple[TN,...]:
    """
    计算给定输出相对于输入的梯度，直接返回梯度张量而不修改原始张量。
    
    这是Riemann框架中的核心梯度计算函数。与backward()方法类似，
    但它直接返回计算出的梯度张量，而不是将它们存储在输入张量的.grad属性中。
    这使其更适合于高级梯度计算场景，如计算雅可比矩阵、海森矩阵等。
    
    工作原理：
        1. 验证 outputs/inputs 参数合法性，初始化计算图（_init_calc_graph）
        2. 设置初始输出梯度（标量隐式 1.0 或用户提供的 grad_outputs）
        3. 使用 list 作为栈（pop(-1) O(1)）对计算图进行 DFS 反向遍历
        4. 在 with set_grad_enabled(create_graph) 上下文中执行梯度传播，
           确保 create_graph 参数能正确控制中间张量的 autograd 追踪
        5. 复用 TN._propagate_grad_to_sources 向来源节点传播梯度，
           与 TN._backward_without_hooks 使用完全相同的传播逻辑
        6. 节点出栈时先检查是否是 inputs 之一，若是则保存梯度到返回列表；
           所有 inputs 的梯度一旦全部收集（O(1) 计数器检测），立即 break 提前退出
        7. 遍历结束后不再显式清理计算图缓存：下次 grad/backward 的 _init_calc_graph
           会重新遍历整张图覆盖所有节点的 grad_value 和 rcv_grad_count，完全避免残留状态
    
    与backward()方法的区别：
        1. backward(): 将梯度存储在张量的.grad属性中，适合参数更新
        2. grad(): 返回梯度张量而不修改原始张量，适合高级梯度计算
        3. grad() 在计算完所需梯度后立即停止（lazy evaluation），而backward()遍历整个计算图
        4. grad() 不累积梯度，每次调用都重新计算；backward() 的 .grad 会累加
    
    参数：
        outputs (TN): 需要计算梯度的输出张量
        inputs (TN | list[TN] | tuple[TN]): 需要计算梯度的输入张量或张量列表
        grad_outputs (TN | None, 可选): 输出张量的梯度
            - None: 默认值，要求输出张量是标量，梯度将设置为1.0
            - TN: 用户提供的梯度张量，必须与输出张量形状相同
        retain_graph (bool, 可选): 该参数用于与PyTorch兼容，
                riemann反向传播不依赖此参数，
                无论为True还是False，riemann都支持多次grad调用
        create_graph (bool, 可选): 是否在梯度计算过程中创建计算图
            - False: 默认值，梯度计算不被追踪，无法计算高阶导数
            - True: 梯度计算被追踪，可用于高阶导数计算（如海森矩阵）
        allow_unused (bool, 可选): 严格模式控制
            - False: 默认值，当输出与输入无关时抛出RuntimeError
            - True: 当输出与输入无关时返回None作为梯度

    返回：
        tuple[TN, ...]: 与输入张量列表对应的梯度张量元组，顺序与inputs相同
        - 对于与输出相关的输入，返回相应的梯度张量
        - 对于与输出无关的输入，在非严格模式下返回None
    
    异常：
        RuntimeError: 在以下情况下抛出：
            1. 输入张量不是TN类型或requires_grad=False
            2. 当未提供grad_outputs时，输出不是标量
            3. 当未提供grad_outputs时，输出是复标量
            4. 提供的grad_outputs张量形状与输出张量形状不匹配
            5. 在strict=True模式下，当检测到输出与某个输入无关时
    
    示例：
        >>> # 基本用法 - 计算单个输入的梯度
        >>> x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = x.sum()
        >>> dx = grad(y, x)[0]
        >>> print(dx)  # 输出: [1. 1. 1.]
        
        >>> # 计算多个输入的梯度
        >>> x = tensor([1.0, 2.0], requires_grad=True)
        >>> w = tensor([0.5, 0.5], requires_grad=True)
        >>> y = (x * w).sum()
        >>> dx, dw = grad(y, [x, w])
        >>> print(dx)  # 输出: [0.5 0.5]
        >>> print(dw)  # 输出: [1. 2.]
        
        >>> # 高阶导数计算
        >>> x = tensor(2.0, requires_grad=True)
        >>> y = x * x * x  # y = x^3
        >>> dy_dx = grad(y, x, create_graph=True)[0]  # dy/dx = 3x^2
        >>> d2y_dx2 = grad(dy_dx, x)[0]  # d²y/dx² = 6x
        >>> print(d2y_dx2)  # 输出: [12.]
        
    注意：
        1. 此函数不会修改任何输入张量的.grad属性
        2. 设置create_graph=True会创建额外的计算图，可能会增加内存使用
        3. 对于大型计算图，grad()可能比backward()更高效，因为它在计算完所需梯度后立即停止
        4. 如果只需要计算特定输入的梯度，使用grad()比backward()更节省内存
        5. 可安全地与backward()交错调用，两者各自独立初始化/重置计算图状态
    """
    if not outputs.requires_grad:
        raise RuntimeError('Only a tensor require grad can call grad()')
    
    # 处理输入为单个张量或张量列表的情况
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    
    # 验证输入张量是否可求导
    for i, inp in enumerate(inputs):
        if not isinstance(inp, TN) or not inp.requires_grad:
            raise RuntimeError(f"Input {i} must be a tensor that requires grad")

    if grad_outputs is None:
        if outputs._array.ndim > 0:
            raise RuntimeError('Only a scalar can call grad() without argument grad_outputs')
        
        if outputs.is_complex():
            raise RuntimeError(f'grad can be implicitly created only for real scalar outputs but got {outputs.dtype}')
        
        outputs._init_calc_graph()
        outputs.grad_value = tensor(1.0,
                                    dtype=outputs.dtype,
                                    device=outputs.device,
                                    requires_grad=create_graph)
    
    elif isinstance(grad_outputs,TN):
        if grad_outputs._array.shape == outputs._array.shape:
            outputs._init_calc_graph()
            outputs.grad_value = grad_outputs.to(outputs.device).detach().requires_grad_(create_graph)
        else:
            raise RuntimeError('shape of grad_outputs need to be same as the shape of outputs')
    else:
        raise TypeError(f'grad_outputs can be either tensor or None, but got {type(grad_outputs)}')
        
    grad_list = [None] * len(inputs)      # 与 inputs 一一对应的返回梯度列表
    saved_flags = [False] * len(inputs)   # 标记 inputs[i] 是否已收集完梯度
    saved_count = 0                       # saved_flags[i]==True 的个数，O(1) 检测完成
    q = [outputs]                         # list 作为栈：存放 rcv_grad_count==0 的节点
    
    with set_grad_enabled(create_graph):
        while q:
            item: TN = q.pop(-1)  # O(1) 队尾弹出

            # 节点是 inputs 之一 → 保存梯度到返回列表
            for i, var in enumerate(inputs):
                if not saved_flags[i] and var is item:
                    saved_flags[i] = True
                    saved_count += 1
                    grad_list[i] = var.grad_value if create_graph else var.grad_value.detach()
                    break  # 一个 item 不可能同时匹配 inputs 多个元素

            # 所有 inputs 的梯度已收集 → 立即退出（grad 的 lazy evaluation 核心优化）
            if saved_count == len(inputs):
                break

            # 向 fromvars 传播梯度，rcv_grad_count==0 的来源节点加入 vars_received_grad
            vars_received_grad = set()
            item._propagate_grad_to_sources(vars_received_grad)
            item.grad_value = None
            q.extend(vars_received_grad)
        # end of while
    # end of with
    
    # 遍历结束后不再显式清理计算图缓存：下次 grad/backward 的 _init_calc_graph
    # 会重新遍历整张图，完全覆盖节点状态。Benchmark 已验证交错调用的正确性。

    # strict 模式：不允许对不参与计算的 inputs 求导
    if not allow_unused:
        for i, grad_val in enumerate(grad_list):
            if grad_val is None:
                raise RuntimeError(f"Output is independent of input ({i}), Set allow_unused=True if this is the desired behavior.")
    
    return tuple(grad_list)  # type: ignore
#end of grad


def _sum_all(tensors):
    """辅助函数：累加多个TN张量"""
    if not tensors:
        raise ValueError("Empty tensor list/tuple")
    result = tensors[0]
    for ts in tensors[1:]:
        result = result + ts
    return result


def gradcheck(func: Callable[..., Any], inputs: tuple[TN, ...], eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-3, raise_exception: bool = True, check_sparse_nnz: bool = False, fast_mode: bool = False) -> bool:
    """
    验证给定函数的梯度计算是否正确，通过比较数值梯度和解析梯度。

    该函数通过在输入参数上添加微小扰动来计算数值梯度（有限差分法），
    并将其与使用自动微分计算的解析梯度进行比较。

    参数：
        func (Callable[..., Any]): 需要验证梯度的函数，该函数应接受一个或多个TN张量作为输入，
                                  并返回一个或多个TN张量作为输出。
        inputs (tuple[TN, ...]): 用于测试的输入张量元组，所有张量必须设置requires_grad=True，
                               且必须是双精度（float64）类型。
        eps (float, 可选): 用于计算数值梯度的微小扰动，默认为1e-6。
        atol (float, 可选): 绝对误差容限，默认为1e-5。
        rtol (float, 可选): 相对误差容限，默认为1e-3。
        raise_exception (bool, 可选): 如果梯度检查失败，是否抛出异常，默认为True。
        check_sparse_nnz (bool, 可选): 是否检查稀疏张量的非零元素，默认为False。
        fast_mode (bool, 可选): 是否使用快速模式（只检查每个张量的一个元素），默认为False。

    返回：
        bool: 如果梯度检查通过，返回True；否则返回False（或抛出异常，取决于raise_exception参数）。

    异常：
        RuntimeError: 当梯度检查失败且raise_exception为True时抛出。
        ValueError: 当输入参数不符合要求时抛出。

    示例：
        >>> # 定义一个简单的函数
        >>> def f(x, y):
        >>>     return x * y
        >>> 
        >>> # 准备输入（必须是双精度类型）
        >>> x = tensor(2.0, requires_grad=True, dtype=np.float64)
        >>> y = tensor(3.0, requires_grad=True, dtype=np.float64)
        >>> 
        >>> # 验证梯度
        >>> result = gradcheck(f, (x, y))
        >>> print(result)  # 输出: True

    注意：
        1. 该函数要求输入张量必须是双精度（float64）类型，以确保数值梯度计算的精度。
        2. 该函数目前仅支持实值函数和实值输入。
        3. 对于具有多个输出的函数，梯度将被求和。
        4. 小的误差是正常的，尤其是对于复杂函数或使用较大eps值时。
        5. fast_mode=True可以加速检查，但会降低验证的全面性。
    """
        
    # 验证输入参数
    if not callable(func):
        raise ValueError("func must be a callable")
    
    if not isinstance(inputs, (tuple, list)):
        raise ValueError("inputs must be a tuple or list of tensors")
    
    # 检查是否支持稀疏张量检查
    if check_sparse_nnz:
        raise ValueError("check_sparse_nnz is not supported in current version of Riemann (sparse tensors are not implemented)")
    
    # 检查所有输入是否为TN张量且requires_grad=True，以及是否为双精度类型
    for i, inp in enumerate(inputs):
        if not isinstance(inp, TN):
            raise ValueError(f"Input {i} must be a TN tensor")
        if not inp.requires_grad:
            raise ValueError(f"Input {i} must have requires_grad=True")
        if inp.dtype != np.float64:
            raise ValueError(f"Input {i} must be double precision (float64), got {inp.dtype}")
        if np.issubdtype(inp.dtype, np.complexfloating):
            raise ValueError(f"Input {i} is complex, but gradcheck only supports real inputs")
    
    inputs = tuple(inputs)
    
    # 计算解析梯度（使用自动微分）
    outputs = func(*inputs)
    
    # 处理多个输出的情况
    if isinstance(outputs, (tuple, list)):
        outputs = _sum_all(outputs)
    elif not isinstance(outputs, TN):
        raise ValueError("func must return a TN tensor or a tuple/list of TN tensors")
    
    # 确保输出是标量
    if outputs._array.ndim > 0:
        outputs = outputs.sum()
    
    # 计算解析梯度
    analytical_grads = grad(outputs, list(inputs), create_graph=False)
    
    # 计算数值梯度（有限差分法）
    numerical_grads = []
    
    for i, input_tensor in enumerate(inputs):
        # 创建与输入相同形状的零张量来存储数值梯度
        numerical_grad = zeros_like(input_tensor, dtype=input_tensor.dtype)
        
        # 遍历张量的每个元素或只检查一个元素（快速模式）
        if fast_mode:
            # 只检查第一个元素
            idx = tuple([0] * input_tensor._array.ndim)
            
            # 创建输入副本列表
            inputs_pos = list(inputs)
            inputs_neg = list(inputs)
            
            # 计算正向扰动的输出
            # 为正向扰动创建全新的输入张量
            x_pos_data = input_tensor._array.copy()
            x_pos_data[idx] += eps
            x_pos = zeros_like(input_tensor, dtype=input_tensor.dtype, device=input_tensor.device)
            x_pos._array = x_pos_data
            x_pos.requires_grad = input_tensor.requires_grad
            inputs_pos[i] = x_pos
            outputs_pos = func(*inputs_pos)
            if isinstance(outputs_pos, (tuple, list)):
                outputs_pos = _sum_all(outputs_pos)
            if outputs_pos._array.ndim > 0:
                outputs_pos = outputs_pos.sum()
            
            # 计算负向扰动的输出
            # 为负向扰动创建全新的输入张量
            x_neg_data = input_tensor._array.copy()
            x_neg_data[idx] -= eps
            x_neg = zeros_like(input_tensor, dtype=input_tensor.dtype, device=input_tensor.device)
            x_neg._array = x_neg_data
            x_neg.requires_grad = input_tensor.requires_grad
            inputs_neg[i] = x_neg
            outputs_neg = func(*inputs_neg)
            if isinstance(outputs_neg, (tuple, list)):
                outputs_neg = _sum_all(outputs_neg)
            if outputs_neg._array.ndim > 0:
                outputs_neg = outputs_neg.sum()
            
            # 计算数值梯度（中心差分）
            numerical_grad._array[idx] = (outputs_pos._array - outputs_neg._array) / (2 * eps)
        else:
            # 遍历所有元素
            # 统一使用索引迭代方法，适用于CUDA和CPU张量
            # 获取张量形状
            shape = input_tensor._array.shape
            # 计算总元素数
            num_elements = input_tensor._array.size
            # 遍历每个元素的线性索引
            for linear_idx in range(num_elements):
                # 将线性索引转换为多维索引
                idx = np.unravel_index(linear_idx, shape)
                
                # 创建输入副本列表
                inputs_pos = list(inputs)
                inputs_neg = list(inputs)
                
                # 计算正向扰动的输出
                # 为正向扰动创建全新的输入张量
                x_pos_data = input_tensor._array.copy()
                x_pos_data[idx] += eps
                x_pos = zeros_like(input_tensor, dtype=input_tensor.dtype, device=input_tensor.device)
                x_pos._array = x_pos_data
                x_pos.requires_grad = input_tensor.requires_grad
                inputs_pos[i] = x_pos
                outputs_pos = func(*inputs_pos)
                if isinstance(outputs_pos, (tuple, list)):
                    outputs_pos = _sum_all(outputs_pos)
                if outputs_pos._array.ndim > 0:
                    outputs_pos = outputs_pos.sum()
                
                # 计算负向扰动的输出
                # 为负向扰动创建全新的输入张量
                x_neg_data = input_tensor._array.copy()
                x_neg_data[idx] -= eps
                x_neg = zeros_like(input_tensor, dtype=input_tensor.dtype, device=input_tensor.device)
                x_neg._array = x_neg_data
                x_neg.requires_grad = input_tensor.requires_grad
                inputs_neg[i] = x_neg
                outputs_neg = func(*inputs_neg)
                if isinstance(outputs_neg, (tuple, list)):
                    outputs_neg = _sum_all(outputs_neg)
                if outputs_neg._array.ndim > 0:
                    outputs_neg = outputs_neg.sum()
                
                # 计算数值梯度（中心差分）
                numerical_grad._array[idx] = (outputs_pos._array - outputs_neg._array) / (2 * eps)
        
        numerical_grads.append(numerical_grad)
    
    # 比较数值梯度和解析梯度
    for i, (num_grad, an_grad) in enumerate(zip(numerical_grads, analytical_grads)):
        if num_grad is None or an_grad is None:
            if raise_exception:
                raise RuntimeError(f"Gradient {i} is None")
            return False
        
        # 计算绝对误差和相对误差
        abs_error = np.abs(num_grad._array - an_grad._array)
        rel_error = abs_error / (np.abs(num_grad._array) + np.abs(an_grad._array) + 1e-10)
        
        # 检查误差是否在容限范围内
        if fast_mode:
            # 快速模式下只检查第一个元素
            idx = tuple([0] * num_grad._array.ndim)
            if abs_error[idx] > atol and rel_error[idx] > rtol:
                if raise_exception:
                    raise RuntimeError(f"Gradient check failed for input {i} (fast mode): "
                                      f"absolute error {abs_error[idx]:.6f} at {idx}, "
                                      f"relative error {rel_error[idx]:.6f} at {idx}")
                return False
        else:
            # 正常模式下检查所有元素
            if (abs_error > atol).any() and (rel_error > rtol).any():
                if raise_exception:
                    # 增强错误信息，显示最大误差位置
                    max_abs_error_idx = np.unravel_index(abs_error.argmax(), abs_error.shape)
                    max_rel_error_idx = np.unravel_index(rel_error.argmax(), rel_error.shape)
                    raise RuntimeError(f"Gradient check failed for input {i}: "
                                      f"max absolute error {abs_error.max():.6f} at {max_abs_error_idx}, "
                                      f"max relative error {rel_error.max():.6f} at {max_rel_error_idx}")
                return False
    
    return True
# end of gradcheck

class Function:
    """
    Function类是Riemann框架中用于自定义梯度的基类，参考PyTorch的torch.autograd.Function接口设计。
    
    要使用该类，需要继承并实现forward和backward静态方法：
    - forward：执行前向计算，返回输出张量
    - backward：接收输出梯度，返回输入梯度
    
    示例用法：
    class MyFunction(Function):
        @staticmethod
        def forward(ctx, input1, input2):
            ctx.save_for_backward(input1, input2)
            output = input1 * input2
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            input1, input2 = ctx.saved_tensors
            grad_input1 = grad_output * input2
            grad_input2 = grad_output * input1
            return grad_input1, grad_input2
    """
    
    class _Context:
        """
        上下文类，用于在forward和backward之间传递信息
        """
        def __init__(self):
            self.saved_tensors = ()
        
        def save_for_backward(self, *tensors):
            """保存张量，以便在backward中使用"""
            self.saved_tensors = tensors
    
    @staticmethod
    def forward(ctx, *inputs):
        """
        前向传播函数，需要被子类实现
        
        参数：
            ctx: 上下文对象，用于保存信息供backward使用
            *inputs: 输入张量
        
        返回：
            输出张量
        """
        raise NotImplementedError("forward() must be implemented in subclass")
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播函数，需要被子类实现
        
        参数：
            ctx: 上下文对象，包含forward中保存的信息
            grad_output: 输出张量的梯度
        
        返回：
            输入张量的梯度
        """
        raise NotImplementedError("backward() must be implemented in subclass")
    
    @classmethod
    def apply(cls, *inputs):
        """
        应用函数到输入张量，执行前向计算并设置梯度跟踪
        
        参数：
            *inputs: 输入张量
        
        返回：
            输出张量
        """
        ctx = cls._Context()
        
        # 执行前向计算
        outputs = cls.forward(ctx, *inputs)
        
        # 如果输出不是张量，将其转换为张量
        if not isinstance(outputs, TN):
            outputs = tensor(outputs,device=inputs[0].device)
        
        # 检查是否需要计算梯度
        requires_grad = any(input.requires_grad for input in inputs)
        
        if requires_grad:
            # 设置输出张量的梯度跟踪信息
            outputs.requires_grad = True
            
            # 收集所有需要跟踪的输入张量
            fromvars = tuple(input for input in inputs if input.requires_grad)
            
            # 创建梯度函数（闭包直接捕获cls、ctx、inputs变量）
            def _function_grad(result_tensor: TN, i: int) -> TN:
                
                # 执行反向传播
                grad_output = result_tensor.grad_value
                grad_inputs = cls.backward(ctx, grad_output)
                
                # 确保返回值是元组
                if not isinstance(grad_inputs, tuple):
                    grad_inputs = (grad_inputs,)
                
                # 找到当前梯度对应的原始输入索引
                grad_required_indices = [idx for idx, input in enumerate(inputs) if input.requires_grad]
                original_index = grad_required_indices[i]
                
                return grad_inputs[original_index]
            
            # 设置fromvars和gradfuncs
            outputs.fromvars = fromvars
            outputs.gradfuncs = tuple(_function_grad for _ in fromvars)
            outputs.parms = ()
            
        return outputs
# end of Function class