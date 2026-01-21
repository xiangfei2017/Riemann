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
Functional Gradient Computation Module for the Riemann Library

This module implements stateless gradient computation functions for automatic
differentiation, including:

- jacobian: Computes the Jacobian matrix of a function with respect to inputs
- hessian: Computes the Hessian matrix (gradient of gradients)
- derivative: Computes derivatives of scalar-output functions
- jvp: Computes Jacobian-vector products
- vjp: Computes vector-Jacobian products
- hvp: Computes Hessian-vector products
- vhp: Computes vector-Hessian products

All functions are designed to be compatible with PyTorch's API and support
the Riemann Tensor (TN) type with automatic differentiation capabilities.

Note: This module provides stateless gradient computation functions,
while the grad.py module provides the core grad function for computing
gradients of scalars or tensors with respect to other tensors.
"""

import numpy as np
from typing import Callable, Any, List, Tuple, Optional
import builtins
from ..tensordef import *
from .grad import *

def jacobian(func: Callable[..., TN | List[TN] | Tuple[TN, ...]], 
             inputs: TN | List[TN] | Tuple[TN, ...], 
             create_graph: bool = False, 
             strict: bool = True) -> TN | List[TN] | Tuple[TN, ...]:
    """计算函数的雅可比矩阵。

        该函数计算给定函数在输入点处的雅可比矩阵，支持单个或多个输入、
        单个或多个输出的情况，并与PyTorch的jacobian函数行为保持兼容。

        参数:
            func (Callable[..., TN | List[TN] | Tuple[TN, ...]]): 要计算雅可比矩阵的函数
            inputs (TN | List[TN] | Tuple[TN, ...]): 函数的输入张量或张量列表
            create_graph (bool, 可选): 是否在梯度计算中创建计算图，默认为False
            strict (bool, 可选): 是否严格遵循PyTorch的行为规范，默认为True

        返回:
            TN | List[TN] | Tuple[TN, ...]: 根据输入/输出类型返回相应结构的雅可比矩阵

        异常:
            RuntimeError: 当输入张量不可求导、输出不是张量等情况
    """
    
    # 处理输入为单个张量或张量列表的情况
    is_single_input = not isinstance(inputs, (list, tuple))
    if is_single_input:
        inputs = [inputs]  # type: ignore
    else:
        inputs = list(inputs)  # type: ignore
    
    # 验证输入是否为张量
    for i, inp in enumerate(inputs):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
    
    # Jacobin函数对输入张量是否requires_grad不做要求, 备份requires_grad
    bak_requires_grad = [inp.requires_grad for inp in inputs]
    inputs = [inp.requires_grad_(True) for inp in inputs]
    
    # 计算函数输出
    outputs = func(*inputs)
    
    # 处理输出为单个张量的情况
    is_single_output = not isinstance(outputs, (list, tuple))
    if is_single_output:
        outputs = [outputs]  # type: ignore
    else:
        outputs = list(outputs)  # type: ignore
    
    # 确保输出是张量
    for i, out in enumerate(outputs):
        if not isinstance(out, TN):
            raise RuntimeError(f"Output {i} must be a tensor")
    
    # 获取输入和输出形状
    input_shapes = [inp.shape for inp in inputs]
    output_shapes = [out.shape for out in outputs]
    
    # 获取总输出和总输入的元素数量
    # total_output_elements = builtins.sum(int(np.prod(shape)) for shape in output_shapes)
    total_input_elements = builtins.sum(int(np.prod(shape)) for shape in input_shapes)
    
    # 创建结果存储
    jacobian_results: List[TN] = []
    
    # 对每个输出计算梯度
    current_output_idx = 0
    for i, out in enumerate(outputs):
        num_outputs = int(np.prod(out.shape))
        
        # 统一使用TN张量操作方式实现，不再区分create_graph分支
        # 创建零张量作为雅可比矩阵的初始值 - 确保与out在同一设备上
        jac_shape = output_shapes[i] + (total_input_elements,)
        jac_tensor = zeros(*jac_shape, dtype=out.dtype, device=out.device)
        jac_tensor.requires_grad = create_graph
        
        # 对每个输出元素计算梯度
        for j in range(num_outputs):
            # 创建one-hot向量作为grad_outputs - 使用tensor操作
            grad_output = zeros_like(out)
            # 直接使用numpy的unravel_index，因为它只是一个索引计算函数，不涉及数据操作
            idx = np.unravel_index(j, out.shape)

            # 使用setat设置值
            grad_output = grad_output.setat(idx, 1.0)
            
            # 计算梯度 - 正确传递create_graph参数
            grads = grad(out, inputs, grad_outputs=grad_output, create_graph=create_graph, allow_unused = not strict)
            
            # 填充雅可比矩阵 - 使用非原地操作
            current_input_idx = 0
            for g in grads:
                if g is None:
                    continue
                grad_flat = g.reshape(-1)
                # 将j位置转换回多维索引
                jac_idx = idx + (slice(current_input_idx, current_input_idx + grad_flat.data.size),)
                
                # 直接使用setat操作更新jac_tensor，不再需要mask和temp_grad
                jac_tensor = jac_tensor.setat(jac_idx, grad_flat)
                
                current_input_idx += grad_flat.data.size
        
        jacobian_results.append(jac_tensor)
        current_output_idx += num_outputs
    # end of for i, out in enumerate(outputs)

    # 恢复输入的requires_grad状态
    for inp, req_grad in zip(inputs, bak_requires_grad):
        inp.requires_grad_(req_grad)

    # 返回与输入/输出格式匹配的结果，与PyTorch行为一致
    if is_single_output:
        # 单输出情况
        if is_single_input:
            # 1. 单输入单输出：返回单个张量
            return jacobian_results[0]
        else:
            # 3. 多输入单输出：返回与输入结构匹配的元组
            input_sizes = [int(np.prod(shape)) for shape in input_shapes]
            result_list: List[TN] = []
            start_idx = 0
            for size, shape in zip(input_sizes, input_shapes):
                # 提取每个输入对应的雅可比矩阵部分
                result_shape = output_shapes[0] + shape
                # 从完整雅可比矩阵中切分当前输入对应的部分
                sliced_jac = jacobian_results[0][..., start_idx:start_idx + size]
                # 重塑为正确的形状
                sliced_jac = sliced_jac.reshape(result_shape)
                result_list.append(sliced_jac)
                start_idx += size
            return tuple(result_list)
    else:
        # 多输出情况
        if is_single_input:
            # 2. 单输入多输出：返回与输出结构匹配的元组
            return tuple(jacobian_results)
        else:
            # 4. 多输入多输出：对于每个输出，返回与输入结构匹配的元组
            result_list: List[Tuple[TN, ...]] = []  # type: ignore
            input_sizes = [int(np.prod(shape)) for shape in input_shapes]
            
            for jac, output_shape in zip(jacobian_results, output_shapes):
                output_result_list: List[TN] = []
                start_idx = 0
                for size, inp_shape in zip(input_sizes, input_shapes):
                    # 提取每个输入对应的雅可比矩阵部分
                    result_shape = output_shape + inp_shape
                    # 从完整雅可比矩阵中切分当前输入对应的部分
                    sliced_jac = jac[..., start_idx:start_idx + size]
                    # 重塑为正确的形状
                    sliced_jac = sliced_jac.reshape(result_shape)
                    output_result_list.append(sliced_jac)
                    start_idx += size
                result_list.append(tuple(output_result_list))  # type: ignore
            return tuple(result_list)
# end of jacobian

def derivative(func: Callable[..., TN | List[TN] | Tuple[TN, ...]], 
               create_graph: bool = False) -> Callable[..., TN | List[TN] | Tuple[TN, ...]]:
    """计算函数的导数函数。

    该函数返回一个新函数，该新函数在调用时会计算原始函数func在输入点处的导数。
    支持func的输入为单个或多个张量，返回为单个或多个张量或标量。
    内部基于jacobian函数实现导数计算。

    参数:
        func (Callable[..., TN | List[TN] | Tuple[TN, ...] ]): 要求导的函数
        create_graph (bool, 可选): 是否在梯度计算中创建计算图，默认为False

    返回:
        Callable[..., TN | List[TN] | Tuple[TN, ...]]: 返回导函数，该函数接受与原函数相同的输入

    示例:
        >>> def f(x):
        ...     return x**2
        >>> df = derivative(f)
        >>> x = tensor([2.0])
        >>> df(x)  # 应返回 tensor([4.0])
    """
    
    def _derivative(*inputs: Any) -> TN | List[TN] | Tuple[TN, ...]:
        # 调用jacobian计算导数
        jac = jacobian(func, inputs, create_graph=create_graph)
        
        # 处理结果形状 - 对于标量输出，导数应该是雅可比矩阵的第一行
        # 检测原函数输出是否为标量
        try:
            sample_output = func(*inputs)
            # 检查是否是标量张量输出
            if isinstance(sample_output, TN) and len(sample_output.shape) == 0:
                is_scalar_tensor_output = True
            elif isinstance(sample_output, (list, tuple)) and all(
                isinstance(item, TN) and len(item.shape) == 0 for item in sample_output
            ):
                is_scalar_tensor_output = True
            else:
                is_scalar_tensor_output = False
            
            # 对于标量输出，需要特殊处理结果形状
            if is_scalar_tensor_output:
                # 单输入单输出情况
                if len(inputs) == 1 and not isinstance(sample_output, (list, tuple)):
                    # 对于标量输出，jacobian返回形状为 (1,) + input_shape 的张量
                    # 我们需要去除第一个维度
                    if isinstance(jac, TN) and jac.ndim > inputs[0].ndim:
                        return jac.squeeze(0)
                    return jac  # type: ignore
                # 多输入或多输出标量情况
                # 保持jacobian的结果结构，但可能需要调整形状
                return jac
            else:
                # 非纯标量输出，直接返回jacobian结果
                return jac
        except:
            # 如果无法确定输出类型，直接返回jacobian结果
            return jac
    # end of _derivative
    
    return _derivative
# end of derivative

def hessian(func: Callable[..., TN], 
           inputs: TN | List[TN] | Tuple[TN], 
           create_graph: bool = False,
           strict: bool = True) -> TN | List[TN] | Tuple[TN, ...]:
    """计算函数的Hessian矩阵。

        该函数计算给定函数在输入点处的Hessian矩阵，即函数梯度的雅可比矩阵。
        支持单个或多个输入的情况，并与PyTorch的hessian函数行为保持兼容。

        参数:
            func (Callable[..., TN]): 要计算Hessian矩阵的函数，该函数应返回一个标量张量
            inputs (TN | List[TN] | Tuple[TN]): 函数的输入张量或张量列表
            create_graph (bool, 可选): 是否在梯度计算中创建计算图，默认为False
            strict (bool, 可选): 如果为True，当检测到输出与某个输入无关时将引发错误，默认为True

        返回:
            TN | List[TN] | Tuple[TN, ...]: 根据输入类型返回相应结构的Hessian矩阵

        异常:
            RuntimeError: 当输入张量不可求导、输出不是标量等情况
    """
    # 处理输入为单个张量或张量列表的情况
    is_single_input = not isinstance(inputs, (list, tuple))
    if is_single_input:
        inputs = [inputs]  # type: ignore
    else:
        inputs = list(inputs)  # type: ignore
    
    # 验证输入是否张量
    for i, inp in enumerate(inputs):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
        
    # 定义一个辅助函数，用于计算梯度
    def gradient_fn(*args: TN) -> Tuple[TN, ...]:
        # 计算函数值
        output = func(*args)
        # 确保输出是标量
        if output.data.ndim > 0:
            raise RuntimeError("Hessian only supports scalar-valued functions")
        # 计算梯度
        grads = grad(output, args, create_graph=True, allow_unused = not strict)  # type: ignore
        return tuple(g if g is not None else tensor(0.0,device=output.device) for g in grads)
    
    # 计算Hessian矩阵，即梯度的雅可比矩阵 - 传递strict参数
    hessian_results = jacobian(gradient_fn, inputs, create_graph=create_graph, strict=strict)
    
    # 与PyTorch行为保持一致的结果处理
    # 1. 单输入情况 - 返回形状为(input_size, input_size)的张量
    if is_single_input:
        # 从嵌套结构中提取张量并调整形状
        if isinstance(hessian_results, (list, tuple)):
            if len(hessian_results) == 1 and isinstance(hessian_results[0], (list, tuple)):
                # 对于形如 ((tensor,),) 的结果
                hessian_tensor = hessian_results[0][0]  # type: ignore
            else:
                # 对于其他嵌套情况
                hessian_tensor = hessian_results[0]  # type: ignore
        else:
            hessian_tensor = hessian_results
        
        # 确保形状符合PyTorch的Hessian矩阵格式
        input_size = inputs[0].shape
        expected_shape = input_size + input_size
        if hessian_tensor.shape != expected_shape:
            # 重塑为正确的形状
            hessian_tensor = hessian_tensor.reshape(expected_shape)
        
        return hessian_tensor
    else:
        # 2. 多输入情况 - 与PyTorch保持一致的结构
        # 在这里我们保持jacobian返回的原始结构，因为它已经与PyTorch兼容
        return hessian_results
# end of hessian


# 重写jvp函数，优化精度并保持数学正确性
def jvp(func: Callable[..., TN | List[TN] | Tuple[TN, ...]],
        inputs: TN | List[TN] | Tuple[TN, ...],
        v: Optional[TN | List[TN] | Tuple[TN, ...]] = None,
        create_graph: bool = False,
        strict: bool = False) -> Tuple[TN | List[TN] | Tuple[TN, ...], TN | List[TN] | Tuple[TN, ...]]:
    """计算函数在给定点的雅可比向量积(JVP)。"""
    # 处理输入为单个张量或张量列表的情况
    is_single_input = not isinstance(inputs, (list, tuple))
    if is_single_input:
        inputs = [inputs]  # type: ignore
        # 如果v为None且只有单个输入，设置v为包含1的张量
        if v is None:
            v = [tensor(1.0, dtype=inputs[0].dtype, device=inputs[0].device, requires_grad=create_graph)]
        else:
            v = [v]  # type: ignore
    else:
        inputs = list(inputs)  # type: ignore
        # 如果v为None但有多个输入，抛出错误
        if v is None:
            raise RuntimeError("v must be provided when func has multiple inputs")
        v = list(v)  # type: ignore
    
    # 验证输入张量和v的形状匹配
    for i, (inp, vec) in enumerate(zip(inputs, v)):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
        if not isinstance(vec, TN):
            raise RuntimeError(f"v[{i}] must be a tensor")
        if inp.shape != vec.shape:
            raise RuntimeError(f"Shapes of input {i} and v[{i}] do not match: {inp.shape} vs {vec.shape}")
    
    # 对输入张量是否requires_grad不做要求, 备份requires_grad
    bak_requires_grad = [inp.requires_grad for inp in inputs]
    inputs = [inp.requires_grad_(True) for inp in inputs]
    
    # 计算函数输出
    outputs = func(*inputs)
    
    # 处理输出为单个张量或张量列表的情况
    is_single_output = not isinstance(outputs, (list, tuple))
    if is_single_output:
        outputs_processed = [outputs]
    else:
        outputs_processed = list(outputs)  # type: ignore
    
    # 确保输出是张量
    for i, out in enumerate(outputs_processed):
        if not isinstance(out, TN):
            raise RuntimeError(f"Output {i} must be a tensor")
    
    # 计算JVP结果，对每个输出单独计算JVP
    jvp_results: List[TN] = []
    
    for out in outputs_processed:
        # 对于标量输出，使用点积计算JVP
        if out.ndim == 0:  # type: ignore
            # 优化：直接使用向量点积公式，避免中间变量累积误差
            total_jvp: TN = tensor(0.0, dtype=out.dtype, device=out.device, requires_grad=create_graph)  # type: ignore
            for i, (inp, vec) in enumerate(zip(inputs, v)):
                # 计算标量输出相对于输入的梯度
                grads = grad(out, inp, create_graph=create_graph, allow_unused=True)[0]  # type: ignore
                
                if strict and grads is None:
                    raise RuntimeError(f"At least one of the outputs is independent of input {i}")
                
                if grads is not None:
                    # 对于标量输出，正确的JVP计算是梯度与v向量的点积
                    if create_graph:
                        # 如果需要创建计算图，使用PyTorch风格的点积计算
                        dot_product = (grads * vec).sum()
                    else:
                        # 使用更精确的方式计算点积，避免中间张量创建
                        dot_product = (grads.data * vec.data).sum()
                    
                    total_jvp = total_jvp + dot_product
            jvp_val = total_jvp
        else:
            # 对于非标量输出，采用更精确的方法计算JVP
            # 对于每个输入，计算其对输出的贡献并累加
            jvp_val = zeros_like(out, requires_grad=create_graph)

            # 对每个输入分别计算其对输出的贡献
            for i, (inp, vec) in enumerate(zip(inputs, v)):
                # 为每个输入计算JVP贡献
                # 使用更精确的方式：直接计算雅可比矩阵与向量的乘积
                if create_graph:
                    # 创建一个与输出形状相同的零张量用于累加贡献
                    input_contribution = zeros_like(out, requires_grad=True)

                    # 对于每个输出元素，计算该输入对其的贡献
                    for j in range(out.data.size):  # type: ignore
                        # 创建one-hot向量作为grad_outputs
                        grad_output = zeros_like(out)
                        flat_idx = np.unravel_index(j, out.shape)  # type: ignore
                        grad_output[flat_idx] = 1.0
                        
                        # 计算梯度
                        grads = grad(out, inp, grad_outputs=grad_output, create_graph=create_graph, allow_unused=True)[0]  # type: ignore
                        
                        if strict and grads is None:
                            raise RuntimeError(f"At least one of the outputs is independent of input {i}")
                        
                        if grads is not None:
                            # 计算当前元素的贡献
                            contribution = (grads * vec).sum()
                            
                            # 创建掩码并更新贡献张量
                            mask = zeros_like(out)  # type: ignore
                            mask[flat_idx] = 1.0
                            input_contribution = input_contribution * (1.0 - mask) + contribution * mask
                else:
                    # 不创建计算图时的优化版本
                    input_contribution = zeros_like(out)  # type: ignore
                    
                    # 对于每个输出元素，计算该输入对其的贡献
                    for j in range(out.data.size):  # type: ignore
                        # 创建one-hot向量作为grad_outputs
                        grad_output = zeros_like(out)  # type: ignore
                        flat_idx = np.unravel_index(j, out.shape)  # type: ignore
                        grad_output[flat_idx] = 1.0
                        
                        # 计算梯度
                        grads = grad(out, inp, grad_outputs=grad_output, create_graph=False, allow_unused=True)[0]  # type: ignore
                        
                        if strict and grads is None:
                            raise RuntimeError(f"At least one of the outputs is independent of input {i}")
                        
                        if grads is not None:
                            # 直接计算贡献并设置到对应位置
                            contribution = (grads.data * vec.data).sum()
                            input_contribution.data[flat_idx] = contribution
                
                # 累加到总JVP结果中
                jvp_val = jvp_val + input_contribution
        
        jvp_results.append(jvp_val)
    
    # 恢复原始的输入requires_grad状态
    for inp, req_grad in zip(inputs, bak_requires_grad):
        inp.requires_grad_(req_grad)

    # 恢复原始的输出格式
    if is_single_output:
        original_outputs = outputs  # type: ignore
        original_jvp = jvp_results[0]
    else:
        original_outputs = tuple(outputs)  # type: ignore
        original_jvp = tuple(jvp_results)  # type: ignore
    
    return original_outputs, original_jvp

# 修改后的vjp函数，与PyTorch官方接口保持一致
def vjp(func: Callable[..., TN | List[TN] | Tuple[TN, ...]],
        inputs: TN | List[TN] | Tuple[TN, ...],
        v: Optional[TN | List[TN] | Tuple[TN, ...]] = None,
        create_graph: bool = False,
        strict: bool = False) -> Tuple[TN | List[TN] | Tuple[TN, ...], TN | List[TN] | Tuple[TN, ...]]:
    """计算给定向量v与函数在给定点处的雅可比矩阵的点积。

    计算函数func在inputs点处的向量雅可比积(VJP)。

    参数:
        func (function): 接受Tensor输入并返回Tensor或Tensor元组的Python函数。
        inputs (Tensor或Tensor元组): 函数func的输入。
        v (Tensor或Tensor元组，可选): 用于计算向量雅可比积的向量。
            必须与func的输出大小相同。当func的输出只包含单个元素时，此参数是可选的，
            如果不提供，将设置为包含单个1的Tensor。
        create_graph (bool, 可选): 如果为True，输出和结果都将以可微方式计算。
            请注意，当strict为False时，结果不能需要梯度或与输入断开连接。默认为False。
        strict (bool, 可选): 如果为True，当检测到存在所有输出都与其无关的输入时，将引发错误。
            如果为False，我们为这些输入返回零张量作为vjp，这是预期的数学值。默认为False。

    返回:
        Tuple[TN | List[TN] | Tuple[TN, ...], TN | List[TN] | Tuple[TN, ...]]:
            - 第一个元素是函数的输出
            - 第二个元素是向量雅可比积结果

    异常:
        RuntimeError: 当输入张量不可求导等情况
    """
    # 处理输入为单个张量或张量列表的情况
    is_single_input = not isinstance(inputs, (list, tuple))
    if is_single_input:
        inputs = [inputs]  # type: ignore
    else:
        inputs = list(inputs)  # type: ignore
    
    # 验证输入张量
    for i, inp in enumerate(inputs):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
    
    # 对输入张量是否requires_grad不做要求, 备份requires_grad
    bak_requires_grad = [inp.requires_grad for inp in inputs]
    inputs = [inp.requires_grad_(True) for inp in inputs]
    
    # 计算函数输出
    outputs = func(*inputs)
    
    # 处理输出为单个张量的情况
    is_single_output = not isinstance(outputs, (list, tuple))
    if is_single_output:
        outputs_processed = [outputs]
    else:
        outputs_processed = list(outputs)  # type: ignore
    
    # 确保输出是张量
    for i, out in enumerate(outputs_processed):
        if not isinstance(out, TN):
            raise RuntimeError(f"Output {i} must be a tensor")
    
    # 处理v参数
    if v is None:
        # 检查是否只有单个输出元素
        if is_single_output and outputs_processed[0].ndim == 0:  # type: ignore
            # 输出是标量，创建值为1的Tensor
            v = [tensor(1.0, 
                 dtype=outputs_processed[0].dtype, 
                 device=outputs_processed[0].device, 
                 requires_grad=create_graph)]  # type: ignore
        else:
            raise RuntimeError("v must be provided when the output of func does not contain a single element")
    else:
        # 处理v为单个张量或张量列表的情况
        is_single_v = not isinstance(v, (list, tuple))
        if is_single_v:
            if len(outputs_processed) != 1:
                raise RuntimeError(f"Expected a list of vs with length {len(outputs_processed)}, got a single v")
            v = [v]  # type: ignore
        else:
            v = list(v)  # type: ignore
        
        # 验证v的数量与输出数量匹配
        if len(v) != len(outputs_processed):
            raise RuntimeError(f"Expected {len(outputs_processed)} vs, got {len(v)}")
        
        # 验证v的形状与输出形状匹配
        for i, (out, vec) in enumerate(zip(outputs_processed, v)):
            if not isinstance(vec, TN):
                raise RuntimeError(f"v[{i}] must be a tensor")
            if out.shape != vec.shape:  # type: ignore
                raise RuntimeError(f"Shapes of output {i} and v[{i}] do not match: {out.shape} vs {vec.shape}")  # type: ignore
    
    # 计算VJP，对每个输出单独计算梯度然后相加
    vjp_results = [zeros_like(inp, requires_grad=create_graph) for inp in inputs]

    for i, (out, vec) in enumerate(zip(outputs_processed, v)):
        # 对每个输出使用对应的v计算梯度
        grads = grad(out, inputs, grad_outputs=vec, create_graph=create_graph, allow_unused=True)  # type: ignore
        
        # 处理strict模式
        if strict:
            for j, grad_out in enumerate(grads):
                if grad_out is None:
                    raise RuntimeError(f"At least one of the outputs is independent of input {j}")
        
        # 累加梯度
        for j in range(len(inputs)):
            if grads[j] is not None:
                vjp_results[j] = vjp_results[j] + grads[j]
            # 对于None的梯度，保持为零张量
    
    # 恢复原始的输入requires_grad状态
    for inp, req_grad in zip(inputs, bak_requires_grad):
        inp.requires_grad_(req_grad)

    # 恢复原始的输入格式
    if is_single_input:
        vjp_result = vjp_results[0]
    else:
        vjp_result = tuple(vjp_results)  # type: ignore
    
    # 恢复原始的输出格式
    if is_single_output:
        func_output = outputs  # type: ignore
    else:
        func_output = tuple(outputs)  # type: ignore
    
    return func_output, vjp_result


# 添加在vjp函数后
def _compute_hessian_vector_product(func: Callable[..., TN], inputs: TN | List[TN] | Tuple[TN, ...], v: TN | List[TN] | Tuple[TN, ...], create_graph: bool = False, strict: bool = False) -> Tuple[TN, TN | List[TN] | Tuple[TN, ...]]:
    """
    计算Hessian-vector乘积 (hvp) 或 vector-Hessian乘积 (vhp) 的核心函数。
    
    利用反向自动微分高效计算，无需显式构建完整Hessian矩阵。
    
    参数:
        func: 输入为张量或张量列表，输出为标量张量的函数
        inputs: 函数输入，张量或张量列表
        v: 要与Hessian相乘的向量，形状与inputs匹配
        create_graph (bool, 可选): 如果为True，计算图将被保留以支持高阶导数
        strict (bool, 可选): 与grad函数中的strict参数相同
    
    返回:
        Tuple[TN, Tensor或Tensor列表]: 
            - 第一个元素是函数的输出值
            - 第二个元素是HVP/VHP结果，形状与inputs匹配
    """
    # 处理输入为单个张量或张量列表的情况
    is_single_input = not isinstance(inputs, (list, tuple))
    if is_single_input:
        inputs = [inputs]  # type: ignore
        v = [v]  # type: ignore
    else:
        inputs = list(inputs)  # type: ignore
        v = list(v)  # type: ignore
    
    # 验证输入和v的类型与形状
    for i, (inp, vec) in enumerate(zip(inputs, v)):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
        if not isinstance(vec, TN):
            raise RuntimeError(f"v[{i}] must be a tensor")
        if inp.shape != vec.shape:
            raise RuntimeError(f"Shapes of input {i} and v[{i}] do not match: {inp.shape} vs {vec.shape}")
    
    # 对输入张量是否requires_grad不做要求, 备份requires_grad
    bak_requires_grad = [inp.requires_grad for inp in inputs]
    inputs = [inp.requires_grad_(True) for inp in inputs]
    
    # 计算函数值
    func_output = func(*inputs)
    if not isinstance(func_output, TN) or func_output.ndim != 0:
        raise RuntimeError("func must return a scalar tensor")
    
    # 计算与v的内积
    grads = grad(func_output, inputs, create_graph=True, allow_unused = not strict)  # 必须创建计算图以支持二阶导数
    
    # 处理strict模式
    if strict:
        for i, grad_out in enumerate(grads):
            if grad_out is None:
                raise RuntimeError(f"Output is independent of input {i}")
    
    # 计算内积
    dot_product: TN = tensor(0.0, dtype=func_output.dtype, device=func_output.device, requires_grad=True)
    for grad_out, vec in zip(grads, v):
        if grad_out is not None:
            dot_product = dot_product + (grad_out * vec).sum()
    
    # 计算二阶导数 (Hessian-vector product)
    hessian_vec_product = grad(dot_product, inputs, create_graph=create_graph, allow_unused = not strict)
    
    # 恢复原始的输入requires_grad状态
    for inp, req_grad in zip(inputs, bak_requires_grad):
        inp.requires_grad_(req_grad)

    # 恢复原始格式
    if is_single_input:
        hessian_vec_product = hessian_vec_product[0]  # type: ignore
    else:
        hessian_vec_product = tuple(hessian_vec_product)
    
    # 返回函数值和HVP/VHP结果的元组，与PyTorch行为一致
    return func_output, hessian_vec_product

def hvp(func: Callable[..., TN], inputs: TN | List[TN] | Tuple[TN, ...], v: TN | List[TN] | Tuple[TN, ...], create_graph: bool = False, strict: bool = False) -> Tuple[TN, TN | List[TN] | Tuple[TN, ...]]:
    """
    计算Hessian-vector product (HVP): H @ v
    
    Hessian矩阵H是函数f对输入的二阶导数，此函数计算Hessian矩阵与向量v的乘积，
    无需显式构建完整Hessian矩阵。
    
    参数:
        func: 输入为张量或张量列表，输出为标量张量的函数
        inputs: 函数输入，张量或张量列表
        v: 要与Hessian相乘的向量，形状与inputs匹配
        create_graph (bool, 可选): 如果为True，计算图将被保留以支持高阶导数
        strict (bool, 可选): 如果为True，当检测到输出与某个输入无关时将引发错误
    
    返回:
        Tuple[TN, Tensor或Tensor列表]: 
            - 第一个元素是函数的输出值
            - 第二个元素是VHP结果，形状与inputs匹配
    """
    # 由于Hessian矩阵是对称的，HVP和VHP实际上是相同的
    # 直接调用核心函数计算
    return _compute_hessian_vector_product(func, inputs, v, create_graph, strict)

def vhp(func: Callable[..., TN], inputs: TN | List[TN] | Tuple[TN, ...], v: TN | List[TN] | Tuple[TN, ...], create_graph: bool = False, strict: bool = False) -> Tuple[TN, TN | List[TN] | Tuple[TN, ...]]:
    """
    计算vector-Hessian product (VHP): v @ H
    
    Hessian矩阵H是函数f对输入的二阶导数，此函数计算向量v与Hessian矩阵的乘积，
    无需显式构建完整Hessian矩阵。
    
    由于Hessian矩阵的对称性，vhp和hvp的结果在数值上是相同的。
    
    参数:
        func: 输入为张量或张量列表，输出为标量张量的函数
        inputs: 函数输入，张量或张量列表
        v: 要与Hessian相乘的向量，形状与inputs匹配
        create_graph (bool, 可选): 如果为True，计算图将被保留以支持高阶导数
        strict (bool, 可选): 如果为True，当检测到输出与某个输入无关时将引发错误
    
    返回:
        Tuple[TN, Tensor或Tensor列表]: 
            - 第一个元素是函数的输出值
            - 第二个元素是VHP结果，形状与inputs匹配
    """
    
    return hvp(func, inputs, v, create_graph, strict)
