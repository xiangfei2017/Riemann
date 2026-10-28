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
from typing import Callable, Any
import builtins
from ..tensordef import *
from .grad import *

def _split_jac_by_inputs(jac: TN, output_shape: tuple, input_sizes: list[int], input_shapes: list[tuple]) -> tuple[TN, ...]:
    """将雅可比矩阵沿输入维度按各输入尺寸切分并重塑形状"""
    result = []
    start = 0
    for size, shape in zip(input_sizes, input_shapes):
        result.append(jac[..., start:start + size]._reshape(output_shape + shape))
        start += size
    return tuple(result)

def jacobian(func: Callable[..., TN | list[TN] | tuple[TN, ...]],
             inputs: TN | list[TN] | tuple[TN, ...],
             create_graph: bool = False,
             strict: bool = True) -> TN | list[TN] | tuple[TN, ...]:
    """计算函数的雅可比矩阵。

        该函数计算给定函数在输入点处的雅可比矩阵，支持单个或多个输入、
        单个或多个输出的情况，并与PyTorch的jacobian函数行为保持兼容。

        参数:
            func (Callable[..., TN | list[TN] | tuple[TN, ...]]): 要计算雅可比矩阵的函数
            inputs (TN | list[TN] | tuple[TN, ...]): 函数的输入张量或张量列表
            create_graph (bool, 可选): 是否在梯度计算中创建计算图，默认为False
            strict (bool, 可选): 是否严格遵循PyTorch的行为规范，默认为True

        返回:
            TN | list[TN] | tuple[TN, ...]: 根据输入/输出类型返回相应结构的雅可比矩阵

        异常:
            RuntimeError: 当输入张量不可求导、输出不是张量等情况
    """
    is_single_input = not isinstance(inputs, (list, tuple))
    if is_single_input:
        inputs = [inputs]  # type: ignore
    else:
        inputs = list(inputs)  # type: ignore

    for i, inp in enumerate(inputs):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")

    bak_requires_grad = [inp.requires_grad for inp in inputs]
    inputs = [inp.requires_grad_(True) for inp in inputs]

    outputs = func(*inputs)

    is_single_output = not isinstance(outputs, (list, tuple))
    if is_single_output:
        outputs = [outputs]  # type: ignore
    else:
        outputs = list(outputs)  # type: ignore

    for i, out in enumerate(outputs):
        if not isinstance(out, TN):
            raise RuntimeError(f"Output {i} must be a tensor")

    input_sizes = [int(np.prod(inp.shape)) for inp in inputs]
    input_shapes = [inp.shape for inp in inputs]
    total_input_elements = builtins.sum(input_sizes)

    jacobian_results: list[TN] = []

    # 逐输出计算雅可比：对每个输出元素做一次反向传播，得到雅可比的一行
    for out in outputs:
        num_output_elements = int(np.prod(out.shape))

        # jac_tensor shape = output_shape + (total_input_elements,)
        # 每一行对应一个输出元素对所有输入的偏导
        jac_tensor = zeros(*(out.shape + (total_input_elements,)), dtype=out.dtype, device=out.device)
        jac_tensor.requires_grad = create_graph

        base_grad_output = zeros_like(out)
        all_indices = [np.unravel_index(j, out.shape) for j in range(num_output_elements)]

        for idx in all_indices:
            # 构造 one-hot 向量，选中第 idx 个输出元素
            grad_output = base_grad_output.setat(idx, 1.0)
            # 反向传播：grads[k] = ∂out[idx] / ∂inputs[k]
            grads = grad(out, inputs, grad_outputs=grad_output, create_graph=create_graph, allow_unused=not strict)

            # 将各输入的梯度按偏移量填入 jac_tensor 的对应列
            current_input_idx = 0
            for k, g in enumerate(grads):
                if g is not None:
                    grad_flat = g.reshape(-1)
                    jac_idx = idx + (slice(current_input_idx, current_input_idx + input_sizes[k]),)
                    jac_tensor = jac_tensor.setat(jac_idx, grad_flat)
                current_input_idx += input_sizes[k]

        jacobian_results.append(jac_tensor)

    for inp, req_grad in zip(inputs, bak_requires_grad):
        inp.requires_grad_(req_grad)

    # 返回结构：
    # - 单输出单输入 → TN
    # - 单输出多输入 → tuple[TN]（按输入切分）
    # - 多输出单输入 → tuple[TN]
    # - 多输出多输入 → tuple[tuple[TN, ...]]
    if is_single_output:
        if is_single_input:
            return jacobian_results[0]
        else:
            return _split_jac_by_inputs(jacobian_results[0], outputs[0].shape, input_sizes, input_shapes)
    else:
        if is_single_input:
            return tuple(jacobian_results)
        else:
            return tuple(
                _split_jac_by_inputs(jac, out.shape, input_sizes, input_shapes)
                for jac, out in zip(jacobian_results, outputs)
            )

def derivative(func: Callable[..., TN | list[TN] | tuple[TN, ...]],
               create_graph: bool = False) -> Callable[..., TN | list[TN] | tuple[TN, ...]]:
    """计算函数的导数函数。

    该函数返回一个新函数，该新函数在调用时会计算原始函数func在输入点处的导数。
    支持func的输入为单个或多个张量，返回为单个或多个张量或标量。
    内部基于jacobian函数实现导数计算。

    参数:
        func (Callable[..., TN | list[TN] | tuple[TN, ...] ]): 要求导的函数
        create_graph (bool, 可选): 是否在梯度计算中创建计算图，默认为False

    返回:
        Callable[..., TN | list[TN] | tuple[TN, ...]]: 返回导函数，该函数接受与原函数相同的输入

    示例:
        >>> def f(x):
        ...     return x**2
        >>> df = derivative(f)
        >>> x = tensor([2.0])
        >>> df(x)  # 应返回 (tensor([4.0]),)
    """

    def _derivative(*inputs: Any) -> TN | list[TN] | tuple[TN, ...]:
        jac = jacobian(func, inputs, create_graph=create_graph)
        # 雅可比形状 = output_shape + (input_size,)
        # 标量输出时 output_shape=()，jac.ndim=1
        # 标量输入时 input.ndim=0，jac.shape=(1,)，需 squeeze 到 ()
        if isinstance(jac, TN) and jac.ndim == 1 and inputs[0].ndim == 0:
            return jac.squeeze(0)
        return jac

    return _derivative

def hessian(func: Callable[..., TN],
           inputs: TN | list[TN] | tuple[TN],
           create_graph: bool = False,
           strict: bool = True) -> TN | list[TN] | tuple[TN, ...]:
    """计算函数的Hessian矩阵。

        该函数计算给定函数在输入点处的Hessian矩阵，即函数梯度的雅可比矩阵。
        支持单个或多个输入的情况，并与PyTorch的hessian函数行为保持兼容。

        参数:
            func (Callable[..., TN]): 要计算Hessian矩阵的函数，该函数应返回一个标量张量
            inputs (TN | list[TN] | tuple[TN]): 函数的输入张量或张量列表
            create_graph (bool, 可选): 是否在梯度计算中创建计算图，默认为False
            strict (bool, 可选): 如果为True，当检测到输出与某个输入无关时将引发错误，默认为True

        返回:
            TN | list[TN] | tuple[TN, ...]: 根据输入类型返回相应结构的Hessian矩阵

        异常:
            RuntimeError: 当输入张量不可求导、输出不是标量等情况
    """
    is_single_input = not isinstance(inputs, (list, tuple))
    if is_single_input:
        inputs = [inputs]  # type: ignore
    else:
        inputs = list(inputs)  # type: ignore

    for i, inp in enumerate(inputs):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")

    def gradient_fn(*args: TN) -> tuple[TN, ...]:
        output = func(*args)
        if output._array.ndim > 0:
            raise RuntimeError("Hessian only supports scalar-valued functions")
        grads = grad(output, args, create_graph=True, allow_unused=not strict)  # type: ignore
        return tuple(g if g is not None else tensor(0.0, device=output.device, requires_grad=True) for g in grads)

    # Hessian = Jacobian(gradient)，即对梯度函数求雅可比
    hessian_results = jacobian(gradient_fn, inputs, create_graph=create_graph, strict=strict)

    if is_single_input:
        # jacobian 接收 list 输入 → is_single_input=False
        # gradient_fn 返回 tuple → is_single_output=False
        # 故 jacobian 走多输入多输出路径，返回嵌套 tuple ((jac_tensor,),)
        hessian_tensor = hessian_results[0]
        if isinstance(hessian_tensor, (list, tuple)):
            hessian_tensor = hessian_tensor[0]  # type: ignore
        # 高维输入时雅可比形状需重塑：jac_shape = input_shape + (input_numel,)
        # 期望 Hessian 形状 = input_shape + input_shape
        expected_shape = inputs[0].shape + inputs[0].shape
        if hessian_tensor.shape != expected_shape:
            hessian_tensor = hessian_tensor._reshape(expected_shape)
        return hessian_tensor
    else:
        return hessian_results

def jvp(func: Callable[..., TN | list[TN] | tuple[TN, ...]],
        inputs: TN | list[TN] | tuple[TN, ...],
        v: TN | list[TN] | tuple[TN, ...] | None = None,
        create_graph: bool = False,
        strict: bool = False) -> tuple[TN | list[TN] | tuple[TN, ...], TN | list[TN] | tuple[TN, ...]]:
    """计算函数在给定点的雅可比向量积(JVP)。"""
    is_single_input = not isinstance(inputs, (list, tuple))
    if is_single_input:
        inputs = [inputs]  # type: ignore
        if v is None:
            v = [tensor(1.0, dtype=inputs[0].dtype, device=inputs[0].device, requires_grad=create_graph)]
        else:
            v = [v]  # type: ignore
    else:
        inputs = list(inputs)  # type: ignore
        if v is None:
            raise RuntimeError("v must be provided when func has multiple inputs")
        v = list(v)  # type: ignore

    for i, (inp, vec) in enumerate(zip(inputs, v)):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
        if not isinstance(vec, TN):
            raise RuntimeError(f"v[{i}] must be a tensor")
        if inp.shape != vec.shape:
            raise RuntimeError(f"Shapes of input {i} and v[{i}] do not match: {inp.shape} vs {vec.shape}")

    bak_requires_grad = [inp.requires_grad for inp in inputs]
    inputs = [inp.requires_grad_(True) for inp in inputs]

    outputs = func(*inputs)

    is_single_output = not isinstance(outputs, (list, tuple))
    if is_single_output:
        outputs = [outputs]  # type: ignore
    else:
        outputs = list(outputs)  # type: ignore

    for i, out in enumerate(outputs):
        if not isinstance(out, TN):
            raise RuntimeError(f"Output {i} must be a tensor")

    jvp_results: list[TN] = []

    for out in outputs:
        if out.ndim == 0:
            # 标量输出：JVP = J @ v = sum_i(grad_i · v_i)
            # 一次 grad 调用即可获得所有输入的梯度
            all_grads = grad(out, inputs, create_graph=create_graph, allow_unused=True)
            jvp_val = tensor(0.0, dtype=out.dtype, device=out.device, requires_grad=create_graph)
            for j, (g, vec) in enumerate(zip(all_grads, v)):
                if g is not None:
                    if create_graph:
                        jvp_val = jvp_val + (g * vec).sum()
                    else:
                        jvp_val = jvp_val + (g._array * vec._array).sum()
                elif strict:
                    raise RuntimeError(f"At least one of the outputs is independent of input {j}")
        else:
            # 非标量输出：JVP[j] = sum_i(J[j,i] * v[i]) = sum_i(grad(out_j, inp_i) · v_i)
            # 需逐输出元素做反向传播（每个元素对应雅可比的一行）
            jvp_val = zeros_like(out, requires_grad=create_graph)
            base_grad_output = zeros_like(out)
            all_indices = [np.unravel_index(j, out.shape) for j in range(out._array.size)]  # type: ignore

            for idx in all_indices:
                # one-hot 向量选中第 idx 个输出元素
                grad_output = base_grad_output.setat(idx, 1.0)
                # 一次 grad 调用获得所有输入对该输出元素的梯度
                all_grads = grad(out, inputs, grad_outputs=grad_output, create_graph=create_graph, allow_unused=True)

                if create_graph:
                    contribution = tensor(0.0, dtype=out.dtype, device=out.device, requires_grad=True)
                    for j, (g, vec) in enumerate(zip(all_grads, v)):
                        if g is not None:
                            contribution = contribution + (g * vec).sum()
                        elif strict:
                            raise RuntimeError(f"At least one of the outputs is independent of input {j}")
                    jvp_val = jvp_val.setat(idx, contribution)
                else:
                    contribution = 0.0
                    for j, (g, vec) in enumerate(zip(all_grads, v)):
                        if g is not None:
                            contribution += (g._array * vec._array).sum()
                        elif strict:
                            raise RuntimeError(f"At least one of the outputs is independent of input {j}")
                    jvp_val._array[idx] = contribution

        jvp_results.append(jvp_val)

    for inp, req_grad in zip(inputs, bak_requires_grad):
        inp.requires_grad_(req_grad)

    if is_single_output:
        return outputs[0], jvp_results[0]
    else:
        return tuple(outputs), tuple(jvp_results)

def vjp(func: Callable[..., TN | list[TN] | tuple[TN, ...]],
        inputs: TN | list[TN] | tuple[TN, ...],
        v: TN | list[TN] | tuple[TN, ...] | None = None,
        create_graph: bool = False,
        strict: bool = False) -> tuple[TN | list[TN] | tuple[TN, ...], TN | list[TN] | tuple[TN, ...]]:
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
        tuple[TN | list[TN] | tuple[TN, ...], TN | list[TN] | tuple[TN, ...]]:
            - 第一个元素是函数的输出
            - 第二个元素是向量雅可比积结果

    异常:
        RuntimeError: 当输入张量不可求导等情况
    """
    is_single_input = not isinstance(inputs, (list, tuple))
    if is_single_input:
        inputs = [inputs]  # type: ignore
    else:
        inputs = list(inputs)  # type: ignore

    for i, inp in enumerate(inputs):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")

    bak_requires_grad = [inp.requires_grad for inp in inputs]
    inputs = [inp.requires_grad_(True) for inp in inputs]

    outputs = func(*inputs)

    is_single_output = not isinstance(outputs, (list, tuple))
    if is_single_output:
        outputs = [outputs]  # type: ignore
    else:
        outputs = list(outputs)  # type: ignore

    for i, out in enumerate(outputs):
        if not isinstance(out, TN):
            raise RuntimeError(f"Output {i} must be a tensor")

    # 处理v参数
    if v is None:
        if is_single_output and outputs[0].ndim == 0:  # type: ignore
            v = [tensor(1.0, dtype=outputs[0].dtype, device=outputs[0].device, requires_grad=create_graph)]  # type: ignore
        else:
            raise RuntimeError("v must be provided when the output of func does not contain a single element")
    else:
        is_single_v = not isinstance(v, (list, tuple))
        if is_single_v:
            if len(outputs) != 1:
                raise RuntimeError(f"Expected a list of vs with length {len(outputs)}, got a single v")
            v = [v]  # type: ignore
        else:
            v = list(v)  # type: ignore

        if len(v) != len(outputs):
            raise RuntimeError(f"Expected {len(outputs)} vs, got {len(v)}")

        for i, (out, vec) in enumerate(zip(outputs, v)):
            if not isinstance(vec, TN):
                raise RuntimeError(f"v[{i}] must be a tensor")
            if out.shape != vec.shape:  # type: ignore
                raise RuntimeError(f"Shapes of output {i} and v[{i}] do not match: {out.shape} vs {vec.shape}")

    # 计算VJP：对每个输出使用对应的v计算梯度并累加
    vjp_results = [zeros_like(inp, requires_grad=create_graph) for inp in inputs]

    for out, vec in zip(outputs, v):
        grads = grad(out, inputs, grad_outputs=vec, create_graph=create_graph, allow_unused=True)  # type: ignore

        if strict:
            for j, grad_out in enumerate(grads):
                if grad_out is None:
                    raise RuntimeError(f"At least one of the outputs is independent of input {j}")

        for j in range(len(inputs)):
            if grads[j] is not None:
                vjp_results[j] = vjp_results[j] + grads[j]

    for inp, req_grad in zip(inputs, bak_requires_grad):
        inp.requires_grad_(req_grad)

    if is_single_input:
        vjp_result = vjp_results[0]
    else:
        vjp_result = tuple(vjp_results)  # type: ignore

    if is_single_output:
        return outputs[0], vjp_result
    else:
        return tuple(outputs), vjp_result


def _compute_hessian_vector_product(func: Callable[..., TN], inputs: TN | list[TN] | tuple[TN, ...], v: TN | list[TN] | tuple[TN, ...], create_graph: bool = False, strict: bool = False) -> tuple[TN, TN | list[TN] | tuple[TN, ...]]:
    """计算Hessian-vector乘积 (hvp/vhp) 的核心函数。

    利用双重反向自动微分高效计算，无需显式构建完整Hessian矩阵。

    参数:
        func: 输入为张量或张量列表，输出为标量张量的函数
        inputs: 函数输入，张量或张量列表
        v: 要与Hessian相乘的向量，形状与inputs匹配
        create_graph (bool, 可选): 如果为True，计算图将被保留以支持高阶导数
        strict (bool, 可选): 与grad函数中的strict参数相同

    返回:
        tuple[TN, TN | list[TN] | tuple[TN, ...]]:
            - 第一个元素是函数的输出值
            - 第二个元素是HVP/VHP结果，形状与inputs匹配
    """
    is_single_input = not isinstance(inputs, (list, tuple))
    if is_single_input:
        inputs = [inputs]  # type: ignore
        v = [v]  # type: ignore
    else:
        inputs = list(inputs)  # type: ignore
        v = list(v)  # type: ignore

    for i, (inp, vec) in enumerate(zip(inputs, v)):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
        if not isinstance(vec, TN):
            raise RuntimeError(f"v[{i}] must be a tensor")
        if inp.shape != vec.shape:
            raise RuntimeError(f"Shapes of input {i} and v[{i}] do not match: {inp.shape} vs {vec.shape}")

    bak_requires_grad = [inp.requires_grad for inp in inputs]
    inputs = [inp.requires_grad_(True) for inp in inputs]

    func_output = func(*inputs)
    if not isinstance(func_output, TN) or func_output.ndim != 0:
        raise RuntimeError("func must return a scalar tensor")

    # 第一次反向：计算梯度（保留计算图以支持二阶导数）
    grads = grad(func_output, inputs, create_graph=True, allow_unused=True)

    if strict:
        for i, grad_out in enumerate(grads):
            if grad_out is None:
                raise RuntimeError(f"The output of the user-provided function is independent of input {i}. This is not allowed in strict mode.")

    # 计算梯度与v的内积
    dot_product = zeros_like(func_output, requires_grad=True)
    for grad_out, vec in zip(grads, v):
        if grad_out is not None:
            dot_product = dot_product + (grad_out * vec).sum()

    # 第二次反向：对内积求梯度，得到Hessian-vector product
    hessian_vec_product = grad(dot_product, inputs, create_graph=create_graph, allow_unused=True)

    if strict:
        for i, out in enumerate(hessian_vec_product):
            if out is None:
                raise RuntimeError(f"The hessian of the user-provided function is independent of input {i}. This is not allowed in strict mode.")

    # 对齐 PyTorch 的 _fill_in_zeros 语义：strict=False 时未使用输入的 HVP 返回零张量而非 None
    hessian_vec_product = [zeros_like(inp) if out is None else out for inp, out in zip(inputs, hessian_vec_product)]

    for inp, req_grad in zip(inputs, bak_requires_grad):
        inp.requires_grad_(req_grad)

    if is_single_input:
        hessian_vec_product = hessian_vec_product[0]  # type: ignore
    else:
        hessian_vec_product = tuple(hessian_vec_product)

    return func_output, hessian_vec_product

def hvp(func: Callable[..., TN], inputs: TN | list[TN] | tuple[TN, ...], v: TN | list[TN] | tuple[TN, ...], create_graph: bool = False, strict: bool = False) -> tuple[TN, TN | list[TN] | tuple[TN, ...]]:
    """计算Hessian-vector product (HVP): H @ v

    Hessian矩阵H是函数f对输入的二阶导数，此函数计算Hessian矩阵与向量v的乘积，
    无需显式构建完整Hessian矩阵。

    参数:
        func: 输入为张量或张量列表，输出为标量张量的函数
        inputs: 函数输入，张量或张量列表
        v: 要与Hessian相乘的向量，形状与inputs匹配
        create_graph (bool, 可选): 如果为True，计算图将被保留以支持高阶导数
        strict (bool, 可选): 如果为True，当检测到输出与某个输入无关时将引发错误；
            如果为False，未使用输入的HVP结果为零张量（与PyTorch一致）

    返回:
        tuple[TN, TN | list[TN] | tuple[TN, ...]]:
            - 第一个元素是函数的输出值
            - 第二个元素是HVP结果，形状与inputs匹配
    """
    return _compute_hessian_vector_product(func, inputs, v, create_graph, strict)

def vhp(func: Callable[..., TN], inputs: TN | list[TN] | tuple[TN, ...], v: TN | list[TN] | tuple[TN, ...], create_graph: bool = False, strict: bool = False) -> tuple[TN, TN | list[TN] | tuple[TN, ...]]:
    """计算vector-Hessian product (VHP): v @ H

    Hessian矩阵H是函数f对输入的二阶导数，此函数计算向量v与Hessian矩阵的乘积，
    无需显式构建完整Hessian矩阵。

    由于Hessian矩阵的对称性，vhp和hvp的结果在数值上是相同的。

    参数:
        func: 输入为张量或张量列表，输出为标量张量的函数
        inputs: 函数输入，张量或张量列表
        v: 要与Hessian相乘的向量，形状与inputs匹配
        create_graph (bool, 可选): 如果为True，计算图将被保留以支持高阶导数
        strict (bool, 可选): 如果为True，当检测到输出与某个输入无关时将引发错误；
            如果为False，未使用输入的VHP结果为零张量（与PyTorch一致）

    返回:
        tuple[TN, TN | list[TN] | tuple[TN, ...]]:
            - 第一个元素是函数的输出值
            - 第二个元素是VHP结果，形状与inputs匹配
    """
    return hvp(func, inputs, v, create_graph, strict)
