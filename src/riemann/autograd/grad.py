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

from __future__ import annotations

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
            - 与 torch.autograd.grad 一致，不对 grad_outputs 做 detach：
              create_graph=True 时返回的梯度对 grad_outputs 本身可微（VJP 关于
              余切线性，前向模式转置高阶导数依赖这条连接）；create_graph=False
              时反传在无梯度模式下进行，返回的梯度 requires_grad=False
            - 与 torch 一致，grad_outputs 必须与输出同设备，跨设备直接抛
              RuntimeError（不做隐式搬运）
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
            # 与 torch.autograd.grad 一致：grad_outputs 必须与输出同设备，
            # 不做隐式搬运（torch 报 invalid gradient at index 0 - expected device ...）
            if grad_outputs.device != outputs.device:
                raise RuntimeError(
                    f"invalid gradient at index 0 - expected device {outputs.device} "
                    f"but got {grad_outputs.device}")
            outputs._init_calc_graph()
            # 与 torch.autograd.grad 一致，不 detach grad_outputs：
            # create_graph=True 时梯度结果对余切本身可微（func.jvp 对局部 VJP
            # 做转置时需要这条依赖）；create_graph=False 时下面的反向遍历在
            # set_grad_enabled(False) 下进行，保存的梯度同样不带计算图。
            outputs.grad_value = grad_outputs
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


def _as_output_list(result):
    """辅助函数：把 func 的返回值统一成输出张量列表（单个TN / tuple / list）"""
    if isinstance(result, TN):
        return [result]
    if not isinstance(result, (tuple, list)):
        raise ValueError("func must return a TN tensor or a tuple/list of TN tensors")
    if not result:
        raise ValueError("Empty tensor list/tuple")
    for i, t in enumerate(result):
        if not isinstance(t, TN):
            raise ValueError(f"Output {i} must be a TN tensor")
    return list(result)


def _flatten_outputs(outputs, arrlib):
    """辅助函数：把一组输出张量展平并拼接成一维数组，用于按输出元素比对 Jacobian"""
    flats = [t._array.ravel() for t in outputs]
    if len(flats) == 1:
        return flats[0]
    return arrlib.concatenate(flats)


def _to_host(arr):
    """把设备数组转成 numpy（CUDA 下 cupy 数组需要先取回主机端）"""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def gradcheck(func: Callable[..., Any],
              inputs: TN | tuple[TN, ...] | list[TN],
              *,
              eps: float = 1e-6,
              atol: float = 1e-5,
              rtol: float = 1e-3,
              raise_exception: bool = True,
              nondet_tol: float = 0.0,
              check_undefined_grad: bool = True,
              check_grad_dtypes: bool = False,
              check_batched_grad: bool = False,
              check_batched_forward_grad: bool = False,
              check_forward_ad: bool = False,
              check_backward_ad: bool = True,
              fast_mode: bool = False,
              masked: bool = None) -> bool:
    """
    验证给定函数的梯度计算是否正确，通过逐元素比较数值 Jacobian 和解析 Jacobian。

    与 PyTorch 行为一致：不再把多个输出求和成一个标量后比较，而是对每个输出元素分别
    计算解析梯度并与中心差分得到的数值梯度逐元素比对。因此“不同输出元素的梯度互相
    抵消”这类错误也能被检出。

    参数：
        func (Callable[..., Any]): 需要验证梯度的函数，返回单个 TN 张量或多输出的元组/列表。
        inputs (TN | tuple[TN, ...] | list[TN]): 输入张量或其序列，必须 requires_grad=True
                                  且为双精度（float64）。单个 TN 会自动包装为长度为1的元组。
        eps (float, 可选): 中心差分扰动步长，默认为1e-6。
        atol (float, 可选): 绝对误差容限，默认为1e-5。
        rtol (float, 可选): 相对误差容限，默认为1e-3。
        raise_exception (bool, 可选): 检查失败时是否抛出 RuntimeError，默认 True。
        nondet_tol (float, 可选): 非确定性容限。绝对误差小于该值时视为通过（用于容忍
                                 非确定性算子带来的抖动），默认为0.0。
        check_undefined_grad (bool, 可选): 解析梯度为 None（输出与输入无关）时按零梯度
                                 参与比较，默认 True。
        check_grad_dtypes (bool, 可选): 是否额外校验解析梯度的 dtype 与输入一致，默认 False。
        check_batched_grad (bool, 可选): 未实现，置 True 抛 NotImplementedError。
        check_batched_forward_grad (bool, 可选): 未实现，置 True 抛 NotImplementedError。
        check_forward_ad (bool, 可选): 未实现（仅支持反向模式），置 True 抛 NotImplementedError。
        check_backward_ad (bool, 可选): 是否执行反向模式检查，仅支持 True。
        fast_mode (bool, 可选): 快速模式。用随机方向投影比较 v^T(Ju) 与 (v^T J)u，
                               数值侧每个输入只需 2 次前向，但依旧覆盖完整 Jacobian，
                               默认为 False（逐元素全量检查）。
        masked (bool, 可选): 稀疏张量掩码检查，未实现，非 None 抛 NotImplementedError。

    返回：
        bool: 检查通过返回 True；失败且 raise_exception=False 时返回 False。

    异常：
        RuntimeError: 梯度检查失败且 raise_exception=True 时抛出。
        ValueError: 输入参数不符合要求时抛出。
        NotImplementedError: 使用了当前尚未实现的检查选项时抛出。

    示例：
        >>> def f(x, y):
        >>>     return x * y
        >>> x = tensor(2.0, requires_grad=True, dtype=np.float64)
        >>> y = tensor(3.0, requires_grad=True, dtype=np.float64)
        >>> gradcheck(f, (x, y))
        True

    注意：
        1. 输入张量必须是双精度（float64）实值类型，以保证数值梯度精度。
        2. 多输出函数会逐个输出元素检查，不再退化为“输出求和”的弱校验。
        3. fast_mode=True 会显著加速，但只覆盖每个输入的首个元素。
    """
    # ---------------- 参数校验（与 torch 签名保持一致的选项行为） ----------------
    if not callable(func):
        raise ValueError("func must be a callable")

    if masked is not None:
        raise NotImplementedError("masked is not supported in current version of Riemann")
    if check_forward_ad:
        raise NotImplementedError("check_forward_ad requires forward-mode AD which is not supported")
    if check_batched_grad:
        raise NotImplementedError("check_batched_grad is not supported in current version of Riemann")
    if check_batched_forward_grad:
        raise NotImplementedError("check_batched_forward_grad is not supported in current version of Riemann")
    if not check_backward_ad:
        raise NotImplementedError("only backward-mode AD is supported, check_backward_ad must be True")

    # 与 torch 一致：inputs 允许单个 Tensor，也允许序列
    if isinstance(inputs, TN):
        inputs = (inputs,)
    elif isinstance(inputs, (tuple, list)):
        inputs = tuple(inputs)
    else:
        raise ValueError("inputs must be a TN tensor or a tuple/list of TN tensors")

    if not inputs:
        raise ValueError("inputs must contain at least one tensor")

    for i, inp in enumerate(inputs):
        if not isinstance(inp, TN):
            raise ValueError(f"Input {i} must be a TN tensor")
        if not inp.requires_grad:
            raise ValueError(f"Input {i} must have requires_grad=True")
        if inp.dtype != np.float64:
            raise ValueError(f"Input {i} must be double precision (float64), got {inp.dtype}")
        if np.issubdtype(inp.dtype, np.complexfloating):
            raise ValueError(f"Input {i} is complex, but gradcheck only supports real inputs")

    arrlib = inputs[0]._get_array_lib()

    # ---------------- 前向：保留计算图用于求解析梯度 ----------------
    outputs = _as_output_list(func(*inputs))

    out_sizes = [int(t._array.size) for t in outputs]
    # 注意：模块级 sum 被 tensordef 的归约函数覆盖，这里用 np.sum
    nout = int(np.sum(out_sizes))
    if nout == 0:
        # 输出为空张量：没有任何元素可以校验
        return True

    if fast_mode:
        return _fast_gradcheck(func, inputs, outputs, eps, atol, rtol, nondet_tol,
                               check_grad_dtypes, check_undefined_grad, raise_exception)

    return _slow_gradcheck(func, inputs, outputs, out_sizes, eps, atol, rtol, nondet_tol,
                           check_grad_dtypes, check_undefined_grad, raise_exception)
# end of gradcheck


def _slow_gradcheck(func, inputs, outputs, out_sizes, eps, atol, rtol, nondet_tol,
                    check_grad_dtypes, check_undefined_grad, raise_exception):
    """全量模式：逐输出元素比较完整 Jacobian（与 torch 的 slow mode 语义一致）"""
    arrlib = inputs[0]._get_array_lib()
    nout = int(np.sum(out_sizes))

    # 每个输出张量在展平后的一维数组中的起始偏移
    offsets = np.cumsum([0] + out_sizes[:-1]).tolist()

    # ---------------- 数值 Jacobian（中心差分） ----------------
    # 数值梯度只需要函数输出的数值，不参与求导，因此在禁用梯度模式下执行，
    # 避免每个扰动点都构建并丢弃一整张 autograd 图
    numerical_jacs = []
    for i, input_tensor in enumerate(inputs):
        nin = int(input_tensor._array.size)
        jac = arrlib.zeros((nin, nout), dtype=input_tensor.dtype)

        # 复用同一份扰动缓冲区：逐元素只改一个位置，避免整块拷贝数组并新建张量对象
        original_data = input_tensor._array
        perturbed_data = original_data.copy()
        x_perturbed = zeros_like(input_tensor, dtype=input_tensor.dtype, device=input_tensor.device)
        x_perturbed._array = perturbed_data
        x_perturbed.requires_grad = input_tensor.requires_grad
        perturbed_inputs = list(inputs)
        perturbed_inputs[i] = x_perturbed

        with set_grad_enabled(False):
            for r in range(nin):
                idx = np.unravel_index(r, original_data.shape) if original_data.ndim else ()
                # 正向扰动与负向扰动都从原始元素取值，避免浮点累加漂移
                perturbed_data[idx] = original_data[idx] + eps
                outputs_pos = _flatten_outputs(_as_output_list(func(*perturbed_inputs)), arrlib)

                perturbed_data[idx] = original_data[idx] - eps
                outputs_neg = _flatten_outputs(_as_output_list(func(*perturbed_inputs)), arrlib)

                # 该行即输入元素 r 对所有输出元素的一阶偏导
                jac[r] = (outputs_pos - outputs_neg) / (2 * eps)

        numerical_jacs.append(jac)

    # ---------------- 解析 Jacobian 与逐元素比对 ----------------
    # 复用掩码张量：每个输出张量一个，逐元素置1求导后再置0
    masks = [zeros_like(t, dtype=t.dtype, device=t.device) for t in outputs]

    failures = []
    for j in range(nout):
        # 定位输出元素 j 属于哪个输出张量及其局部下标
        k = 0
        while j >= offsets[k] + out_sizes[k]:
            k += 1
        local_idx = j - offsets[k]

        mask = masks[k]
        mask._array[np.unravel_index(local_idx, outputs[k]._array.shape)] = 1.0
        try:
            # 取第 j 个输出元素构成的标量，求其对各输入的梯度
            scalar = (outputs[k] * mask).sum()
            analytical_grads = grad(scalar, list(inputs), retain_graph=True, allow_unused=True)
        finally:
            mask._array[np.unravel_index(local_idx, outputs[k]._array.shape)] = 0.0

        for i, (num_jac, an_grad) in enumerate(zip(numerical_jacs, analytical_grads)):
            num_col = num_jac[:, j]

            if an_grad is None:
                if not check_undefined_grad:
                    invalids = abs(num_col) > (atol + nondet_tol)
                    if np.any(invalids):
                        failures.append(
                            f"input {i}, output element {j}: analytical grad is undefined "
                            f"but numerical grad is non-zero")
                    continue
                an_col = arrlib.zeros((num_jac.shape[0],), dtype=inputs[i].dtype)
            else:
                an_col = an_grad._array.ravel()

            if check_grad_dtypes and an_grad is not None and an_grad.dtype != inputs[i].dtype:
                failures.append(
                    f"input {i}, output element {j}: grad dtype {an_grad.dtype} "
                    f"does not match input dtype {inputs[i].dtype}")

            abs_error = np.abs(num_col - an_col)
            rel_error = abs_error / (np.abs(num_col) + np.abs(an_col) + 1e-10)
            # 逐元素判定：绝对误差与相对误差同时超限、且超出非确定性容限才算失败
            exceeded = (abs_error > atol) & (rel_error > rtol) & (abs_error > nondet_tol)

            if np.any(exceeded):
                pos = int(np.argmax(np.where(exceeded, abs_error, -1)))
                idx = np.unravel_index(pos, inputs[i]._array.shape) if inputs[i]._array.ndim else ()
                failures.append(
                    f"input {i}, output element {j} at {idx}: "
                    f"absolute error {float(abs_error[pos]):.6e}, "
                    f"relative error {float(rel_error[pos]):.6e}")
                if len(failures) >= 8:
                    break

        if len(failures) >= 8:
            break

    if failures:
        if raise_exception:
            raise RuntimeError("Gradient check failed:\n" + "\n".join(failures))
        return False

    return True


def _random_direction(shape, arrlib, dtype, rng, normalize=False):
    """生成指定形状的随机方向：numpy 生成后转到目标数组库（CPU/CUDA 通用）"""
    arr = rng.standard_normal(shape) if shape else rng.standard_normal(())
    if normalize:
        norm = np.linalg.norm(arr)
        arr = arr / (norm if norm else 1.0)
    return arrlib.asarray(arr.astype(dtype))


def _fast_gradcheck(func, inputs, outputs, eps, atol, rtol, nondet_tol,
                    check_grad_dtypes, check_undefined_grad, raise_exception):
    """快速模式：随机方向投影校验（与 torch 的 fast_mode 语义一致）。

    对每个输入生成随机单位方向 u、每个输出生成随机方向 v，比较
        v^T (J u)   —— 由中心差分沿方向 u 求得（每个输入只需 2 次前向）
        (v^T J) u   —— 由一次反向传播求得（每个输出只需 1 次反向）
    二者对正确梯度恒等。相比逐元素的慢速模式，前向次数与输入元素数无关，
    同时仍覆盖整个 Jacobian（随机投影，漏检概率为零测度）。
    """
    arrlib = inputs[0]._get_array_lib()
    # 固定种子，保证反复运行结果一致（不影响投影的有效性）
    rng = np.random.default_rng(0)
    n_in = len(inputs)
    n_out = len(outputs)

    us = [_random_direction(inp._array.shape, arrlib, inp.dtype, rng, normalize=True)
          for inp in inputs]
    vs = [_random_direction(out._array.shape, arrlib, out.dtype, rng) for out in outputs]

    # ---------------- 数值侧：v^T (J u) ----------------
    numerical = [[0.0] * n_out for _ in range(n_in)]
    for i, input_tensor in enumerate(inputs):
        perturbed_inputs = list(inputs)
        x_perturbed = zeros_like(input_tensor, dtype=input_tensor.dtype, device=input_tensor.device)
        x_perturbed.requires_grad = input_tensor.requires_grad
        perturbed_inputs[i] = x_perturbed

        with set_grad_enabled(False):
            x_perturbed._array = input_tensor._array + eps * us[i]
            pos = [t._array for t in _as_output_list(func(*perturbed_inputs))]

            x_perturbed._array = input_tensor._array - eps * us[i]
            neg = [t._array for t in _as_output_list(func(*perturbed_inputs))]

        for j in range(n_out):
            if outputs[j]._array.size == 0:
                continue
            ju = (pos[j] - neg[j]) / (2 * eps)  # J_ij u_i
            numerical[i][j] = float(arrlib.sum(vs[j] * ju))

    # ---------------- 解析侧：(v^T J) u ----------------
    analytical = [[0.0] * n_out for _ in range(n_in)]
    failures = []
    for j, out in enumerate(outputs):
        v_tensor = zeros_like(out, dtype=out.dtype, device=out.device)
        v_tensor._array = vs[j]
        analytical_grads = grad(out, list(inputs), retain_graph=True,
                                grad_outputs=v_tensor, allow_unused=True)

        for i, an_grad in enumerate(analytical_grads):
            if an_grad is None:
                if check_undefined_grad:
                    analytical[i][j] = 0.0
                continue
            if check_grad_dtypes and an_grad.dtype != inputs[i].dtype:
                failures.append(
                    f"input {i}, output {j}: grad dtype {an_grad.dtype} "
                    f"does not match input dtype {inputs[i].dtype}")
            if inputs[i]._array.size == 0:
                continue
            analytical[i][j] = float(arrlib.sum(an_grad._array * us[i]))

    # ---------------- 比对投影后的标量 ----------------
    for i in range(n_in):
        # 与 torch 一致：atol 需要按 |Σu| * |Σv| 缩放，才能和逐元素容限可比
        sum_u = float(np.abs(np.sum(_to_host(us[i]))))
        for j in range(n_out):
            if outputs[j]._array.size == 0:
                continue
            sum_v = float(np.abs(np.sum(_to_host(vs[j]))))
            effective_atol = atol * sum_u * sum_v

            num_val, an_val = numerical[i][j], analytical[i][j]
            abs_error = abs(num_val - an_val)
            rel_error = abs_error / (abs(num_val) + abs(an_val) + 1e-10)

            if abs_error > effective_atol + rtol * abs(an_val) and abs_error > nondet_tol:
                failures.append(
                    f"input {i}, output {j}: projected absolute error {abs_error:.6e}, "
                    f"relative error {rel_error:.6e}")

    if not failures:
        return True

    if not raise_exception:
        return False

    # 快速模式失败时用全量模式定位具体元素，给出可操作的报错信息
    try:
        _slow_gradcheck(func, inputs, outputs, [int(t._array.size) for t in outputs],
                        eps, atol, rtol, nondet_tol, check_grad_dtypes,
                        check_undefined_grad, raise_exception=True)
    except RuntimeError as exc:
        raise RuntimeError("Fast gradient check failed:\n" + str(exc)) from None

    raise RuntimeError(
        "Fast gradient check failed but element-wise differences are small; "
        "the same check passes in slow mode. The function's gradient may be "
        "non-deterministic or numerically unstable. Details:\n" + "\n".join(failures))
# end of _fast_gradcheck

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