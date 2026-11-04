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
func 模块 —— 函数式自动微分接口，参考 torch.func 设计。

提供纯函数式的微分原语，不修改函数输入的 requires_grad 状态，
支持高阶导数、向量积、Jacobian/Hessian 等常见梯度操作。

核心函数：
    - grad / grad_and_value: 标量值函数的梯度
    - jvp / vjp: Jacobian-vector / vector-Jacobian 积
    - jacrev / jacfwd / jacobian: Jacobian 矩阵的反向/正向/默认计算方式
    - hessian / hvp / vhp: Hessian 矩阵及其向量积
    - vmap: 向量化映射（沿指定批次维度循环调用并堆叠）
    - functional_call: 以自定义参数调用 Module 实例

所有函数间相互复用，底层统一调用 autograd.grad。
"""

import numpy as np
import builtins
from typing import Callable, Any, Optional, Union
from .tensordef import *
from .autograd.grad import grad as _grad
from .autograd.functional import (
    jacobian as _jacobian,
    hessian as _hessian,
    jvp as _jvp,
    vjp as _vjp,
    hvp as _hvp,
    vhp as _vhp,
)

# ---------------------------------------------------------------------------
# grad / grad_and_value
# ---------------------------------------------------------------------------

def grad(func: Callable[..., TN],
         argnums: Union[int, tuple[int, ...]] = 0,
         has_aux: bool = False,
         create_graph: bool = False) -> Callable[..., Union[TN, tuple[TN, Any]]]:
    """返回一个新函数，该新函数计算 func 在参数 argnums 处的梯度。

    与 autograd.grad 的区别：
        autograd.grad 接受输出和输入；func.grad 接受函数和 argnums，
        返回一个包裹函数，该包裹函数接受 inputs 并自动前向+反向。

    参数:
        func: 可微函数，输出必须是标量 TN
        argnums: 需要求导的参数位置，单个 int 或 tuple[int, ...]；默认 0
        has_aux: 如果 True，func 可以返回 (value, aux) 二元组；
                  包裹函数也会额外返回 aux 而不对其求导。
        create_graph: 如果 True，梯度张量会被追踪，可用于高阶微分。默认 False。

    返回:
        Callable: 包裹函数，输入为 func 的参数，返回梯度（若 has_aux=True 则返回 (grad, aux)）
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def _wrapper(*args, **kwargs):
        outputs = func(*args, **kwargs)
        aux = None
        if has_aux:
            outputs, aux = outputs

        if outputs.ndim != 0:
            raise RuntimeError(
                f"grad() requires func to return a scalar tensor, got shape {outputs.shape}"
            )

        grad_args = [args[i] for i in argnums]
        grad_result = _grad(outputs, grad_args, create_graph=create_graph)

        if len(argnums) == 1:
            grad_result = grad_result[0]

        if has_aux:
            return grad_result, aux
        return grad_result

    return _wrapper


def grad_and_value(func: Callable[..., TN],
                   argnums: Union[int, tuple[int, ...]] = 0,
                   has_aux: bool = False,
                   create_graph: bool = False) -> Callable[..., Union[tuple[TN, TN], tuple[TN, TN, Any]]]:
    """同时返回梯度和函数值的包裹函数。

    内部只做一次前向+一次反向（与分开调用 func 和 grad(func) 等价）。

    返回:
        Callable: 输入为 func 的参数，返回 (grad, value) 或 (grad, value, aux)
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def _wrapper(*args, **kwargs):
        outputs = func(*args, **kwargs)
        aux = None
        if has_aux:
            outputs, aux = outputs

        if outputs.ndim != 0:
            raise RuntimeError(
                f"grad_and_value() requires func to return a scalar tensor, got shape {outputs.shape}"
            )

        grad_args = [args[i] for i in argnums]
        grad_result = _grad(outputs, grad_args, create_graph=create_graph)

        if len(argnums) == 1:
            grad_result = grad_result[0]

        if has_aux:
            return grad_result, outputs, aux
        return grad_result, outputs

    return _wrapper


# ---------------------------------------------------------------------------
# jvp / vjp (薄封装，直接复用 autograd.functional)
# ---------------------------------------------------------------------------

def _wrap_single_or_tuple(result):
    """将 (可能是 list 的) 结果包装为 tuple 或单 TN（若仅一个元素）"""
    if isinstance(result, list):
        result = tuple(result)
    if isinstance(result, tuple) and len(result) == 1:
        result = result[0]
    return result


def jvp(func: Callable[..., Union[TN, tuple[TN, ...]]],
        primals: Union[TN, list[TN], tuple[TN, ...]],
        tangents: Union[TN, list[TN], tuple[TN, ...]],
        create_graph: bool = False,
        strict: bool = False) -> tuple[Union[TN, tuple[TN, ...]], Union[TN, tuple[TN, ...]]]:
    """Jacobian-vector 积：func 与 tangents 的 JVP，返回 (输出值, JVP 结果)"""
    out, jvp_res = _jvp(func, primals, tangents, create_graph=create_graph, strict=strict)
    return out, _wrap_single_or_tuple(jvp_res)


def vjp(func: Callable[..., Union[TN, tuple[TN, ...]]],
        primals: Union[TN, list[TN], tuple[TN, ...]],
        cotangents: Optional[Union[TN, list[TN], tuple[TN, ...]]] = None,
        create_graph: bool = False,
        strict: bool = False) -> tuple[Union[TN, tuple[TN, ...]], Union[TN, tuple[TN, ...]]]:
    """vector-Jacobian 积：func 与 cotangents 的 VJP，返回 (输出值, VJP 结果)"""
    out, vjp_res = _vjp(func, primals, cotangents, create_graph=create_graph, strict=strict)
    return out, _wrap_single_or_tuple(vjp_res)


# ---------------------------------------------------------------------------
# jacrev / jacfwd / jacobian
# ---------------------------------------------------------------------------

def jacrev(func: Callable[..., Union[TN, tuple[TN, ...]]],
           argnums: Union[int, tuple[int, ...]] = 0,
           create_graph: bool = False,
           strict: bool = True) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """通过反向模式 (reverse-mode autodiff) 计算 Jacobian 矩阵。

    对于输出形状 (y1, y2, ...) 和输入形状 (x1, x2, ...)，
    Jacobian 形状为 (y1, y2, ..., x1, x2, ...)。

    返回一个包裹函数，包裹函数接受 func 的参数并返回 Jacobian。
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def _wrapper(*args, **kwargs):
        is_single_argnum = (len(argnums) == 1 and argnums[0] == 0 and len(args) == 1)
        if is_single_argnum:
            return _jacobian(
                lambda *a: func(*a, **kwargs),
                args[0],
                create_graph=create_graph,
                strict=strict,
            )
        all_args = list(args)
        result = _jacobian(
            lambda *a: func(*a, **kwargs),
            all_args,
            create_graph=create_graph,
            strict=strict,
        )
        # 底层返回 tuple（多输入），按 argnums 过滤结果
        if isinstance(result, tuple):
            filtered = tuple(result[i] for i in argnums)
            if len(filtered) == 1:
                return filtered[0]
            return filtered
        return result

    return _wrapper


def jacfwd(func: Callable[..., Union[TN, tuple[TN, ...]]],
           argnums: Union[int, tuple[int, ...]] = 0,
           create_graph: bool = False) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """通过正向模式 (forward-mode) 计算 Jacobian 矩阵。

    Riemann 当前没有前向模式 autodiff，因此通过 vmap+jvp 实现：
    对每个输入元素构造 one-hot tangents，调用一次 jvp 得到 Jacobian 的一列，
    最后将所有列 stack 成完整 Jacobian。

    返回一个包裹函数，包裹函数接受 func 的参数并返回 Jacobian。
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def _wrapper(*args, **kwargs):
        all_args = list(args)
        # [(inp_tensor, inp_pos_in_all_args), ...]  —— 同时记录张量和它在 args 中的位置
        result_specs = [(all_args[i], i) for i in argnums]
        is_single_input = len(result_specs) == 1

        # 先跑一次正向拿到输出形状
        outputs = func(*all_args, **kwargs)
        is_single_output = isinstance(outputs, TN)
        output_tensor = outputs if is_single_output else outputs[0]

        jac_blocks = []
        for inp, inp_pos in result_specs:
            numel = int(np.prod(inp.shape))
            tangent_flat = zeros(numel, dtype=inp.dtype, device=inp.device)
            cols = []
            for idx in range(numel):
                tangent_flat._array[idx] = 1.0
                tangent = tangent_flat.reshape(inp.shape)
                # 构造完整 tangents：对当前 inp 位置放 one-hot tangent，其余放零
                full_tangents = []
                for ai, a in enumerate(all_args):
                    if ai == inp_pos:
                        full_tangents.append(tangent)
                    elif ai in argnums:
                        full_tangents.append(zeros_like(a))
                    else:
                        full_tangents.append(zeros_like(a))
                _, jvp_res = _jvp(
                    lambda *a: func(*a, **kwargs),
                    all_args, full_tangents,
                    create_graph=create_graph,
                    strict=False,
                )
                tangent_flat._array[idx] = 0.0
                # 正确取对应 inp 的 jvp 结果 —— 单输入时 jvp_res 可能直接是 TN
                if isinstance(jvp_res, TN):
                    cols.append(jvp_res)
                else:
                    # jvp_res 是 tuple/list，长度等于 all_args；取 inp_pos 对应位置
                    cols.append(jvp_res[inp_pos])
            # cols[i] 是形状 output_shape 的张量，stack 后再 reshape 到 output_shape + input_shape
            jac_block = stack(cols, dim=-1).reshape(output_tensor.shape + inp.shape)
            jac_blocks.append(jac_block)

        if is_single_input:
            return jac_blocks[0]
        return tuple(jac_blocks)

    return _wrapper


def jacobian(func: Callable[..., Union[TN, tuple[TN, ...]]],
             argnums: Union[int, tuple[int, ...]] = 0,
             create_graph: bool = False,
             strict: bool = True,
             mode: str = "reverse") -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """Jacobian 矩阵的通用接口，mode 选择算法。

    参数:
        mode: "reverse" (默认, 调用 jacrev) 或 "forward" (调用 jacfwd)
    """
    if mode == "reverse":
        return jacrev(func, argnums=argnums, create_graph=create_graph, strict=strict)
    elif mode == "forward":
        return jacfwd(func, argnums=argnums, create_graph=create_graph)
    else:
        raise ValueError(f"mode must be 'reverse' or 'forward', got {mode!r}")


# ---------------------------------------------------------------------------
# hessian (薄封装)
# ---------------------------------------------------------------------------

def hessian(func: Callable[..., TN],
            argnums: Union[int, tuple[int, ...]] = 0,
            create_graph: bool = False,
            strict: bool = True) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """计算标量值函数的 Hessian 矩阵，返回包裹函数。"""
    if isinstance(argnums, int):
        argnums = (argnums,)

    def _wrapper(*args, **kwargs):
        is_single_argnum = (len(argnums) == 1 and argnums[0] == 0 and len(args) == 1)
        if is_single_argnum:
            return _hessian(
                lambda *a: func(*a, **kwargs),
                args[0],
                create_graph=create_graph,
                strict=strict,
            )
        all_args = list(args)
        # 当 argnums 不是全部参数位置时，内部自动放宽 strict=False，
        # 因为 cross-Hessian 对非 argnums 的依赖不是用户关心的范围。
        effective_strict = strict if len(argnums) == len(args) else False
        result = _hessian(
            lambda *a: func(*a, **kwargs),
            all_args,
            create_graph=create_graph,
            strict=effective_strict,
        )
        # 底层 _hessian 多输入返回 tuple-of-tuples:
        #   ((∂grad_0/∂x_0, ∂grad_0/∂x_1, ...),
        #    (∂grad_1/∂x_0, ∂grad_1/∂x_1, ...), ...)
        # 对角线 result[k][k] 才是第 k 个 input 的 Hessian 矩阵
        if isinstance(result, tuple):
            filtered = tuple(result[i][i] for i in argnums)
            if len(filtered) == 1:
                return filtered[0]
            return filtered
        return result

    return _wrapper


def hvp(func: Callable[..., TN],
        argnums: Union[int, tuple[int, ...]] = 0,
        create_graph: bool = False,
        strict: bool = False) -> Callable[..., tuple[Union[TN, tuple[TN, ...]], Union[TN, tuple[TN, ...]]]]:
    """Hessian-vector product，返回包裹函数。

    包裹函数签名: (inputs, v) -> (output, hvp_result)
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def _wrapper(inputs, v, **kwargs):
        single_input = not isinstance(inputs, (list, tuple))
        if single_input:
            inputs = [inputs]
        else:
            inputs = list(inputs)
        single_v = not isinstance(v, (list, tuple))
        if single_v:
            v = [v]
        else:
            v = list(v)
        out, hvp_res = _hvp(
            lambda *a: func(*a, **kwargs),
            inputs, v,
            create_graph=create_graph,
            strict=strict,
        )
        # 底层 _hvp 返回 list，这里统一回包成 tuple 或单 TN
        if isinstance(hvp_res, list):
            hvp_res = tuple(hvp_res)
        if len(hvp_res) == 1:
            hvp_res = hvp_res[0]
        return out, hvp_res

    return _wrapper


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------

def vmap(func: Callable[..., Union[TN, tuple[TN, ...]]],
         in_dims: Union[int, tuple[int, ...], list[Optional[int]]] = 0,
         out_dims: Union[int, tuple[int, ...], list[Optional[int]]] = 0) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """向量化映射 (vectorized map)。

    将原本作用于单个样本的函数扩展到批次维度上并行（通过 loop+stack 实现，
    不是真正的向量化，但接口与 torch.func.vmap 对齐）。

    参数:
        in_dims: 每个输入的批次维度。int 表示所有输入共用该维度；
                 tuple[int,...] 表示每个输入各自的维度位置；None 表示该输入不批处理。
        out_dims: 每个输出的批次维度，语义同 in_dims。

    返回:
        包裹函数，接受批次输入，返回批次输出。

    注意:
        - Riemann 的 __getitem__ 支持整轴切片，可直接用 inputs[batch_idx] 切样本
        - 批次维通常是 dim=0；若指定 dim>0，需要先 moveaxis 到 0 再切
        - 非批处理输入 (in_dims=None) 直接透传给底层 func
    """
    def _normalize_dims(dims, n_expected):
        if isinstance(dims, int):
            return [dims] * n_expected
        if isinstance(dims, (tuple, list)):
            return list(dims) + [None] * builtins.max(0, n_expected - len(dims))
        return [None] * n_expected

    def _wrapper(*args, **kwargs):
        # 先跑一次正向拿到每个输入的批次大小
        n_inputs = len(args)
        in_dims_list = _normalize_dims(in_dims, n_inputs)

        batch_size = None
        for dim, a in zip(in_dims_list, args):
            if dim is not None and isinstance(a, TN):
                # dim 可能是负数，需要规范化
                ndim = a.ndim
                d = dim if dim >= 0 else ndim + dim
                batch_size = a.shape[d]
                break

        if batch_size is None:
            # 没有批处理维度 — 直接调用 func
            return func(*args, **kwargs)

        outputs_list = []
        for b in range(batch_size):
            sliced_args = []
            for dim, a in zip(in_dims_list, args):
                if dim is None:
                    sliced_args.append(a)
                    continue
                if not isinstance(a, TN):
                    sliced_args.append(a)
                    continue
                d = dim if dim >= 0 else a.ndim + dim
                # 构建切片：对 dim 位置取 b，其它位置取 :
                idx = tuple(slice(None) if i != d else b for i in range(a.ndim))
                sliced = a[idx]
                # 如果输入原本有多个批次维，sliced 会丢掉 dim，这里保持形状不变
                sliced_args.append(sliced)
            outputs_list.append(func(*sliced_args, **kwargs))

        # 堆叠结果
        first_out = outputs_list[0]
        is_single = isinstance(first_out, TN)
        out_dims_list = _normalize_dims(out_dims, 1 if is_single else len(first_out))

        def _stack_result(items, out_dim):
            # 沿 out_dim 堆叠；若 out_dim 是 None 直接 list（罕见）
            axis = out_dim if out_dim is not None else 0
            return stack(items, dim=axis)

        if is_single:
            return _stack_result(outputs_list, out_dims_list[0])
        else:
            n_out = len(first_out)
            result = []
            for oi in range(n_out):
                stacked = _stack_result([out[oi] for out in outputs_list],
                                         out_dims_list[oi] if oi < len(out_dims_list) else 0)
                result.append(stacked)
            return tuple(result)

    return _wrapper


# ---------------------------------------------------------------------------
# functional_call
# ---------------------------------------------------------------------------

def functional_call(module: Any,
                    parameter_values: Any,
                    args: Union[TN, tuple[TN, ...]],
                    kwargs: Optional[dict] = None) -> Any:
    """以自定义参数调用 module.forward，不修改 module 自身的参数。

    内部临时将 module 的具名参数属性替换为 parameter_values[i]，调用 forward 后恢复。
    用于无状态（stateless）地调用 Module，便于与 jacrev/vmap 等函数式原语组合。

    参数:
        module: 带有 forward 方法的对象（通常是 riemann.nn.Module 或子类）
        parameter_values: 与 module.named_parameters() 顺序一致的参数值。
                          可以是 list/tuple/TN 中的任一种。若 module 有参数而 parameter_values 为 None，
                          将使用 module 自身参数（退化为普通 forward 调用）。
        args: forward 的位置参数
        kwargs: forward 的关键字参数

    返回:
        module.forward(*args, **kwargs) 的返回值
    """
    if kwargs is None:
        kwargs = {}

    # 解析 module 的参数
    if hasattr(module, "named_parameters"):
        module_params = list(module.named_parameters())
    elif hasattr(module, "_named_parameters"):
        module_params = list(module._named_parameters())
    else:
        # 没有参数的 module —— 直接调用 forward（或 __call__）
        forward = getattr(module, "forward", None) or module
        return forward(*args, **kwargs)

    if parameter_values is None:
        forward = getattr(module, "forward", None) or module
        return forward(*args, **kwargs)

    # 归一化 parameter_values 为 list
    if isinstance(parameter_values, TN):
        parameter_values = [parameter_values]
    elif not isinstance(parameter_values, (list, tuple)):
        raise TypeError(
            f"parameter_values must be a TN or tuple/list of TN, got {type(parameter_values)}"
        )

    if len(parameter_values) != len(module_params):
        raise ValueError(
            f"module has {len(module_params)} parameters but got "
            f"{len(parameter_values)} values in parameter_values"
        )

    # 临时替换 module 的参数属性，调用 forward，然后恢复
    orig_values = []
    try:
        for (name, orig), new_val in zip(module_params, parameter_values):
            orig_values.append(orig)
            setattr(module, name, new_val)
        forward = getattr(module, "forward", None) or module
        return forward(*args, **kwargs)
    finally:
        for (name, _), orig in zip(module_params, orig_values):
            setattr(module, name, orig)
