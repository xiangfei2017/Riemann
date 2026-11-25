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
    - linearize: 一次前向记录 primal 图，返回可复用的 J(x)·v 线性函数
    - jacrev / jacfwd: Jacobian 矩阵的反向/正向计算方式
    - hessian: Hessian 矩阵
    - vmap: 向量化映射（沿指定批次维度循环调用并堆叠）
    - functional_call: 以自定义参数/缓冲区调用 Module 实例
    - stack_module_state: 堆叠多个 Module 的状态，供 functional_call + vmap 使用

所有函数间相互复用：jvp / vjp 是前向 / 反向模式原语；jvp 基于 _record_primal +
_jvp_with_primal_graph 复用前向图，vjp 基于 _call_and_split + _vjp_with_output 复用
反向图；jvp2 走 _traceable_args / _prepare_forward 的旧路径，jacrev / jacfwd 也复用
_traceable_args 做参数归一化。
jacrev / jacfwd 共用 _make_jacobian 做 argnums 校验与结果整形，内核分别是 vmap(vjp)
与逐列 jvp（func 只前向求值一次并记录图，各列复用该图，等价于 vmap(jvp)）。两者均
支持 has_aux；jacrev 另支持 chunk_size（与 torch.func 一致，jacfwd 无 chunk_size）。
linearize 复用 _record_primal + _jvp_with_primal_graph，与 jacfwd 共享同一条前向图。
hessian 复用 _jacobian_via_vjp 内核，对 grad 的结果再求一次 Jacobian（反向叠反向）。

补充说明：jvp 走直接规则路径——算子登记 `_jvp_rule`，前向执行时记录 primal 图，
再沿拓扑序直接传播 tangent；缺少规则的节点会直接报错，内置算子均已覆盖。
jvp2 走旧路径——算子只登记局部 VJP（反向模式），jvp2 通过"对局部 VJP 再做
一次转置"沿计算图正向传播 tangent（细节见 _local_jvp）。
"""

import builtins
from typing import Callable, Any, Optional, Union
from .core import *
from .core.ad_context import (
    TANGENT,
    JVPFrame,
    PRIMAL,
    pop_jvp_frame,
    push_jvp_frame,
    set_jvp_phase,
    forward_ad_trace_active,
)
from .core.tensordef import _getitem_grad, _getitem_jvp
from .core.tensorfuncs import (
    _apply_inplace_jvp_steps,
    _frame_inplace_steps,
)
from .autograd.grad import grad as _grad


# ---------------------------------------------------------------------------
# argnums 校验（与 torch.func 的错误信息保持一致）
# ---------------------------------------------------------------------------

_RANDOMNESS_CHOICES = ('error', 'different', 'same')


def _check_randomness(randomness: str):
    """校验 randomness 取值，错误信息与 torch.func 一致。

    参数:
        randomness: 随机性策略，合法值为 'error' / 'different' / 'same'。

    异常:
        RuntimeError: 取值不在合法集合中。
    """
    if randomness not in _RANDOMNESS_CHOICES:
        raise RuntimeError(
            "Only allowed values for randomness are 'error', 'different', or "
            f"'same'. Got {randomness}")


def _check_argnums(argnums, allow_none: bool = False):
    """校验并归一化 argnums，返回位置元组；allow_none=True 时 None 表示全部参数。

    合法取值：int 或非空 tuple[int, ...]。校验在调用包裹函数时进行，错误信息
    与 torch.func 一致（() -> argnums must be non-empty；
    None / list 等 -> argnums must be int or Tuple[int, ...]）。

    参数:
        argnums: 待校验的输入位置，int 或 tuple[int, ...]；allow_none 时可为 None。
        allow_none: 是否允许 None 表示"全部参数"。

    返回:
        归一化后的 tuple[int, ...]；argnums 为 None 且 allow_none=True 时返回 None。
    """
    if allow_none and argnums is None:
        return None
    if isinstance(argnums, int):
        return (argnums,)
    if not isinstance(argnums, tuple):
        raise RuntimeError(
            f"argnums must be int or Tuple[int, ...], got: {type(argnums)}"
        )
    for a in argnums:
        if not isinstance(a, int):
            raise RuntimeError(f"argnum must be int, got: {type(a)}")
    if len(argnums) == 0:
        raise RuntimeError("argnums must be non-empty")
    return argnums


# ---------------------------------------------------------------------------
# grad / grad_and_value
# ---------------------------------------------------------------------------

def _traceable_call_args(args, positions):
    """返回把 positions 处不可微张量替换为可微副本后的实参列表。

    与 torch.func 一致：grad 系列不要求输入自带 requires_grad；未入选位置
    原样透传，用户张量本身不被修改。

    参数:
        args: 完整实参序列。
        positions: 选中求导的位置序列。

    返回:
        新的实参列表；入选位置若非张量则抛 RuntimeError。
    """
    call_args = list(args)
    for i in positions:
        a = call_args[i]
        if not isinstance(a, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
        if not a.requires_grad:
            call_args[i] = a.detach().requires_grad_(True)
    return call_args


def _materialize_grads(outputs, grad_args):
    """计算 outputs 对 grad_args 的梯度，未连通/不可微位置以零张量补齐。

    与 torch.func 一致：输出不依赖任何输入（常量、已 detach）或某输入与输出
    不连通时，该输入位置的梯度为零张量，而不是报错。

    参数:
        outputs: 标量输出张量。
        grad_args: 需要求导的参数序列（已按 argnums 取出）。

    返回:
        与 grad_args 等长的梯度列表；不可微位置为 zeros_like(输入)。
    """
    for i, a in enumerate(grad_args):
        if not isinstance(a, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
    active = [a for a in grad_args if a.requires_grad]
    if outputs.requires_grad and active:
        raw = iter(_grad(outputs, active, create_graph=True, allow_unused=True))
        grads = [next(raw) if a.requires_grad else None for a in grad_args]
        return [zeros_like(a) if g is None else g for a, g in zip(grad_args, grads)]
    return [zeros_like(a) for a in grad_args]


def grad(func: Callable[..., TN],
         argnums: Union[int, tuple[int, ...]] = 0,
         has_aux: bool = False) -> Callable[..., Union[TN, tuple[TN, Any]]]:
    """返回 func 的导函数（一个包裹函数）。

    接口严格对齐 torch.func.grad：无 create_graph 参数，高阶导数自然可用。

    注意区分三层输出：
        1. func 的输出：默认必须是标量 TN（ndim == 0）；has_aux=True 时必须是
           (value, aux) 二元组，value 是标量 TN，aux 是任意辅助结果。
        2. func.grad(...) 的输出：一个函数对象，即 func 的导函数（包裹函数），
           它接受与 func 相同的参数。
        3. 导函数的输出：has_aux=False 时返回 grad_result；has_aux=True 时返回
           (grad_result, aux)，aux 原样透传。grad_result 是 value 关于 argnums
           所选参数的梯度：只含 1 个参数位置时为单个梯度张量（形状与对应输入
           相同），含多个位置时为由多个梯度张量构成的元组（顺序与 argnums 一致）。

    与 torch.func.grad 一致：func 的输出不依赖某个入选参数（或输出本身不可微）
    时，该参数位置的梯度为零张量，而不是报错。

    参数:
        func: 可微函数，输出必须是标量 TN（ndim == 0），否则抛出 RuntimeError
        argnums: 需要求导的参数位置，int 或 tuple[int, ...]，默认 0。非法取值
                 （()、None、含非 int 元素的元组、list 等）在调用导函数时抛
                 RuntimeError，错误信息与 torch.func 一致
        has_aux: 如果 True，func 必须返回 (value, aux) 二元组：value 是 func 的
                 前向计算输出，grad 要求它必须是标量张量；aux 是 func 顺带输出
                 的辅助结果（任意对象，不参与求导）。导函数只从 value 计算梯度，
                 并把 aux 原样透传在返回值中

    返回:
        Callable: func 的导函数，其输出见上文第 3 点
    """
    def _wrapper(*args, **kwargs):
        an = _check_argnums(argnums)
        call_args = _traceable_call_args(args, an)
        outputs = func(*call_args, **kwargs)
        aux = None
        if has_aux:
            outputs, aux = outputs

        if outputs.ndim != 0:
            raise RuntimeError(
                f"grad() requires func to return a scalar tensor, got shape {outputs.shape}"
            )

        grad_args = [call_args[i] for i in an]
        grad_result = _materialize_grads(outputs, grad_args)

        if len(an) == 1:
            grad_result = grad_result[0]
        else:
            grad_result = tuple(grad_result)

        if has_aux:
            return grad_result, aux
        return grad_result

    return _wrapper


def grad_and_value(func: Callable[..., TN],
                   argnums: Union[int, tuple[int, ...]] = 0,
                   has_aux: bool = False) -> Callable[..., Union[tuple[TN, TN], tuple[TN, TN, Any]]]:
    """返回 func 的导函数，同时返回 func 的前向输出（value）。

    接口严格对齐 torch.func.grad_and_value：无 create_graph 参数，高阶导数自然可用。

    注意区分三层输出：
        1. func 的输出：同 func.grad（默认标量 TN；has_aux=True 时为 (value, aux)）
        2. func.grad_and_value(...) 的输出：一个函数对象，接受与 func 相同的参数
        3. 该函数的输出：has_aux=False 时返回 (grad_result, value)；has_aux=True
           时返回 (grad_result, value, aux)。grad_result 是 value 关于 argnums 所选
           参数的梯度：只含 1 个位置时为单个梯度张量，含多个位置时为梯度元组

    与 torch.func.grad_and_value 一致：func 的输出不依赖某个入选参数（或输出本身
    不可微）时，该参数位置的梯度为零张量，而不是报错。

    内部只做一次前向+一次反向（与分开调用 func 和 grad(func) 等价）。

    参数:
        argnums: 同 func.grad（非法取值在调用导函数时抛 RuntimeError，
                 错误信息与 torch.func 一致）

    返回:
        Callable: 导函数，其输出见上文第 3 点
    """
    def _wrapper(*args, **kwargs):
        an = _check_argnums(argnums)
        call_args = _traceable_call_args(args, an)
        outputs = func(*call_args, **kwargs)
        aux = None
        if has_aux:
            outputs, aux = outputs

        if outputs.ndim != 0:
            raise RuntimeError(
                f"grad_and_value() requires func to return a scalar tensor, got shape {outputs.shape}"
            )

        grad_args = [call_args[i] for i in an]
        grad_result = _materialize_grads(outputs, grad_args)

        if len(an) == 1:
            grad_result = grad_result[0]
        else:
            grad_result = tuple(grad_result)

        if has_aux:
            return grad_result, outputs, aux
        return grad_result, outputs

    return _wrapper


# ---------------------------------------------------------------------------
# jvp / vjp 原语
#
# 设计对齐 torch.func：
#   - jvp 是真正的前向模式原语，沿计算图正向传播 tangent；
#   - vjp 是反向模式原语，反向传播 cotangent；
#   - jacfwd / jacrev / hessian 均由这两个原语与 vmap 组合而成。
# ---------------------------------------------------------------------------

def _as_list(value):
    """把单个对象或 list/tuple 序列统一转为 list。

    参数:
        value: 单个对象，或对象序列。

    返回:
        对象列表；输入本身是 list/tuple 时返回其浅拷贝。
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _ensure_traceable(tensors):
    """确保每个张量 requires_grad=True；关闭梯度的输入会被 detach 成新的叶子。

    参数:
        tensors: 张量序列。

    返回:
        与输入等长的 list；原本可微的张量原样返回，否则返回其可微副本。
    """
    result = []
    for t in tensors:
        if not t.requires_grad:
            t = t.detach().requires_grad_(True)
        result.append(t)
    return result


def _traceable_args(args):
    """把实参序列拆成可微副本与位置映射，非张量参数原样透传。

    参数:
        args: 完整实参序列，可能包含非张量参数。

    返回:
        (call_args, primals, slot_of)：call_args 为把张量替换成可微副本后的
        实参列表；primals 为参与求导的叶子张量序列；slot_of[i] 是第 i 个实参
        在 primals 中的下标，非张量参数为 None。
    """
    call_args = list(args)
    primals = []
    slot_of = {}
    for i, arg in enumerate(args):
        if isinstance(arg, TN):
            slot_of[i] = len(primals)
            if not arg.requires_grad:
                arg = arg.detach().requires_grad_(True)
            primals.append(arg)
            call_args[i] = arg
        else:
            slot_of[i] = None
    return call_args, primals, slot_of


def _call_and_split(func, args):
    """执行 func 并把输出规范化为 (out_list, is_single_output)。

    参数:
        func: 待调用的函数。
        args: 传给 func 的位置参数序列。

    返回:
        (out_list, is_single_output)：out_list 是输出张量列表，逐个校验类型，
        便于后续按输出维度循环；is_single_output 表示 func 是否只返回一个张量。
    """
    outputs = func(*args)
    is_single_output = isinstance(outputs, TN)
    out_list = [outputs] if is_single_output else _as_list(outputs)
    for i, out in enumerate(out_list):
        if not isinstance(out, TN):
            raise RuntimeError(f"Output {i} must be a tensor")
    return out_list, is_single_output


def _split_aux(func, has_aux: bool):
    """按 has_aux 包装 func：剥离 (output, aux) 中的 aux 并缓存。

    只有 output 参与微分，aux 在最后一次前向求值时被记录，供调用方原样返回。
    未启用 has_aux 时原样返回 func 与 None。

    参数:
        func: 待包装的函数；has_aux=True 时必须返回 (output, aux)。
        has_aux: 是否剥离辅助输出。

    返回:
        (wrapped, aux_cell)：wrapped 只返回 output；aux_cell 为长度至多 1 的
        列表（首次前向时写入 aux），未启用时为 None。
    """
    if not has_aux:
        return func, None
    aux_cell = []

    def _wrapped(*args, **kwargs):
        output, aux = func(*args, **kwargs)
        if not aux_cell:
            aux_cell.append(aux)
        return output

    return _wrapped, aux_cell


def _vjp_with_output(out: TN, inputs, cotangent: TN, strict: bool):
    """对已求值的前向输出做一次 VJP，复用其计算图。

    VJP 关于 cotangent 是线性的：对固定的前向输出，可反复以不同 cotangent
    调用，无需重跑 func。jacrev 的每个输出基向量因此共享同一次前向计算。
    结果始终保留计算图（create_graph=True），因此可对 Jacobian 继续求导。

    参数:
        out: 已求值的前向输出节点。
        inputs: 目标输入张量序列。
        cotangent: 输出侧余切，形状与 out 一致。
        strict: 为 True 时，若某个输入与输出完全不连通（梯度为 None）则报错。

    返回:
        与 inputs 等长的 list；不连通输入以同形零张量补齐。
    """
    if not out.requires_grad:
        # 常量输出与所有输入都不连通：strict 时报错，否则返回同形零块，
        # 避免触发 grad 对不可求导输出根的校验错误。
        if strict:
            raise RuntimeError(
                "Output is independent of input (0), "
                "Set allow_unused=True if this is the desired behavior.")
        return [zeros_like(inp, requires_grad=True) for inp in inputs]
    grads = _grad(out, inputs, grad_outputs=cotangent,
                  create_graph=True, allow_unused=True)
    if strict:
        for j, gv in enumerate(grads):
            if gv is None:
                raise RuntimeError(
                    f"Output is independent of input ({j}), "
                    f"Set allow_unused=True if this is the desired behavior.")
    return [zeros_like(inp, requires_grad=True) if gv is None else gv
            for inp, gv in zip(inputs, grads)]

def _make_shadow(node: TN, leaf_map: dict) -> TN:
    """构造计算图节点的"局部影子"。

    影子节点与真实节点共享底层数据，其中 TN 来源替换为叶子副本，
    非 TN 原始操作数保持不变，因此在影子上求局部 VJP 的转置时只会遍历当前算子。

    参数:
        node: 前向图中的真实节点。
        leaf_map: id(原始张量) -> 叶子副本 的映射，保证同一张量在多处被引用时
            共用同一个叶子副本。

    返回:
        影子节点：共享 node._array 与 node.parms，TN 来源为无计算图的叶子，
        非 TN 来源保留原值。
    """
    shadow = TN(node._array, requires_grad=False)
    shadow.parms = node.parms
    fromvars = []
    for var in node.fromvars:
        if not isinstance(var, TN):
            fromvars.append(var)
            continue
        var_shadow = leaf_map.get(id(var))
        if var_shadow is None:
            # 叶子副本仍需 requires_grad=True：梯度规则按来源是否需要梯度决定
            # 是否返回梯度，若为 False 会得到 None，导致局部 JVP 无法进行。
            # 这里它没有 fromvars/_grad_rule，反向仍不会越过当前算子。
            var_shadow = TN(var._array, requires_grad=True)
            leaf_map[id(var)] = var_shadow
        fromvars.append(var_shadow)
    shadow.fromvars = tuple(fromvars)
    return shadow


def _local_vjp(node: TN, target: TN, i: int) -> TN:
    """计算第 i 条入边的局部 VJP，即 J_iᵀ g_out。

    梯度规则统一为 (result) -> tuple，按 fromvars 位置返回各来源的梯度。
    target 可能是影子节点（共享数据、来源换成叶子），因此规则取自真实节点
    node，而作用对象是 target。
    """
    rule = node._grad_rule
    if rule is None:
        raise RuntimeError(
            f"node {type(node).__name__} has no gradient rule; "
            "cannot compute local JVP"
        )
    grads = rule(target)
    grad = grads[i] if i < len(grads) else None
    if grad is None:
        raise RuntimeError(f"gradient rule returned no gradient for source {i}")
    return grad


def _local_jvp_by_basis(node: TN, i: int, target: TN, tangent: TN) -> TN:
    """梯度规则未建立计算图时的回退：用输出基向量显式构造局部 Jacobian。

    全程使用张量运算（不经过 numpy），因此 CPU/CUDA 都适用：对输出的每个
    基向量 k 求局部 VJP 得到 J_i 的第 k 列，堆叠为 (numel_in, numel_out) 的
    转置矩阵，再与 tangent 相乘得到 J_i @ tangent。

    参数:
        node: 当前算子的输出节点，提供形状、dtype、device 与 numel。
        i: 入边下标，对应 node.fromvars[i]。
        target: 调用梯度规则的目标节点（真实节点或影子节点），其 _grad_value
            会被临时改写为各基向量。
        tangent: 与第 i 个输入同形的切向量。

    返回:
        与输出同形的张量：J_i @ tangent。
    """
    numel_out = node.numel()
    basis = eye(numel_out, dtype=node.dtype, device=node.device)
    cols = []
    saved = target._grad_value
    try:
        for k in range(numel_out):
            target._grad_value = basis[k].reshape(node.shape)
            cols.append(_local_vjp(node, target, i).reshape(-1))
    finally:
        target._grad_value = saved
    mat_t = stack(cols, dim=-1)
    # 局部 VJP 登记的是共轭转置形式（梯度规则中对局部导数取 conj），
    # 故此处用共轭转置还原 J；实数情形与 T 等价。
    out = mat_t.mH @ tangent.reshape(-1)
    return out.reshape(node.shape)


def _local_jvp(node: TN, shadow: Optional[TN], i: int, tangent: TN,
               create_graph: bool) -> TN:
    """计算单个节点的局部 JVP：J_i @ tangent。

    背景：每个算子只登记了梯度规则 node._grad_rule，它把 node 的余切 g_out
    （与输出同形）映射为 g_in = J_iᵀ g_out（与输入同形），并没有前向规则。这里用
    "对局部 VJP 再做一次反向"补出前向结果：

    1. 以零值 g_out 调梯度规则，只为得到 g_in = J_iᵀ g_out 这张小图；
    2. 小图对应线性映射 L(g) = J_iᵀ g，线性映射的 VJP 就是其转置，
       (J_iᵀ)ᵀ = J_i；
    3. 以 tangent 作为 g_in 的余切对 L 反向，得到 Lᵀ tangent = J_i @ tangent。

    注意角色互换：node 的余切始终是 g_out；tangent 只在内层反向里充当 g_in
    的余切，因此形状与输入一致，与输出同形的 g_out 无关。复数情形梯度规则
    登记的是 J_iᴴ，两次共轭随两次转置一起抵消，结果同样是 J_i。

    参数:
        node: 前向图中的当前算子输出节点，node._grad_rule 给出各入边的
            局部 VJP 规则。
        shadow: node 的局部影子节点（共享底层数据、fromvars 换成叶子副本），
            让内层反向只遍历当前算子；为 None 时退化为 node 本身。
        i: 入边下标，同时索引 node.fromvars[i] 与梯度规则返回值。
        tangent: 沿该入边传入的切向量，形状与第 i 个输入一致。
        create_graph: 为 True 时保留内层反向的计算图以支持高阶导数，此时必须
            使用真实节点（影子节点会与原始输入断链）。

    返回:
        与第 i 个输入同形的张量，即该条边上的局部 JVP：J_i @ tangent。
    """
    # create_graph=True 时必须真实节点上建图：若用影子节点，g_in 与原始输入
    # 断开连接，高阶导数会丢失。其余情况用影子节点，使局部反传只遍历当前算子，
    # 不回溯整张计算图。
    target = node if create_graph else shadow
    if target is None:
        target = node
    # 局部 VJP 关于输出余切是线性的，这里用零值 seed 只为拿到 g_in 的图结构
    g_out = zeros_like(node, requires_grad=True)
    saved = target._grad_value
    target._grad_value = g_out
    try:
        g_in = _local_vjp(node, target, i)
    finally:
        target._grad_value = saved

    if not isinstance(g_in, TN):
        raise RuntimeError("gradient rule must return a tensor")
    if g_in.requires_grad:
        # 常规路径：梯度规则建了图，再反传一次即得 J_i @ tangent
        return _grad(g_in, [g_out], grad_outputs=tangent,
                     create_graph=create_graph, allow_unused=True)[0]
    # 回退路径：梯度规则未建图（如梯度开关关闭），只能显式构造局部 Jacobian
    return _local_jvp_by_basis(node, i, target, tangent)


def _graph_topological_order(
        outputs: Union[TN, list, tuple]) -> list:
    """返回前向 AD（JVP）所需的前向计算图拓扑序：输入在前、输出在后。

    以 `fromvars` 为唯一的边定义，从 outputs 出发做一次迭代式 DFS（显式栈，
    不递归）。不读写 `_rcv_grad_count` 等反向传播状态，纯图遍历、无副作用。
    outputs 可以是单个 TN 或 TN 的 list/tuple，重复节点按 id 去重；`fromvars`
    中的非 TN 原始操作数不是图节点，直接跳过。

    这里不关心节点来自内置算子还是装饰器：注册规则的一方负责把“影响前向值的
    全部 TN 实参”写进 `fromvars`（与内置算子一致），反向传播再按各来源的
    requires_grad 决定是否累计梯度。因此本函数无需了解任何规则实现的细节。

    算法：每个节点入栈两次
      1. `(node, False)`：展开节点，把它的 `fromvars` 压到栈上；
      2. `(node, True)` ：收尾标记。由于栈是后进先出，且标记先压、输入后压，
         标记一定在它的全部输入处理完之后才弹出；此时才把 node 追加到结果。
    因此结果是“完成序”：任何节点的全部输入都先于它入列，即天然合法的
    前向拓扑序（输入在前、输出在后）。规则调用依赖这一顺序——节点必须在
    它的全部输入之后才调用 `jvp_rule`；但拓扑序不唯一，同层兄弟之间、
    多个根之间的先后不影响正确性，规则本身按位置从 fromvars/parms 取值，
    不依赖具体遍历出的序列。

    参数:
        outputs: 前向输出节点，单个 TN 或 TN 序列。

    返回:
        list[TN]：从 outputs 可达的全部图节点，按“输入在前、输出在后”排列。
    """
    if isinstance(outputs, TN):
        stack = [(outputs, False)]
    else:
        # 多个根：先按 id 去重，避免同一输出被重复展开。
        # 此时 emitted 还是空的，所以这里单独用 seen。
        stack = []
        seen = set()
        for out in outputs:
            if id(out) not in seen:
                seen.add(id(out))
                stack.append((out, False))

    # emitted 的语义是“已写入 order”，不是“已访问过”：
    # 节点被多个后继或多个输出共享时会多次入栈，只有在收尾标记弹出、
    # 真正 append 之后才标记，保证它只出现一次，且位置在其全部输入之后。
    # 若在首次发现时就标记，后续共享路径会跳过它，导致缺失或顺序错误。
    order = []
    emitted = set()
    while stack:
        node, expanded = stack.pop()
        node_id = id(node)

        if expanded:
            # 收尾标记：该节点的输入都已处理完，可以入列
            if node_id not in emitted:
                emitted.add(node_id)
                order.append(node)
            continue

        # 已入列（可能由另一条共享路径先完成），无需再展开
        if node_id in emitted:
            continue

        # 先压收尾标记，再压输入，保证输入先于标记弹出
        stack.append((node, True))
        for var in node.fromvars:
            # 非 TN 操作数（如 pow 的指数）不是图节点；
            # 已入列的输入不必再压栈（这仅是减少入栈，正确性由弹出时的检查保证）
            if isinstance(var, TN) and id(var) not in emitted:
                stack.append((var, False))

    return order


def _prepare_forward(run, primals, create_graph: bool):
    """执行一次前向求值，并缓存可复用的计算图结构。

    jacfwd 的每个输入基向量共享同一次前向与同一批影子节点，
    从而避免为每个基向量重复构造相同的计算图。

    参数:
        run: 无参可调用对象，执行真实的 func 前向并返回单个 TN 或 TN 序列。
        primals: 参与播种的可微输入张量序列（叶子在前）。
        create_graph: 为 False 时预先构建影子节点；为 True 时跳过（局部 JVP 需要
            在真实节点上建图）。

    返回:
        dict，键为：
        primals: 参与播种的（可微）叶子张量
        out_list / is_single_output: 前向输出
        order: 拓扑序（叶子在前、输出在后）
        shadow_map: id(node) -> 局部影子节点（仅 create_graph=False 时构建）
    """
    out_list, is_single_output = _call_and_split(run, ())

    order = _graph_topological_order(out_list)
    shadow_map = None
    if not create_graph:
        # create_graph=True 时不能建影子：局部 JVP 必须在真实节点上建图，
        # 否则梯度与原始输入断开，高阶导数失效。
        # leaf_map 保证同一原始张量只生成一个叶子副本，
        # 使影子节点内部对同一输入的多处引用保持一致。
        leaf_map = {}
        shadow_map = {}
        for node in order:
            if node.fromvars:
                shadow_map[id(node)] = _make_shadow(node, leaf_map)
    return {
        'primals': list(primals),
        'out_list': out_list,
        'is_single_output': is_single_output,
        'order': order,
        'shadow_map': shadow_map,
    }


def _jvp_with_prepared(prepared, tangents, create_graph: bool, strict: bool):
    """在已准备好的前向计算图上传播一组 tangent。

    tangents 与 prepared['primals'] 一一对应，元素为 None 表示该输入的
    tangent 为零（跳过，不做任何传播）。

    参数:
        prepared: _prepare_forward 的返回值（前向输出、拓扑序、影子节点等）。
        tangents: 切向量序列，与 prepared['primals'] 一一对应；None 表示零方向。
        create_graph: 是否为高阶导数保留内层反向的计算图。
        strict: 为 True 时，若某输出与某输入完全不连通则报错。

    返回:
        单输出时返回 (输出张量, tangent 张量)；多输出时返回
        (tuple[输出], tuple[tangent])，与输入输出数量对应。
    """
    primals = prepared['primals']
    out_list = prepared['out_list']
    shadow_map = prepared['shadow_map']

    # 播种：同一个张量对象可能占据多个输入位置（如 f(x, x)），
    # 此时 tangent 必须相加；seed_mask 的第 k 位记录“该节点依赖第 k 个输入”，
    # 供 strict 独立性检查使用。
    seed_tangent = {}
    seed_mask = {}
    for k, (inp, vec) in enumerate(zip(primals, tangents)):
        if vec is None:
            continue
        inp_id = id(inp)
        if inp_id in seed_tangent:
            seed_tangent[inp_id] = seed_tangent[inp_id] + vec
            seed_mask[inp_id] |= 1 << k
        else:
            seed_tangent[inp_id] = vec
            seed_mask[inp_id] = 1 << k

    # 沿拓扑序（叶子 -> 输出）正向传播；每个节点的 tangent 由其各条入边
    # 的局部 JVP 结果累加得到（链式法则在多入边时表现为求和）。
    # node_mask 是依赖关系位掩码：只要任一输入的 bit 被置位，输出就继承。
    node_tangent = {}
    node_mask = {}
    for node in prepared['order']:
        node_id = id(node)
        if node_id in seed_tangent:
            node_tangent[node_id] = seed_tangent[node_id]
            node_mask[node_id] = seed_mask[node_id]
            continue

        acc = None
        mask = 0
        if node.fromvars:
            shadow = None if shadow_map is None else shadow_map.get(node_id)
            for i, var in enumerate(node.fromvars):
                if not isinstance(var, TN):
                    continue
                var_id = id(var)
                mask |= node_mask.get(var_id, 0)
                tangent = node_tangent.get(var_id)
                if tangent is None:
                    # 该入边 tangent 为零：跳过，不必构造 zeros_like 参与运算
                    continue
                local = _local_jvp(node, shadow, i, tangent, create_graph)
                acc = local if acc is None else acc + local
        node_tangent[node_id] = acc
        node_mask[node_id] = mask

    if strict:
        # strict 语义（与 torch.func 一致）：某个输出对某个输入完全不依赖时报错。
        # 这里用“输出节点的依赖位掩码中第 k 位是否为 1”来判断。
        for k in range(len(primals)):
            for j, out in enumerate(out_list):
                if not (node_mask.get(id(out), 0) >> k) & 1:
                    raise RuntimeError(
                        f"Output {j} is independent of input {k}")

    # 与输入无关的输出，其 tangent 为零张量
    results = []
    for out in out_list:
        tangent = node_tangent.get(id(out))
        results.append(zeros_like(out) if tangent is None else tangent)

    if prepared['is_single_output']:
        return out_list[0], results[0]
    return tuple(out_list), tuple(results)


def jvp2(func: Callable[..., Union[TN, tuple[TN, ...]]],
         primals: Union[TN, list[TN], tuple[TN, ...]],
         tangents: Union[TN, list[TN], tuple[TN, ...]],
         create_graph: bool = False,
         strict: bool = False) -> tuple[Union[TN, tuple[TN, ...]], Union[TN, tuple[TN, ...]]]:
    """Jacobian-vector 积（前向模式，基于局部 VJP 转置的旧实现）。

    先执行 func 的前向计算，再沿计算图从输入到输出正向传播 tangent，
    得到 J(x)·v。返回 (输出值, JVP 结果)。

    参数:
        func: 待求方向导数的函数，输入为 TN，输出为单个 TN 或 TN 序列。
        primals: 求值点输入，单个 TN 或 TN 序列，顺序与 func 的入参一致。
        tangents: 切向量，结构须与 primals 一致；仅当 primals 为单个输入时可传
            None，等价于以全 1 向量作为方向（0 维输入即标量 1.0）。
        create_graph: 为 True 时保留计算图，JVP 结果可继续求导（高阶导数）。
        strict: 为 True 时，若某输出与某输入完全不连通则抛 RuntimeError。

    返回:
        (输出值, JVP 结果)。输出值结构与 func 的返回一致（单输出为 TN，
        多输出为 tuple）；JVP 结果即 J(x)·v，结构与输出值一致。
    """
    is_single_input = not isinstance(primals, (list, tuple))
    if is_single_input:
        primals = [primals]
        if tangents is None:
            # 全 1 方向：0 维输入即标量 1.0，向量输入则逐元素为 1
            tangents = [ones_like(primals[0], requires_grad=create_graph)]
        else:
            tangents = [tangents]
    else:
        primals = list(primals)
        if tangents is None:
            raise RuntimeError("v must be provided when func has multiple inputs")
        tangents = list(tangents)

    if len(primals) != len(tangents):
        raise RuntimeError(f"Expected {len(primals)} tangents, got {len(tangents)}")
    for i, (inp, vec) in enumerate(zip(primals, tangents)):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
        if not isinstance(vec, TN):
            raise RuntimeError(f"v[{i}] must be a tensor")
        if inp.shape != vec.shape:
            raise RuntimeError(
                f"Shapes of input {i} ({inp.shape}) and v[{i}] ({vec.shape}) do not match")

    traceable = _ensure_traceable(primals)
    prepared = _prepare_forward(lambda: func(*traceable), traceable, create_graph)
    return _jvp_with_prepared(prepared, tangents, create_graph, strict)


def _seed_primals_for_jvp(primals, create_graph):
    """为 jvp 准备输入：按 create_graph 决定是否把输入转成可微叶子。

    背景：一次 jvp 分两个阶段——PRIMAL 阶段记录原函数的图，TANGENT 阶段
    传播切向量。前向请求存在期间始终记录计算图（见 forward_ad_trace_active），
    与输入的 requires_grad 无关。因此本函数不做“为了记录 primal 图”而改变
    输入，只在 create_graph=True 时为 tangent 表达式补上可微性：

        tangent 形如 3*x^2*v（由 primal 与切向量参与构造）。若 x 不可微，
        该表达式的 TN 节点也不会 requires_grad，之后无法再用 func.grad 等
        求二阶导；把 x 换成 requires_grad=True 的叶子后，这些节点才会按普通
        反向图规则记录 fromvars/_grad_rule，tangent 即可继续求导。

    分支逻辑：
      1. create_graph=False：原样返回。tangent 只做纯数值传播，不改变输入
         的可微性；不可微输入的 tangent 不带梯度追踪，与 torch.func.jvp 一致。
         （若用户传入的输入本身 requires_grad=True，其 tangent 仍会因普通
         反向图记录而带图——这与 torch 相同，可导性继承自 primal。）
      2. create_graph=True 且在顶层（无外层前向上下文）：仅把“纯叶子”
         （requires_grad=False、无 fromvars、无 _grad_rule）转成可微副本。
         带 fromvars/_grad_rule 的节点已有自己的图路径，无需替换。
      3. create_graph=True 且处于外层前向上下文：原样使用。外层已经记录了
         primal 图，替换输入会产生新的叶子、切断外层 JVP 的依赖链，
         导致嵌套 JVP 的高阶结果错误。

    参数:
        primals: 已校验的输入张量序列。
        create_graph: 是否保留 tangent 的计算图（只影响可导性，不影响数值）。

    返回:
        与输入等长的 list；create_graph=True 且处于顶层时可能包含替换后的
        可微叶子，其余情况与输入元素相同。
    """
    # create_graph=False：不播种，避免改变输入可微性、也避免给 tangent 附加
    # 用户没有要求的图（与 torch.func.jvp 对不可微 primal 的行为一致）。
    if not create_graph:
        return list(primals)
    # 外层已在前向上下文中：原样透传，保留外层 JVP 的依赖链。
    outer_trace = forward_ad_trace_active()
    result = []
    for tensor in primals:
        if (not tensor.requires_grad and not outer_trace
                and not tensor.fromvars and tensor._grad_rule is None):
            # 纯叶子替换成可微副本（不修改用户传入的张量本身）。
            tensor = tensor.detach().requires_grad_(True)
        result.append(tensor)
    return result


def _propagate_jvp(frame: JVPFrame, order, primals, tangents,
                   strict: bool, out_list):
    """JVP 的核心驱动：沿前向拓扑序把输入 tangent 传播到每个节点与输出。

    设计意图
    --------
    前向模式 AD 只需一次前向遍历。对每条边都成立的链式法则是
        tangent(node) = Σ_j ∂node/∂in_j · tangent(in_j)
    每个算子的 ``_jvp_rule`` 就是其“局部 JVP”（给定入边 tangent 求本节点
    tangent）。本函数负责按拓扑序组织这些局部计算：整次传播是一次 O(V+E)
    的图扫描，每个节点的规则至多调用一次（≤V 次），每条边的 tangent 只被
    其后继节点的规则读取一次。

    关键设计
    --------
    1. tangent 统一存放在 ``frame.node_tangent``（id(node) -> TN|None），不写回
       TN 的持久字段：一次调用（或 jacfwd 的一列方向）的中间状态只属于本次
       frame，换方向时清空 frame 即可重放；多个方向共享同一张 primal 图
       （见 _jvp_with_primal_graph）；嵌套 JVP 时内外层各持一个 frame，互不
       覆盖。唯一例外是原地步骤，它按 frame 登记在节点上（见第 5 条）。
    2. 拉取式（pull-based）逐节点求值：本函数按 order 遍历，轮到某个节点时才
       调用它的 jvp_rule，从“已就位的入边 tangent”拉取出本节点 tangent。
       规则内部调用的普通算子只做数值运算（前向请求存在时会按普通反向规则
       记录图），JVP 传播只由本函数驱动，因此不会重入，也不会重复计算。
    3. 规则保持无副作用：规则只读入边 tangent 与节点自身的 primal 入参
       （fromvars），返回本节点 tangent（None 表示零）；写回 node_tangent、
       判断“是否需要调用规则”等簿记工作集中在此，规则代码与
       register_grad_rule 一样简洁。
    4. None 即零方向：若某节点的全部入边 tangent 均为 None（且没有待重放的
       原地步骤），其 tangent 必为零，直接跳过规则调用；下游节点也因此只剩
       None 入边，规则调用被逐层跳过。带原地步骤的节点例外——原地写入的
       右值 tangent 可能非零，必须先重放步骤。最终输出若无 tangent，返回
       同形 zeros_like，对外始终是“与输出同形的 tangent 张量”。
    5. 原地修改：原地算子把操作步骤按“产生它的 PRIMAL frame”登记在节点上
       （见 _jvp_append_step）；本函数在该节点的拓扑位置只重放属于本 frame
       的步骤。调用规则前临时剥离这些步骤追加在 fromvars 前缀的来源
       （r_n, …, r_1），保证规则看到的是本次前向的真实入参，调用后再恢复。
       语义与反向模式一致（细节见 _apply_inplace_jvp_steps）。
    6. strict：额外用位掩码 dependency_mask 记录“节点依赖哪些输入”——第 k 位
       为 1 表示该节点依赖第 k 个输入。掩码沿拓扑序逐节点按位或传递（并集），
       传播结束后再检查每个输出对每个输入的依赖位；不连通则抛 RuntimeError，
       关闭时该情形按数学语义贡献 0，且完全不建掩码。Python int 为任意精度，
       不受输入个数限制。

    参数:
        frame: 本次 JVP 调用的状态（node_tangent / dependency_mask 等）；进入
            本函数前已切换到 TANGENT 阶段（见 _jvp_with_primal_graph），因此
            规则内构造的运算会继续记录图。
        order: _graph_topological_order 给出的前向拓扑序（list[TN]），按 id
            去重、每个节点只出现一次。性质：
            - 任何节点的 TN 入边都排在它前面，因此轮到某节点时其全部入边
              tangent 已计算完毕，jvp_rule 可直接拉取使用（这是“一次扫描”
              完成传播的前提）；
            - 输入在最前、输出在最后；被多条路径共享的节点只出现一次，
              其规则因此至多调用一次。
        primals: 播种输入张量序列（与 tangents 等长，同一张量可占据多个位置）。
        tangents: 与 primals 一一对应的切向量；None 表示该位置的方向为零。
        strict: 为 True 时，某输出对某输入完全不依赖会报错。
        out_list: 前向输出张量列表。

    返回:
        与 out_list 等长的 list，元素为各输出的 tangent（无贡献时为同形零张量）。
    """
    # 播种：把输入切向量写入 node_tangent。同一张量对象可能占据多个输入位置
    # （如 f(x, x)），此时它对输出方向的贡献是各切向量之和，必须累加；
    # strict 下同步记录依赖位：第 k 位（1 << k）表示“该节点依赖第 k 个输入”，
    # 多个输入位置命中同一张量时用按位或合并（两个输入的依赖都要保留）。
    for k, (inp, vec) in enumerate(zip(primals, tangents)):
        if vec is None:
            continue
        inp_id = id(inp)
        existing = frame.node_tangent.get(inp_id)
        if existing is None:
            frame.node_tangent[inp_id] = vec
            if strict:
                frame.dependency_mask[inp_id] = 1 << k
        else:
            frame.node_tangent[inp_id] = existing + vec
            if strict:
                frame.dependency_mask[inp_id] |= 1 << k

    # 主扫描：拉取式（pull-based）传播。order 中每个节点只出现一次，因此
    # 轮到某个非播种节点时，其入边 tangent 必已就位，可直接调用规则。
    for node in order:
        node_id = id(node)
        if node_id in frame.node_tangent:
            # order 按 id 去重，能在这里命中的只有播种输入：tangent 已就位，
            # 无需再调用规则；但原地节点仍须把属于本 frame 的步骤施加到它上面。
            steps = _frame_inplace_steps(node, frame)
            if steps:
                if strict:
                    # 播种输入的掩码已在函数开头的播种处设置；原地写入还会把
                    # 右值的依赖带入本节点，这里取“原有依赖集合”与“右值节点
                    # 依赖集合”的并集。
                    for _name, record in steps:
                        right = record.right
                        if isinstance(right, TN):
                            frame.dependency_mask[node_id] = (
                                frame.dependency_mask.get(node_id, 0)
                                | frame.dependency_mask.get(id(right), 0))
                _apply_inplace_jvp_steps(node, frame)
            continue

        steps = _frame_inplace_steps(node, frame)
        rule = node._jvp_rule
        if strict:
            # 位掩码的传递：第 k 位为 1 表示“节点依赖第 k 个输入”。一个节点的
            # 依赖集合 = 所有 TN 入边依赖集合的并集，用按位或实现（O(入度)）。
            # order 是拓扑序，入边掩码必已就位，多跳依赖逐层累积，因此这里是
            # 全局依赖而非单层依赖。播种输入与含原地步骤的节点不会走到这里，
            # 它们的掩码分别在播种处和上方跳过分支中设置。
            mask = 0
            for var in node.fromvars:
                if isinstance(var, TN):
                    mask |= frame.dependency_mask.get(id(var), 0)
            frame.dependency_mask[node_id] = mask

        # 快速判定本节点是否存在非零入边方向：任一入边 tangent 存在就需要
        # 调用规则；全部为 None 时可安全跳过（零方向剪枝）。
        has_tangent = False
        for var in node.fromvars:
            if isinstance(var, TN) and frame.node_tangent.get(id(var)) is not None:
                has_tangent = True
                break

        if not has_tangent and not steps:
            # 全部入边方向为零且无原地步骤：本节点 tangent 必为零，跳过规则调用
            frame.node_tangent[node_id] = None
            continue

        # 有原地步骤时，节点的 fromvars 前追加了被修改的来源（r_n, …, r_1），
        # 而 _jvp_rule 是按节点创建时的位置读取 fromvars 的；调用规则前先剥离
        # 前缀，调用后恢复，保证规则看到的是本次前向的真实入参。
        saved_fromvars = node.fromvars
        if steps:
            node.fromvars = saved_fromvars[len(steps):]
        try:
            if not has_tangent:
                # base tangent 为零方向，但仍需在下面重放原地步骤
                # （原地写入的右值 tangent 可能非零）。
                tangent = None
            else:
                if rule is None:
                    # 与 torch.func.jvp 一致：不支持前向 AD 的节点直接报错，
                    # 不做局部 VJP 回退。内置算子均已登记规则，只有仅实现
                    # 反向的自定义算子需要 register_jvp_rule 或 Function.jvp。
                    raise RuntimeError(
                        f"jvp: node (shape={node.shape}) has no forward (jvp) rule; "
                        "custom operators must define jvp_rule (register_jvp_rule) "
                        "or Function.jvp to support forward-mode AD")
                tangent = rule(frame, node)
            frame.node_tangent[node_id] = tangent
        finally:
            if steps:
                node.fromvars = saved_fromvars

        if steps:
            # 在 base tangent 就位后，按时间顺序重放本 frame 的原地修改。
            _apply_inplace_jvp_steps(node, frame)

    if strict:
        # 连通性判定放在传播之后：不能依据 tangent 是否为零来判断，因为零切线
        # 也会使 tangent 为 0；必须依赖整张图累积出的依赖位。逐输出逐输入取
        # 第 k 位：为 0 说明该输出与第 k 个输入之间没有任何连通路径，strict 下
        # 视为调用方“输出依赖全部输入”的假设被违反，按 torch.func.jvp 语义报错。
        for k in range(len(primals)):
            for j, out in enumerate(out_list):
                if not (frame.dependency_mask.get(id(out), 0) >> k) & 1:
                    raise RuntimeError(
                        f"Output {j} is independent of input {k}")

    results = []
    for out in out_list:
        tangent = frame.node_tangent.get(id(out))
        # None 表示所有输入方向都未到达该输出：按 J·v = 0 返回同形零张量，
        # 保持对外返回结构稳定（调用方无需区分“零”与“无”）。
        results.append(zeros_like(out) if tangent is None else tangent)
    return results


def _record_primal(func, primals, create_graph: bool, strict: bool):
    """执行一次前向求值并记录可复用的 primal 图（线性化）。

    jvp 的单次调用与 jacfwd 的多列共享都走这里：图只建一次，之后每个方向
    只需重新播种并传播 tangent（见 _jvp_with_primal_graph）。

    参数:
        func: 待求值函数，按位置接收 primals 序列。
        primals: 参与前向的输入张量序列。
        create_graph: 是否把不可微输入播种为可微叶子，使 tangent 表达式可继续
            求导（只影响可导性，不影响数值）。
        strict: 输出与输入不连通时是否报错。

    返回:
        dict，键为 inputs（播种后的输入）、out_list / is_single_output（前向输出）、
        order（前向拓扑序）、frame（记录这张图的 JVPFrame）。
    """
    # 播种必须在压入新 frame 之前完成：嵌套调用时据此保留外层前向图
    inputs = _seed_primals_for_jvp(primals, create_graph)

    frame = JVPFrame(strict=strict, phase=PRIMAL)
    token = push_jvp_frame(frame)
    try:
        # PRIMAL 阶段：原函数前向求值，同时按前向请求记录 fromvars 与 jvp_rule
        out_list, is_single_output = _call_and_split(func, inputs)
        order = _graph_topological_order(out_list)
    finally:
        pop_jvp_frame(token)
    return {
        'inputs': inputs,
        'out_list': out_list,
        'is_single_output': is_single_output,
        'order': order,
        'frame': frame,
    }


def _prune_zero_tangent(vec):
    """把可证明为零的常量切向量归一为 None，使传播跳过该输入的整棵子树。

    只有“叶子常量”（不 requires_grad、无 fromvars）且数值全零时才可剪枝：
    这类方向对输出及其对 primals 的导数贡献恒为零，跳过不会改变任何结果。
    非常量切向量即使当前数值为零也不能剪，否则会切断对切向量自身的依赖。
    """
    if vec is None:
        return None
    if vec.requires_grad or vec.fromvars:
        return vec
    return None if not bool(vec._array.any()) else vec


def _jvp_with_primal_graph(state, tangents, strict: bool):
    """在已记录的 primal 图上传播一组 tangent，多个方向可反复调用。

    frame 状态（node_tangent / dependency_mask）不写入 TN 的持久字段，
    每列开始前清空即可；原地步骤按 frame 身份登记在节点上，仍能被
    _propagate_jvp 找到并重放。

    参数:
        state: _record_primal 的返回值。
        tangents: 切向量序列，与 state['inputs'] 一一对应；None 表示零方向，
            该输入对应的整棵子树会被跳过，不调用任何规则。非 strict 模式下，
            全零的常量切向量也会被自动归一为 None。
        strict: 输出与输入不连通时是否报错。strict 模式下不做零剪枝，
            以保持“某输入方向是否参与计算”的既有判定语义。

    返回:
        与 state['out_list'] 等长的 list，元素为各输出的 tangent。
    """
    if not strict:
        tangents = [_prune_zero_tangent(vec) for vec in tangents]
    frame = state['frame']
    frame.node_tangent.clear()
    frame.dependency_mask.clear()
    # 切到 TANGENT 后栈仍非空，trace 保持为真：jvp_rule 内部构造的 TN 运算
    # 会继续记录 fromvars/_jvp_rule，使 tangent 表达式可再求导。
    set_jvp_phase(frame, TANGENT)
    token = push_jvp_frame(frame)
    try:
        return _propagate_jvp(frame, state['order'], state['inputs'],
                              tangents, strict, state['out_list'])
    finally:
        pop_jvp_frame(token)
        # 传播结果已全部物化进返回值，这里立即清空本方向的中间表：下次方向
        # 开头同样会清空，提前释放可避免 frame 被长期引用（linearize 闭包、
        # 原地步骤 node._inplace_jvp_steps 都持有 frame 身份）时，把整张图的
        # tangent 及其数据一起留住。
        frame.node_tangent.clear()
        frame.dependency_mask.clear()


def jvp(func: Callable[..., Union[TN, tuple[TN, ...]]],
        primals,
        tangents,
        *,
        strict: bool = False,
        has_aux: bool = False) -> tuple[Union[TN, tuple[TN, ...]],
                                        Union[TN, tuple[TN, ...]],
                                        Any]:
    """Jacobian-vector 积（前向模式）。接口严格对齐 torch.func.jvp。

    参数:
        func: 待求方向导数的函数，输入为 TN，输出为单个 TN 或 TN 序列。
        primals: 求值点输入 tuple（单输入也须包成 1-tuple）。
        tangents: 切向量 tuple，长度与 primals 相同，每个元素是 TN，
            形状与对应 primal 一致。
        strict: 关键字参数。某输出与某输入完全不连通时抛 RuntimeError。
        has_aux: 关键字参数。func 返回 (output, aux) 时为 True，
            只对 output 求 JVP，aux 附在返回值末尾。

    返回:
        has_aux=False 时返回 (输出值, JVP 结果)；has_aux=True 时返回
        (输出值, JVP 结果, aux)。
    """
    if not isinstance(primals, tuple):
        raise RuntimeError(
            f"jvp(f, primals, tangents): Expected primals to be a tuple. "
            f"E.g. it should be valid to call f(*primals)."
        )
    if not isinstance(tangents, tuple):
        raise RuntimeError(
            f"jvp(f, primals, tangents): Expected tangents to be a tuple."
        )
    if len(primals) != len(tangents):
        raise RuntimeError(
            f"jvp(f, primals, tangents): Expected primals and tangents to have "
            f"the same python structure, got primals of length {len(primals)} and "
            f"tangents of length {len(tangents)}."
        )

    for i, (inp, vec) in enumerate(zip(primals, tangents)):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
        if not isinstance(vec, TN):
            raise RuntimeError(
                f"jvp(f, primals, tangents): Expected tangents to only contain Tensors, "
                f"got {type(vec).__name__} at position {i}"
            )
        if inp.shape != vec.shape:
            raise RuntimeError(
                f"Shapes of input {i} ({inp.shape}) and tangent[{i}] ({vec.shape}) do not match")

    wrapped, aux_cell = _split_aux(func, has_aux)
    state = _record_primal(wrapped, list(primals), True, strict)
    results = _jvp_with_primal_graph(state, list(tangents), strict)

    out_list = state['out_list']
    is_single_output = state['is_single_output']
    out_values = out_list[0] if is_single_output else tuple(out_list)
    tangent_values = results[0] if is_single_output else tuple(results)
    if has_aux:
        return out_values, tangent_values, aux_cell[0]
    return out_values, tangent_values

def vjp(func: Callable[..., Union[TN, tuple[TN, ...]]],
        *primals: TN,
        has_aux: bool = False):
    """vector-Jacobian 积（torch.func 风格惰性形态）。

    接口严格对齐 torch.func.vjp：先执行一次 func 的前向计算，返回
    ``(输出值, vjp_fn[, aux])``。随后可多次调用 ``vjp_fn(cotangents)``
    以不同 cotangent 计算 VJP，无需重跑前向。

    参数:
        func: 待求 VJP 的函数，输入为 TN，输出为单个 TN 或 TN 序列。
        *primals: 求值点输入，按 func 的位置参数顺序给出（必须都是 TN）。
        has_aux: 为 True 时 func 必须返回 ``(output, aux)`` 二元组，只对 output
            求 VJP，aux 在 ``vjp_fn`` 的闭包外附在返回值末尾。

    返回:
        ``(输出值, vjp_fn)`` 或 ``(输出值, vjp_fn, aux)``：
            - 输出值是 func 在前向处的求值结果（TN 或 TN 元组）；
            - ``vjp_fn(cotangents, retain_graph=True, create_graph=None)`` 是惰性闭包，
              输入 cotangent 结构须与输出一致，返回各 primal 的余切传播结果组成的
              tuple（与 torch.func 一致，单 primal 也返回 1-tuple）。
    """
    if not primals:
        raise RuntimeError("vjp requires at least one input")
    for i, inp in enumerate(primals):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")

    inputs = list(primals)
    inputs = _ensure_traceable(inputs)

    wrapped, aux_cell = _split_aux(func, has_aux)
    out_list, is_single_output = _call_and_split(wrapped, inputs)

    def vjp_fn(cotangents, retain_graph: bool = True,
               create_graph: Optional[bool] = None):
        """惰性 VJP 闭包：对捕获的前向图，以给定 cotangent 做一次反向传播。

        retain_graph 只为对齐 torch.func 签名而接受——本实现的图不主动释放，
        恒等价于 retain_graph=True。create_graph 为 None 时取当前梯度模式
        （与 torch.func 一致）。
        """
        if create_graph is None:
            create_graph = is_grad_enabled()
        return _vjp_on_outputs(out_list, inputs, cotangents, create_graph)

    out_values = out_list[0] if is_single_output else tuple(out_list)
    if has_aux:
        return out_values, vjp_fn, aux_cell[0]
    return out_values, vjp_fn


def _vjp_on_outputs(out_list: list[TN], inputs: list[TN],
                    cotangents, create_graph: bool):
    """对已求值的前向输出做一次 VJP。

    多输出时过滤掉不可求导的常量输出，合并为一次多根调用；未连通的输入按零块
    处理（与 torch.func 一致）。

    参数:
        out_list: 前向输出张量列表（持有计算图）。
        inputs: 目标输入张量序列。
        cotangents: 输出侧余切，单个 TN 或 list/tuple[TN]，结构须与输出一致。
        create_graph: 是否为高阶导数保留计算图。

    返回:
        tuple[TN]：每个输入一项，顺序与 inputs 一致。
    """
    kot = _as_list(cotangents)
    if len(kot) != len(out_list):
        raise RuntimeError(f"Expected {len(out_list)} cotangents, got {len(kot)}")
    cot_list = []
    for i, (out, cot) in enumerate(zip(out_list, kot)):
        if not isinstance(cot, TN):
            raise RuntimeError(f"cotangents[{i}] must be a tensor")
        if out.shape != cot.shape:
            raise RuntimeError(
                f"Shapes of output {i} ({out.shape}) and cotangents[{i}] ({cot.shape}) do not match")
        cot_list.append(cot)

    active = [(out, cot) for out, cot in zip(out_list, cot_list) if out.requires_grad]
    if active:
        grads = _grad([out for out, _ in active], inputs,
                      grad_outputs=[cot for _, cot in active],
                      create_graph=create_graph, allow_unused=True)
        results = [zeros_like(inp, requires_grad=create_graph) if gv is None else gv
                   for inp, gv in zip(inputs, grads)]
    elif out_list:
        results = [zeros_like(inp, requires_grad=create_graph) for inp in inputs]
    else:
        results = [None] * len(inputs)

    return tuple(results)


def linearize(func: Callable[..., Union[TN, tuple[TN, ...]]],
              *primals: TN):
    """在 primals 处线性化 func，返回 (primal_out, linear_fn)。

    前向只求值一次并记录 primal 图；随后可用不同的切向量多次调用 linear_fn，
    每次沿同一张图传播 tangent，得到 J(x)·v。等价于对 func 做一次前向 AD 的
    “常量项 + 线性项”分解：``func(x + v) ≈ primal_out + linear_fn(v)``
    （忽略高阶项）。与 torch.func.linearize 对齐。

    参数:
        func: 待线性化的函数，输入为 TN，输出为单个 TN 或 TN 序列。
        *primals: 求值点输入，按 func 的位置参数顺序给出，必须都是 TN。

    返回:
        (primal_out, linear_fn)：
            - primal_out 与 func 的返回结构一致（单输出为 TN，多输出为 tuple）；
            - linear_fn(*tangents) 返回 J(x)·v，结构同 primal_out；
              tangents 的个数与形状必须和 primals 一一对应，元素可为 None
              （表示该输入方向为零，传播时跳过其整棵子树）。

    注意:
        linear_fn 对 tangents 是线性映射（只保留一阶依赖），因此结果可再求
        一阶导；对 linear_fn 再求二阶导恒为零张量（与 torch.func 一致）。
        需要真实高阶导数请对原始 func 使用 ``jvp`` / ``grad``。
    """
    if not primals:
        raise RuntimeError("linearize requires at least one input")
    for i, inp in enumerate(primals):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")

    state = _record_primal(func, list(primals), False, False)

    def linear_fn(*tangents):
        if len(tangents) != len(primals):
            raise RuntimeError(
                f"Expected {len(primals)} tangents, got {len(tangents)}")
        for i, (inp, vec) in enumerate(zip(primals, tangents)):
            if vec is not None and not isinstance(vec, TN):
                raise RuntimeError(f"v[{i}] must be a tensor or None")
            if vec is not None and inp.shape != vec.shape:
                raise RuntimeError(
                    f"Shapes of input {i} ({inp.shape}) and v[{i}] ({vec.shape}) do not match")
        results = _jvp_with_primal_graph(state, list(tangents), False)
        return results[0] if state['is_single_output'] else tuple(results)

    out_list = state['out_list']
    primal_out = out_list[0] if state['is_single_output'] else tuple(out_list)
    return primal_out, linear_fn


# ---------------------------------------------------------------------------
# jacrev / jacfwd
# ---------------------------------------------------------------------------

def _identity_basis(numel: int, shape, dtype, device) -> TN:
    """构造 numel 个标准基向量，形状为 (numel,) + shape。

    用一次 eye 填充整个 (numel, numel) 平面，避免 numel 次逐元素赋值。

    参数:
        numel: 展平后的元素个数，也是基向量个数。
        shape: 单个基向量的形状（即被展开张量的形状）。
        dtype: 数据类型。
        device: 计算设备。

    返回:
        形状为 (numel,) + shape 的张量，第 k 个切片为第 k 个单位基向量。
    """
    basis = zeros(numel, *shape, dtype=dtype, device=device)
    arrlib = basis._get_array_lib()
    basis._array.reshape(numel, numel)[...] = arrlib.eye(numel, dtype=dtype)
    return basis


def _identity_basis_chunk(start: int, count: int, flat_numel: int,
                          shape, dtype, device) -> TN:
    """构造 identity 矩阵的第 [start, start+count) 行，形状为 (count,) + shape。

    与 _identity_basis 不同，只分配当前分块所需的内存（count × numel），
    供 chunk_size 分块计算 Jacobian 时限制峰值内存。

    参数:
        start: 首个基向量在展平空间中的下标。
        count: 本块的基向量个数。
        flat_numel: 展平后的元素总数。
        shape: 单个基向量的形状。
        dtype: 数据类型。
        device: 计算设备。

    返回:
        形状为 (count,) + shape 的张量，第 i 个切片为第 start+i 个单位基向量。
    """
    basis = zeros(count, *shape, dtype=dtype, device=device)
    flat = basis._array.reshape(count, flat_numel)
    for i in range(count):
        flat[i, start + i] = 1
    return basis


def _jacobian_via_vjp(func, args, positions, strict: bool,
                      chunk_size: Optional[int] = None):
    """用 vjp 原语计算 Jacobian：对输出的每个基向量做一次 VJP。

    func 只执行一次，所有列共享同一张前向计算图；每列仅做一次独立反向遍历
    （见 _vjp_with_output）。只为 positions 选中的输入收集梯度，未选中的参数
    既不参与 strict 检查，也不产生多余的反向遍历。
    结果始终保留计算图（create_graph=True），因此可对 Jacobian 继续求导。

    参数:
        func: 待求 Jacobian 的函数。
        args: 完整实参序列（含非张量参数，非张量原样透传）。
        positions: 需要求导的参数位置序列。
        strict: 输出与输入不连通时是否报错。
        chunk_size: 每次处理多少个输出基向量；None 表示一次性处理全部。

    返回:
        (blocks, is_single_output)。blocks[k][j] 是第 k 个被选中输入对第 j 个
        输出的 Jacobian，形状为 output_shape + input_shape。
    """
    call_args, primals, slot_of = _traceable_args(args)

    inputs = []
    for pos in positions:
        slot = slot_of.get(pos)
        if slot is None:
            raise RuntimeError(f"Input {pos} must be a tensor")
        inputs.append(primals[slot])

    out_list, is_single_output = _call_and_split(func, call_args)

    blocks = [[None] * len(out_list) for _ in inputs]
    for j, out in enumerate(out_list):
        numel = out.numel()
        if numel == 0 or not out.requires_grad:
            # 不可微或空输出（如 sort 返回的 indices）没有梯度，直接给零块
            for k, inp in enumerate(inputs):
                blocks[k][j] = zeros(out.shape + inp.shape, dtype=out.dtype,
                                     device=out.device)
            continue

        def _single_vjp(cot, _out=out):
            # 复用已建立的前向计算图：VJP 关于 cotangent 线性，
            # 无需为每个输出基向量重跑 func。
            return _vjp_with_output(_out, tuple(inputs), cot, strict)

        # 按 chunk_size 分批构造输出基向量，每批一次 vmap(vjp)，最后沿列拼接。
        parts = [[] for _ in inputs]
        step = numel if chunk_size is None else chunk_size
        for start in range(0, numel, step):
            count = builtins.min(step, numel - start)
            if count == numel:
                cot_batch = _identity_basis(numel, out.shape, out.dtype, out.device)
            else:
                cot_batch = _identity_basis_chunk(
                    start, count, numel, out.shape, out.dtype, out.device)
            stacked = _as_list(vmap(_single_vjp)(cot_batch))
            for k, blk in enumerate(stacked):
                parts[k].append(blk)

        for k, (chunks, inp) in enumerate(zip(parts, inputs)):
            # 每个输入的分量形状为 (numel,)，还原成 Jacobian 块
            # out.shape + inp.shape；dtype 统一向输出对齐
            blk = chunks[0] if len(chunks) == 1 else cat(chunks, dim=0)
            blk = blk.reshape(out.shape + inp.shape)
            if blk.dtype != out.dtype:
                blk = blk.to(dtype=out.dtype)
            blocks[k][j] = blk

    return blocks, is_single_output


def _basis_column(inp: TN, k: int) -> TN:
    """构造与 inp 同形的第 k 个单位基向量，作为无计算图的 tangent 叶子。

    逐列按需分配（峰值内存 O(numel)），避免 _identity_basis 一次性分配
    numel×numel 的平面；tangent 只读，因此可直接写入底层数组。

    参数:
        inp: 决定形状的输入张量。
        k: 展平后的下标，取值 [0, inp.numel())。

    返回:
        与 inp 同形、requires_grad=False 的叶子张量。
    """
    basis = zeros_like(inp, requires_grad=False)
    basis._array.reshape(-1)[k] = 1
    return basis


def _jacobian_via_jvp(func, args, positions,
                      chunk_size: Optional[int] = None):
    """用直接前向规则的 jvp 原语计算 Jacobian：对每个输入基向量做一次 JVP。

    等价于 vmap(jvp)，并按 linearize 方式复用 primal 图：``func`` 只前向
    求值一次并记录图，随后每个输入元素播种一个 one-hot tangent，复用同一张
    图与同一份拓扑序传播得到 Jacobian 的一列，最后堆叠成完整 Jacobian。
    未选中的输入以 ``None`` 播种，其对应的子树会被整棵剪掉。func 需为纯函数
    （无随机性或副作用），与 torch.func 的约定一致。
    结果始终保留计算图（create_graph=True），因此可对 Jacobian 继续求导。

    参数:
        func: 待求 Jacobian 的函数。
        args: 完整实参序列（含非张量参数，非张量原样透传）。
        positions: 需要求导的参数位置序列。
        chunk_size: 每次处理多少个输入基向量；None 表示一次性处理全部。
            分块只影响列的堆叠方式，结果与不分块一致。

    返回:
        (blocks, is_single_output)。blocks[k][j] 是第 k 个被选中输入对第 j 个
        输出的 Jacobian，形状为 output_shape + input_shape。
    """
    call_args, primals, slot_of = _traceable_args(args)

    slots = []
    for pos in positions:
        slot = slot_of.get(pos)
        if slot is None:
            raise RuntimeError(f"Input {pos} must be a tensor")
        slots.append(slot)

    def _func_from_primals(*ps):
        # 把播种后的 primals 放回完整实参位置，非张量参数原样透传
        full = list(call_args)
        for arg_i, slot in slot_of.items():
            if slot is not None:
                full[arg_i] = ps[slot]
        return func(*full)

    # primal 图只建一次，所有列（含多个 argnums 位置）共享
    state = _record_primal(_func_from_primals, primals, True, False)
    out_list = state['out_list']
    n_primals = len(primals)

    blocks = []
    for slot in slots:
        inp = primals[slot]
        numel = inp.numel()
        if numel == 0:
            blocks.append([zeros(out.shape + inp.shape, dtype=out.dtype,
                                 device=out.device) for out in out_list])
            continue

        cols = [[] for _ in out_list]
        step = numel if chunk_size is None else chunk_size
        for start in range(0, numel, step):
            stop = builtins.min(start + step, numel)
            local = [[] for _ in out_list]
            for k in range(start, stop):
                tangents = [None] * n_primals
                tangents[slot] = _basis_column(inp, k)
                results = _jvp_with_primal_graph(state, tangents, False)
                for j, res in enumerate(results):
                    local[j].append(res)
            # 每块内沿最后一维堆叠为 (*out.shape, count)，块间再沿最后一维拼接
            for j, chunk in enumerate(local):
                cols[j].append(stack(chunk, dim=-1))

        # 每列结果的形状与 out 相同；沿最后一维堆叠得到 (*out.shape, numel)，
        # 再 reshape 成 (*out.shape, *inp.shape)，即第 k 个输入维度在末尾。
        blocks.append([(col[0] if len(col) == 1 else cat(col, dim=-1)
                        ).reshape(out.shape + inp.shape)
                       for col, out in zip(cols, out_list)])

    return blocks, state['is_single_output']


def _format_jacobian(blocks, is_single_output: bool):
    """把 blocks[k][j] 整理为公开返回结构。

    参数:
        blocks: 每个被选中输入位置对各输出的 Jacobian，blocks[k][j] 的形状为
            output_shape + input_shape。
        is_single_output: func 是否只返回一个输出。

    返回:
        单输出且只选中 1 个位置时返回单个矩阵；否则返回 tuple。多输出时外层
        按输出顺序、内层按 argnums 顺序；单输出多位置时直接返回位置元组。
    """
    n_pos = len(blocks)
    if is_single_output:
        per_output = [(blocks[k][0],) for k in range(n_pos)]
    else:
        per_output = [tuple(blocks[k][j] for k in range(n_pos))
                      for j in range(len(blocks[0]))]

    if len(per_output) == 1:
        inner = per_output[0]
        return inner[0] if len(inner) == 1 else inner
    return tuple(inner[0] if len(inner) == 1 else inner for inner in per_output)


def _make_jacobian(compute_blocks, func, argnums, allow_none: bool,
                   has_aux: bool = False, **opts):
    """构造 jacrev / jacfwd 共用的导函数：校验 argnums、算 blocks、统一格式化。

    参数:
        compute_blocks: 底层原语，签名为 (func, args, positions, **opts)，
            返回 (blocks, is_single_output)。
        func: 待求 Jacobian 的函数。
        argnums: 需要求导的参数位置，int、tuple[int, ...]；allow_none 时可为 None。
        allow_none: 是否允许 None 表示全部参数（jacfwd 支持，与 torch.func 一致）。
        has_aux: 为 True 时 func 返回 (output, aux)，结果返回 (jacobian, aux)。
        **opts: 透传给 compute_blocks 的选项（strict / chunk_size）。

    返回:
        Callable：接受与 func 相同的参数，返回 _format_jacobian 整理后的结果。
    """
    chunk_size = opts.get('chunk_size')
    if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size <= 0):
        # 仅 jacrev 会传 chunk_size、且 torch.func.jacrev 在建函数时即校验
        raise ValueError("jacrev: `chunk_size` should be greater than 0.")

    def _wrapper(*args, **kwargs):
        an = _check_argnums(argnums, allow_none=allow_none)
        positions = tuple(range(len(args))) if an is None else an
        wrapped, aux_cell = _split_aux(lambda *a: func(*a, **kwargs), has_aux)
        blocks, is_single_output = compute_blocks(
            wrapped, args, positions, **opts)
        result = _format_jacobian(blocks, is_single_output)
        # argnums=None 表示“对全部参数求导”，此时即使只有 1 个参数也返回元组
        if an is None and not isinstance(result, tuple):
            result = (result,)
        if has_aux:
            return result, aux_cell[0]
        return result

    return _wrapper


def jacrev(func: Callable[..., Union[TN, tuple[TN, ...]]],
           argnums: Union[int, tuple[int, ...]] = 0,
           *,
           has_aux: bool = False,
           chunk_size: Optional[int] = None) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """通过反向模式 (reverse-mode autodiff) 计算 Jacobian 矩阵。

    接口严格对齐 torch.func.jacrev：has_aux / chunk_size 为 keyword-only，
    无 create_graph / strict 参数；结果始终保留计算图，可继续求导。

    实现方式为 vmap(vjp)：对输出的标准基逐一做 VJP，堆叠成完整 Jacobian。
    前向只执行一次（见 _jacobian_via_vjp）：VJP 关于 cotangent 线性，
    各列复用同一张已建立的计算图，只做独立的反向遍历。
    对于输出形状 (y1, y2, ...) 和输入形状 (x1, x2, ...)，
    Jacobian 形状为 (y1, y2, ..., x1, x2, ...)。

    参数:
        func: 待求 Jacobian 的函数，输入为 TN，输出为单个 TN 或 TN 序列。
        argnums: 需要计算 Jacobian 的输入位置，int 或 tuple[int, ...]，默认 0；
                 多位置时返回元组，顺序与 argnums 一致。非法取值（()、None、
                 含非 int 元素的元组、list 等）在调用导函数时抛 RuntimeError，
                 错误信息与 torch.func.jacrev 一致
        has_aux: 为 True 时 func 必须返回 (output, aux)，只对 output 求 Jacobian，
            返回 (jacobian, aux)。
        chunk_size: 每次处理的输出基向量个数，用于限制峰值内存；None 表示一次
            处理全部。结果与不分块一致。

    返回:
        Callable：func 的导函数，接受与 func 相同的参数并返回 Jacobian
        （has_aux=True 时返回 (Jacobian, aux)）。
        结果结构与输出结构对应：单输出单输入时返回单个矩阵，形状为
        output_shape + input_shape；单输出多输入时返回 tuple，顺序与 argnums
        一致；多输出单输入时返回 tuple，顺序与输出顺序一致；多输出多输入时
        返回 tuple-of-tuples（输出在外层、输入在内层），内层顺序与 argnums 一致。
    """
    return _make_jacobian(_jacobian_via_vjp, func, argnums,
                          allow_none=False, has_aux=has_aux,
                          strict=False,
                          chunk_size=chunk_size)


def jacfwd(func: Callable[..., Union[TN, tuple[TN, ...]]],
           argnums: Union[int, tuple[int, ...]] = 0,
           has_aux: bool = False,
           *,
           randomness: str = 'error') -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """通过正向模式 (forward-mode autodiff) 计算 Jacobian 矩阵。

    接口严格对齐 torch.func.jacfwd：has_aux 为位置参数，randomness 为
    keyword-only，取值 'error' / 'different' / 'same' 会被校验（非法值在调用
    导函数时抛 RuntimeError，与 torch.func 一致），但当前不做随机性重参数化；
    无 create_graph / chunk_size 参数；结果始终保留计算图。

    基于真正的前向 AD 原语 ``func.jvp``：primal 图只前向求值并记录一次，
    随后对每个输入元素构造 one-hot tangent，复用该图沿各算子的 ``_jvp_rule``
    传播，得到 Jacobian 的一列，最后堆叠成完整 Jacobian。实现等价于
    vmap(jvp)（linearize + 逐列传播），当前用逐列循环而非 vmap。

    参数:
        func: 待求 Jacobian 的函数，输入为 TN，输出为单个 TN 或 TN 序列。
        argnums: 需要计算 Jacobian 的输入位置，int、tuple[int, ...] 或 None，
                 默认 0。与 torch.func.jacfwd 一致：None 表示对全部参数求导
                 （此时即使只有 1 个参数也返回元组）；() 和含非 int 元素的元组、
                 list 等非法取值在调用导函数时抛 RuntimeError，错误信息与
                 torch.func 一致
        has_aux: 为 True 时 func 必须返回 (output, aux)，只对 output 求 Jacobian，
            返回 (jacobian, aux)。
        randomness: 仅占位，接受 'error' / 'different' / 'same'（非法值在调用
            导函数时抛 RuntimeError，错误信息与 torch.func 一致）；当前不做随机性
            重参数化。

    返回:
        Callable：func 的导函数，接受与 func 相同的参数并返回 Jacobian
        （has_aux=True 时返回 (Jacobian, aux)）。
        结果结构与输出结构对应：单输出单输入时返回单个矩阵，形状为
        output_shape + input_shape（argnums=None 时即使只有 1 个参数也返回
        元组）；单输出多输入时返回 tuple，顺序与 argnums 一致；多输出单输入时
        返回 tuple，顺序与输出顺序一致；多输出多输入时返回 tuple-of-tuples
        （输出在外层、输入在内层），内层顺序与 argnums 一致。
    """
    jac = _make_jacobian(_jacobian_via_jvp, func, argnums,
                         allow_none=True, has_aux=has_aux)

    def _wrapper(*args, **kwargs):
        # 与 torch.func.jacfwd 一致：randomness 在调用导函数时校验
        _check_randomness(randomness)
        return jac(*args, **kwargs)

    return _wrapper


# ---------------------------------------------------------------------------
# hessian
# ---------------------------------------------------------------------------

def hessian(func: Callable[..., TN],
            argnums: Union[int, tuple[int, ...]] = 0) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """计算标量值函数的 Hessian 矩阵，返回包裹函数。

    接口严格对齐 torch.func.hessian：无 create_graph / strict / has_aux 参数；
    结果始终保留计算图，可继续求导；未连通的 Hessian 块按零块返回。

    实现方式为 jacrev(grad(func))：先由反向模式 grad 得到梯度张量，
    再由反向模式 jacrev 对梯度求 Jacobian，即 Hessian。

    参数:
        func: 标量值函数，输出必须是 0 维 TN（ndim == 0），否则抛 RuntimeError。
        argnums: 需要计算 Hessian 的输入位置，int 或 tuple[int, ...]，默认 0；
                 非法取值（()、None、含非 int 元素的元组、list 等）在调用导函数时
                 抛 RuntimeError，错误信息与 torch.func.hessian 一致

    返回:
        Callable：func 的导函数，接受与 func 相同的参数并返回 Hessian。
        单个 argnums 位置时返回形状 input_shape + input_shape 的矩阵；多个位置时
        返回 tuple，第 j 项是第 argnums[j] 个输入的梯度对该输入的 Hessian 块。
    """
    def _wrapper(*args, **kwargs):
        an = _check_argnums(argnums)

        def f(*gargs):
            return func(*gargs, **kwargs)

        output = f(*args)
        if not isinstance(output, TN) or output.ndim != 0:
            raise RuntimeError("Hessian only supports scalar-valued functions")

        # hessian = jacrev(grad(func))：反向叠反向。
        # 内层用 create_graph=True 求出标量输出梯度的张量（对 argnums 所选输入），
        # 外层 jacrev 再对梯度求 Jacobian，即得 Hessian。
        # 相比 jacrev(jacfwd(func))，梯度图只构建一次：本库的前向模式由局部 VJP
        # 转置模拟，jacfwd 会为每个基向量生成一条独立的列子图，
        # 外层每次反向都要遍历这些子图，导致 O(n^2) 放大；反向叠反向则避免该开销。
        # argnums 未覆盖全部参数、或梯度与某输入不连通时，对应 Hessian 块为零
        # （如 ∂²(xy²)/∂x²），按 torch.func 语义返回零块而非报错。

        def _grad_vector(*gargs):
            out = f(*gargs)
            if not isinstance(out, TN) or out.ndim != 0:
                raise RuntimeError("Hessian only supports scalar-valued functions")
            grads = _grad(out, [gargs[i] for i in an],
                          create_graph=True, allow_unused=True)
            grads = [zeros_like(gargs[i], requires_grad=True) if g is None else g
                     for i, g in zip(an, grads)]
            if len(grads) == 1:
                return grads[0]
            return tuple(grads)

        # 只对 argnums 所选输入求 Jacobian（未选中的参数不产生多余反向遍历）。
        # blocks[j][j] 即第 j 个梯度对第 an[j] 个输入的 Hessian 块，因此非连续
        # argnums（如 (0, 2)）也能直接定位。
        blocks, _ = _jacobian_via_vjp(_grad_vector, args, an, strict=False)

        if len(an) == 1:
            return blocks[0][0]
        return tuple(blocks[j][j] for j in range(len(an)))

    return _wrapper


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------
def _batch_slice(tensor: TN, index: tuple) -> TN:
    """vmap 专用快速切片，等价于 tensor[index] 但跳过通用索引分派。

    仅用于基础索引（slice / 整数）且索引已由调用方归一化为非负的情形：
    直接取底层数组的视图，并在需要梯度（反向或前向追踪）时保留到原张量的
    依赖边（fromvars / parms / _grad_rule / _jvp_rule 与 __getitem__ 一致）。
    建边逻辑须与 _getitem 保持同步，否则 vmap 在前向/反向路径上的行为会分叉。
    """
    data = tensor._array[index]
    # 标量索引可能返回 numpy/cupy 标量而非 0 维数组，
    # 统一转成数组，保持与 __getitem__ 的一致性（否则 ndim/_get_array_lib 会失效）
    arrlib = tensor._get_array_lib()
    if not isinstance(data, arrlib.ndarray):
        data = arrlib.asarray(data)

    ret = TN(data, requires_grad=is_grad_enabled() and tensor.requires_grad)
    ret._is_view = True
    ret._base = tensor
    ret._view_index = index
    # 反向需要 requires_grad；前向追踪下输入可能是 requires_grad=False 的
    # detach 张量，此时也必须建边，否则 tangent 会被静默丢弃
    if ret.requires_grad or forward_ad_trace_active():
        ret.fromvars = (tensor,)
        ret.parms = (index,)
        ret._grad_rule = _getitem_grad
        # 与 _grad_rule 一起登记直接规则，使 jvp 能对 tangent 施加同一索引
        ret._jvp_rule = _getitem_jvp
    return ret

def vmap(func: Callable[..., Union[TN, tuple[TN, ...]]],
         in_dims: Union[int, tuple[int, ...], list[Optional[int]]] = 0,
         out_dims: Union[int, tuple[int, ...], list[int]] = 0,
         randomness: str = 'error',
         *,
         chunk_size: Optional[int] = None) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """向量化映射 (vectorized map)。

    将原本作用于单个样本的函数扩展到批次维度上并行（通过 loop + 堆叠实现，
    不是真正的向量化，但接口与 torch.func.vmap 对齐）。

    参数:
        in_dims: 每个输入的批次维度。int 表示所有输入共用该维度；
                 tuple[int,...] 表示每个输入各自的维度位置（长度必须与输入个数
                 一致）；元素 None 表示该输入不批处理。与 torch.func 一致，
                 顶层 None 不合法。
        out_dims: 每个输出的批次维度，语义同 in_dims；长度必须与输出个数一致，
                 元素不能为 None（torch.func 同样不允许）。
        randomness: 仅占位，接受 'error' / 'different' / 'same'（非法值在创建
             包裹函数时抛 RuntimeError，错误信息与 torch.func 一致）；当前不做
             随机性重参数化。
        chunk_size: 关键字参数。每次处理的批次样本数；None 表示一次处理整个批次。
             分块只改变循环与堆叠的粒度，结果与不分块一致。

    返回:
        包裹函数，接受批次输入，返回批次输出。

    注意:
        - 切片使用内部 _batch_slice：等价于 inputs[batch_idx]，但跳过通用索引分派
        - 批次维通常是 dim=0；若指定 dim>0，切片索引为 (slice(None),)*d + (b,) + ...
        - 非批处理输入（in_dims 对应元素为 None）直接透传给底层 func
        - 输出需要梯度时用 stack 保留拼接的梯度边；纯前向时预分配缓冲写入
    """
    _check_randomness(randomness)
    if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size <= 0):
        raise ValueError(
            f"vmap: chunk_size should be None or greater than 0. (got {chunk_size})")

    def _normalize_in_dims(dims, args):
        n_inputs = len(args)
        if isinstance(dims, int):
            return [dims] * n_inputs
        if isinstance(dims, (tuple, list)):
            if len(dims) != n_inputs:
                raise ValueError(
                    f"vmap({func}, in_dims={dims}, ...)(<inputs>): in_dims is not "
                    "compatible with the structure of `inputs`.")
            for d in dims:
                if d is not None and not isinstance(d, int):
                    raise ValueError(
                        f"vmap({func}, in_dims={dims}, ...)(<inputs>): expected "
                        "`in_dims` to be int or a (potentially nested) tuple "
                        f"matching the structure of inputs, got element: {d}")
            return list(dims)
        raise ValueError(
            f"vmap({func}, in_dims={dims}, ...)(<inputs>): expected `in_dims` to be "
            "int or a (potentially nested) tuple matching the structure of inputs, "
            f"got: {type(dims)}")

    def _normalize_out_dims(dims, n_outputs):
        if isinstance(dims, int):
            return [dims] * n_outputs
        if dims is None:
            raise ValueError(
                f"vmap({func}, ...): `{func}` can not return a BatchedTensor when "
                "out_dim is None")
        if isinstance(dims, (tuple, list)):
            if len(dims) != n_outputs:
                raise ValueError(
                    f"vmap({func}, ..., out_dims={dims})(<inputs>): out_dims is not "
                    "compatible with the structure of `outputs`.")
            out = []
            for d in dims:
                if d is None:
                    raise ValueError(
                        f"vmap({func}, ...): `{func}` can not return a BatchedTensor "
                        "when out_dim is None")
                if not isinstance(d, int):
                    raise ValueError(
                        f"vmap({func}, ..., out_dims={dims})(<inputs>): expected "
                        "out_dims to be int or a tuple of ints.")
                out.append(d)
            return out
        raise ValueError(
            f"vmap({func}, ..., out_dims={dims})(<inputs>): expected out_dims to be "
            "int or a (potentially nested) tuple matching the structure of outputs, "
            f"got: {type(dims)}")

    def _wrapper(*args, **kwargs):
        in_dims_list = _normalize_in_dims(in_dims, args)

        # 预计算每个输入的切片计划（ndim, 批维位置 d）；None 表示该输入透传。
        # 循环内不再做 isinstance / 负维归一化 / 逐维生成 idx 元组。
        plans = []
        sizes = []
        for dim, a in zip(in_dims_list, args):
            if dim is None:
                plans.append(None)
                continue
            if not isinstance(a, TN):
                raise ValueError(
                    f"vmap({func}, in_dims={in_dims}, ...)(<inputs>): Got "
                    f"in_dim={dim} for an input but the input is of type "
                    f"{type(a)}. We cannot vmap over non-Tensor arguments, "
                    "please use None as the respective in_dims.")
            ndim = a.ndim
            if dim < -ndim or dim >= ndim:
                raise ValueError(
                    f"vmap({func}, in_dims={in_dims}, ...)(<inputs>): Got "
                    f"in_dim={dim} for some input, but that input is a Tensor of "
                    f"dimensionality {ndim} so expected in_dim to satisfy "
                    f"-{ndim} <= in_dim <= {ndim - 1}.")
            d = dim if dim >= 0 else ndim + dim
            plans.append((ndim, d))
            sizes.append(a.shape[d])

        if sizes and any(s != sizes[0] for s in sizes):
            raise ValueError(
                "vmap: Expected all tensors to have the same size in the mapped "
                f"dimension, got sizes {sizes} for the mapped dimension")
        batch_size = sizes[0] if sizes else None

        if batch_size is None:
            # 没有批处理维度 — 直接调用 func
            return func(*args, **kwargs)

        full = (slice(None),)
        # 预计算每个输入的索引前后缀，循环内只做一次元组拼接；
        # 非批处理输入无需每次重建容器。
        prefixes = []
        for plan in plans:
            if plan is None:
                prefixes.append(None)
            else:
                ndim, d = plan
                prefixes.append((full * d, full * (ndim - d - 1)))

        outputs_list = []
        sliced_args = list(args)   # 复用容器，循环内只替换需要切片的元素
        for b in range(batch_size):
            for k, plan in enumerate(plans):
                if plan is None:
                    sliced_args[k] = args[k]
                else:
                    pre, post = prefixes[k]
                    # _batch_slice 等价于 a[idx]，但跳过通用索引分派
                    sliced_args[k] = _batch_slice(args[k], pre + (b,) + post)
            outputs_list.append(func(*sliced_args, **kwargs))

        # 按 chunk_size 切分批次结果；None 表示整批一个 chunk
        step = batch_size if chunk_size is None else chunk_size
        chunks = [outputs_list[s:s + step] for s in range(0, batch_size, step)]

        # 堆叠结果
        first_out = chunks[0][0]
        is_single = isinstance(first_out, TN)
        out_dims_list = _normalize_out_dims(out_dims, 1 if is_single else len(first_out))

        def _stack_result(items, out_dim):
            axis = out_dim if out_dim is not None else 0
            # 存在梯度需求时必须用 stack：它登记了拼接的梯度边，
            # 否则 grad/vjp 无法回传到批输入。
            if any(it.requires_grad for it in items):
                return stack(items, dim=axis)
            # 纯前向场景：预分配输出缓冲并逐样本写入，省去 stack 的
            # 逐项校验与拼接开销。形状/dtype 不一致时回退到 stack。
            first = items[0]
            ref_shape, ref_dtype = first.shape, first.dtype
            if any(it.shape != ref_shape or it.dtype != ref_dtype for it in items):
                return stack(items, dim=axis)
            ndim = len(ref_shape) + 1
            ax = axis if axis >= 0 else ndim + axis
            shape = list(ref_shape)
            shape.insert(ax, len(items))
            buf = zeros(tuple(shape), dtype=ref_dtype, device=first.device)
            prefix = (slice(None),) * ax
            for b, it in enumerate(items):
                buf._array[prefix + (b,)] = it._array
            return buf

        def _stack_chunks(per_chunk, out_dim, sample_ndim):
            """先逐 chunk 堆叠，再把多个 chunk 沿同一批次维拼接。"""
            stacked = [_stack_result(items, out_dim) for items in per_chunk]
            if len(stacked) == 1:
                return stacked[0]
            ndim = sample_ndim + 1
            ax = (out_dim if out_dim is not None else 0)
            if ax < 0:
                ax += ndim
            return cat(stacked, dim=ax)

        if is_single:
            return _stack_chunks(chunks, out_dims_list[0], len(first_out.shape))
        else:
            n_out = len(first_out)
            result = []
            for oi in range(n_out):
                per_chunk = [[out[oi] for out in items] for items in chunks]
                result.append(_stack_chunks(
                    per_chunk,
                    out_dims_list[oi] if oi < len(out_dims_list) else 0,
                    len(first_out[oi].shape)))
            return tuple(result)

    return _wrapper


# ---------------------------------------------------------------------------
# functional_call / stack_module_state
# ---------------------------------------------------------------------------

def _named_state(module: Any):
    """收集 module 的具名参数与具名缓冲区。

    参数:
        module: riemann.nn.Module 或提供 named_parameters / _named_parameters
            的任意对象。

    返回:
        (params, buffers)：两个 (name, tensor) 列表，保持模块注册顺序。
    """
    if hasattr(module, "named_parameters"):
        params = list(module.named_parameters())
    elif hasattr(module, "_named_parameters"):
        params = list(module._named_parameters())
    else:
        params = []
    if hasattr(module, "named_buffers"):
        buffers = list(module.named_buffers())
    else:
        buffers = []
    return params, buffers


def _resolve_state_slot(module: Any, name: str):
    """把点分名称定位到可读写的落点。

    对 riemann.nn.Module，点号路径沿 ``_modules`` 向下遍历，叶子名在
    ``_parameters`` / ``_buffers`` 字典中查找（直接改字典可绕开 __setattr__
    把普通 TN 误注册为 buffer）；其余对象回退到 getattr / setattr。

    参数:
        module: 目标模块。
        name: 参数或缓冲区的具名路径，如 "l1.weight"。

    返回:
        ("dict", store, key) 或 ("attr", owner, key)；定位失败返回 None。
    """
    parts = name.split('.')
    owner = module
    for part in parts[:-1]:
        modules = getattr(owner, "_modules", None)
        if isinstance(modules, dict) and part in modules:
            owner = modules[part]
        else:
            owner = getattr(owner, part, None)
            if owner is None:
                return None
    leaf = parts[-1]
    for store_name in ("_parameters", "_buffers"):
        store = getattr(owner, store_name, None)
        if isinstance(store, dict) and leaf in store:
            return ("dict", store, leaf)
    # Module 类对象存在注册表：未注册的名字不能被当成普通属性改写
    if (isinstance(getattr(owner, "_parameters", None), dict)
            or isinstance(getattr(owner, "_buffers", None), dict)):
        return None
    if hasattr(owner, leaf):
        return ("attr", owner, leaf)
    return None


def _read_slot(slot) -> Any:
    """读取 _resolve_state_slot 返回的落点当前值。"""
    kind, owner, key = slot
    return owner[key] if kind == "dict" else getattr(owner, key)


def _write_slot(slot, value) -> None:
    """把落点当前值替换为 value。"""
    kind, owner, key = slot
    if kind == "dict":
        owner[key] = value
    else:
        setattr(owner, key, value)


def _normalize_state_dicts(values, params):
    """把 parameter_and_buffer_dicts 归一化为 {name: tensor}。

    支持三种形式（与 torch.func 对齐并保留旧式位置序列）：
        - dict：直接使用；
        - dict 序列：合并，键重复时报 ValueError；
        - TN / TN 序列：按 named_parameters() 的顺序映射到参数名（旧式用法）。

    参数:
        values: 用户传入的参数/缓冲区集合。
        params: (name, tensor) 列表，用于旧式位置序列的命名。

    返回:
        {name: tensor} 字典。

    异常:
        TypeError: values 既不是 dict 也不是 TN/序列。
        ValueError: 旧式位置序列长度与模块参数个数不符；dict 序列键重复；或序列中
            混有 dict 与非 dict 元素。
    """
    if isinstance(values, dict):
        return dict(values)
    if isinstance(values, TN):
        values = [values]
    if isinstance(values, (list, tuple)):
        items = list(values)
        if items and builtins.all(isinstance(d, dict) for d in items):
            merged = {}
            counts = {}
            for d in items:
                for k in d:
                    counts[k] = counts.get(k, 0) + 1
            repeated = [k for k, n in counts.items() if n > 1]
            if repeated:
                raise ValueError(
                    f"{repeated} appeared in multiple dictionaries; "
                    "behavior of functional call is ambiguous")
            for d in items:
                merged.update(d)
            return merged
        if builtins.any(isinstance(d, dict) for d in items):
            raise ValueError(
                "Expected all elements of parameter_and_buffer_dicts to be "
                "dictionaries")
        names = [n for n, _ in params]
        if len(items) != len(names):
            raise ValueError(
                f"module has {len(names)} parameters but got "
                f"{len(items)} values in parameter_values")
        return dict(zip(names, items))
    raise TypeError(
        "parameter_and_buffer_dicts must be a dict, a sequence of dicts, "
        f"or a TN / sequence of TN, got {type(values)}")


def functional_call(module: Any,
                    parameter_and_buffer_dicts: Any,
                    args: Any = None,
                    kwargs: Optional[dict] = None,
                    *,
                    tie_weights: bool = True,
                    strict: bool = False) -> Any:
    """以自定义参数与缓冲区调用 module.forward，不修改 module 自身的状态。

    内部按名字临时替换 module 的参数/缓冲区，调用 forward 后无条件恢复，
    用于无状态（stateless）地调用 Module，便于与 grad / jacrev / vmap 等
    函数式原语组合。接口与 torch.func.functional_call 对齐。

    参数:
        module: 带有 forward 方法的对象（通常是 riemann.nn.Module 或子类）
        parameter_and_buffer_dicts: {名称: 张量}。名称遵循 state_dict 约定，
             如 "weight"、"l1.bias"；参数与缓冲区可放在同一个字典里，也可以
             传入多个互不重叠的字典。传 None 表示使用 module 自身状态。
             为兼容旧用法，也接受按 named_parameters() 顺序排列的 TN / TN 序列。
        args: forward 的位置参数；None 表示无参数，单个 TN 不会被解包
        kwargs: forward 的关键字参数
        tie_weights: 为 True 时尊重模块中原有的权重绑定：只提供绑定组中的一个
             名字也会同步更新同组的其他名字；若同一组给出不同对象则报错。
             为 False 时只替换显式给出的名字。
        strict: 为 True 时，提供的名称必须与模块的参数/缓冲区名称完全一致，
             缺失或多出都报错；为 False 时允许多出（忽略）或缺失（保留原值）。

    返回:
        module.forward(*args, **kwargs) 的返回值
    """
    if kwargs is None:
        kwargs = {}

    # 归一化 args：None -> ()；单个 TN 不被解包；list 视为位置参数序列
    if args is None:
        args = ()
    elif isinstance(args, TN):
        args = (args,)
    elif isinstance(args, list):
        args = tuple(args)

    forward = getattr(module, "forward", None) or module

    params, buffers = _named_state(module)
    if parameter_and_buffer_dicts is None:
        return forward(*args, **kwargs)
    if not params and not buffers:
        return forward(*args, **kwargs)

    state = _normalize_state_dicts(parameter_and_buffer_dicts, params)

    all_names = [n for n, _ in params] + [n for n, _ in buffers]
    if strict:
        missing = sorted(set(all_names) - set(state))
        unexpected = sorted(set(state) - set(all_names))
        if missing or unexpected:
            msgs = []
            if unexpected:
                msgs.append(f"Unexpected key(s): {', '.join(map(repr, unexpected))}.")
            if missing:
                msgs.append(f"Missing key(s): {', '.join(map(repr, missing))}.")
            raise RuntimeError(
                "Error(s) in reparametrizing for {}:\n\t{}".format(
                    type(module).__name__, "\n\t".join(msgs)))

    # 权重绑定：同一底层张量对应多个名字时补齐未提供的名字，并检查冲突值
    if tie_weights:
        groups = {}
        for name, tensor in list(params) + list(buffers):
            groups.setdefault(id(tensor), []).append(name)
        for names in groups.values():
            if len(names) < 2:
                continue
            given = [(n, state[n]) for n in names if n in state]
            if not given:
                continue
            first_value = given[0][1]
            for _, value in given[1:]:
                if value is not first_value:
                    raise ValueError(
                        f"functional_call got multiple values for keys {sorted(names)}, "
                        "which are tied. Consider using tie_weights=False")
            for n in names:
                state[n] = first_value

    # 临时替换并记录原值；finally 中恢复，保证前向抛异常也不污染模块
    slots = []
    orig_values = []
    try:
        for name, value in state.items():
            slot = _resolve_state_slot(module, name)
            if slot is None:
                # strict=False 时忽略模块中不存在的名字
                continue
            slots.append(slot)
            orig_values.append(_read_slot(slot))
            _write_slot(slot, value)
        return forward(*args, **kwargs)
    finally:
        for slot, value in zip(slots, orig_values):
            _write_slot(slot, value)


def stack_module_state(models: Any):
    """把多个同构 Module 的参数与缓冲区沿新维度 0 堆叠，用于 vmap 集成。

    与 torch.func.stack_module_state 对齐：返回 (params, buffers) 两个
    {名称: 堆叠张量} 字典，可直接配合 functional_call 在 vmap 下做模型集成。

    参数:
        models: Module 序列或 ModuleList，至少一个；所有模型必须同类、同
            training/eval 模式，且参数/缓冲区名称一致。

    返回:
        (params, buffers)。

    异常:
        RuntimeError: 模型为空、类型/模式/名称不一致，或同名张量的
            requires_grad 不一致。
    """
    models = list(models)
    if len(models) == 0:
        raise RuntimeError("stack_module_state: Expected at least one model, got 0.")
    if not (all(getattr(m, "training", True) for m in models)
            or all(not getattr(m, "training", True) for m in models)):
        raise RuntimeError(
            "stack_module_state: Expected all models to have the same training/eval mode.")
    first_type = type(models[0])
    if not builtins.all(type(m) is first_type for m in models):
        raise RuntimeError(
            "stack_module_state: Expected all models to be of the same class.")

    def _stack_named(dicts, kind):
        names = list(dicts[0])
        expected = set(names)
        for d in dicts[1:]:
            if set(d) != expected:
                raise RuntimeError(
                    f"stack_module_state: Expected all models to have the same "
                    f"{kind} names")
        stacked = {}
        for name in names:
            tensors = [d[name] for d in dicts]
            all_requires_grad = builtins.all(t.requires_grad for t in tensors)
            none_requires_grad = builtins.all(not t.requires_grad for t in tensors)
            if not all_requires_grad and not none_requires_grad:
                raise RuntimeError(
                    f"Expected {name} from each model to have the same .requires_grad")
            value = stack(tensors, dim=0)
            if all_requires_grad:
                value = value.detach().requires_grad_(True)
            stacked[name] = value
        return stacked

    all_params = [dict(m.named_parameters()) for m in models]
    all_buffers = [dict(m.named_buffers()) if hasattr(m, "named_buffers") else {}
                   for m in models]
    return _stack_named(all_params, "parameter"), _stack_named(all_buffers, "buffer")
