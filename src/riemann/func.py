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
    - jacrev / jacfwd: Jacobian 矩阵的反向/正向计算方式
    - hessian: Hessian 矩阵
    - vmap: 向量化映射（沿指定批次维度循环调用并堆叠）
    - functional_call: 以自定义参数调用 Module 实例

所有函数间相互复用：jvp / vjp 是前向 / 反向模式原语，共享参数归一化与前向求值
辅助（_traceable_args / _prepare_forward / _call_and_split）。
jacrev / jacfwd 共用 _make_jacobian 做 argnums 校验与结果整形，内核分别是 vmap(vjp)
与逐列 jvp（共享一次前向求值与局部 VJP 图缓存，与 vmap(jvp) 语义相同）。
hessian 复用 _jacobian_via_vjp 内核，对 grad 的结果再求一次 Jacobian（反向叠反向）。

补充说明：本库算子只登记局部 VJP（反向模式），因此 jvp 并非基于对偶数，而是通过
"对局部 VJP 再做一次转置"沿计算图正向传播 tangent；细节见 _local_jvp。
"""

import builtins
from typing import Callable, Any, Optional, Union
from .tensordef import *
from .tensordef import _batch_slice
from .autograd.grad import grad as _grad


# ---------------------------------------------------------------------------
# argnums 校验（与 torch.func 的错误信息保持一致）
# ---------------------------------------------------------------------------

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

def grad(func: Callable[..., TN],
         argnums: Union[int, tuple[int, ...]] = 0,
         has_aux: bool = False,
         create_graph: bool = False) -> Callable[..., Union[TN, tuple[TN, Any]]]:
    """返回 func 的导函数（一个包裹函数）。

    注意区分三层输出：
        1. func 的输出：默认必须是标量 TN（ndim == 0）；has_aux=True 时必须是
           (value, aux) 二元组，value 是标量 TN，aux 是任意辅助结果。
        2. func.grad(...) 的输出：一个函数对象，即 func 的导函数（包裹函数），
           它接受与 func 相同的参数。
        3. 导函数的输出：has_aux=False 时返回 grad_result；has_aux=True 时返回
           (grad_result, aux)，aux 原样透传。grad_result 是 value 关于 argnums
           所选参数的梯度：只含 1 个参数位置时为单个梯度张量（形状与对应输入
           相同），含多个位置时为由多个梯度张量构成的元组（顺序与 argnums 一致）。

    与 autograd.grad 的区别：
        autograd.grad 接受输出和输入；func.grad 接受函数和 argnums，
        返回的导函数接受 func 的参数并自动前向+反向。

    参数:
        func: 可微函数，输出必须是标量 TN（ndim == 0），否则抛出 RuntimeError
        argnums: 需要求导的参数位置，int 或 tuple[int, ...]，默认 0。非法取值
                 （()、None、含非 int 元素的元组、list 等）在调用导函数时抛
                 RuntimeError，错误信息与 torch.func 一致
        has_aux: 如果 True，func 必须返回 (value, aux) 二元组：value 是 func 的
                 前向计算输出，grad 要求它必须是标量张量；aux 是 func 顺带输出
                 的辅助结果（任意对象，不参与求导）。导函数只从 value 计算梯度，
                 并把 aux 原样透传在返回值中
        create_graph: 如果 True，梯度张量会被追踪，可用于高阶微分。默认 False

    返回:
        Callable: func 的导函数，其输出见上文第 3 点
    """
    def _wrapper(*args, **kwargs):
        an = _check_argnums(argnums)
        outputs = func(*args, **kwargs)
        aux = None
        if has_aux:
            outputs, aux = outputs

        if outputs.ndim != 0:
            raise RuntimeError(
                f"grad() requires func to return a scalar tensor, got shape {outputs.shape}"
            )

        grad_args = [args[i] for i in an]
        grad_result = _grad(outputs, grad_args, create_graph=create_graph)

        if len(an) == 1:
            grad_result = grad_result[0]

        if has_aux:
            return grad_result, aux
        return grad_result

    return _wrapper


def grad_and_value(func: Callable[..., TN],
                   argnums: Union[int, tuple[int, ...]] = 0,
                   has_aux: bool = False,
                   create_graph: bool = False) -> Callable[..., Union[tuple[TN, TN], tuple[TN, TN, Any]]]:
    """返回 func 的导函数，同时返回 func 的前向输出（value）。

    注意区分三层输出：
        1. func 的输出：同 func.grad（默认标量 TN；has_aux=True 时为 (value, aux)）
        2. func.grad_and_value(...) 的输出：一个函数对象，接受与 func 相同的参数
        3. 该函数的输出：has_aux=False 时返回 (grad_result, value)；has_aux=True
           时返回 (grad_result, value, aux)。grad_result 是 value 关于 argnums 所选
           参数的梯度：只含 1 个位置时为单个梯度张量，含多个位置时为梯度元组

    内部只做一次前向+一次反向（与分开调用 func 和 grad(func) 等价）。

    参数:
        argnums: 同 func.grad（非法取值在调用导函数时抛 RuntimeError，
                 错误信息与 torch.func 一致）

    返回:
        Callable: 导函数，其输出见上文第 3 点
    """
    def _wrapper(*args, **kwargs):
        an = _check_argnums(argnums)
        outputs = func(*args, **kwargs)
        aux = None
        if has_aux:
            outputs, aux = outputs

        if outputs.ndim != 0:
            raise RuntimeError(
                f"grad_and_value() requires func to return a scalar tensor, got shape {outputs.shape}"
            )

        grad_args = [args[i] for i in an]
        grad_result = _grad(outputs, grad_args, create_graph=create_graph)

        if len(an) == 1:
            grad_result = grad_result[0]

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


def _vjp_with_output(out: TN, inputs, cotangent: TN,
                     create_graph: bool, strict: bool):
    """对已求值的前向输出做一次 VJP，复用其计算图。

    VJP 关于 cotangent 是线性的：对固定的前向输出，可反复以不同 cotangent
    调用，无需重跑 func。jacrev 的每个输出基向量因此共享同一次前向计算。

    参数:
        out: 已求值的前向输出节点。
        inputs: 目标输入张量序列。
        cotangent: 输出侧余切，形状与 out 一致。
        create_graph: 是否为高阶导数保留计算图。
        strict: 为 True 时，若某个输入与输出完全不连通（梯度为 None）则报错。

    返回:
        与 inputs 等长的 list；不连通输入以同形零张量补齐。
    """
    if not out.requires_grad:
        # 常量输出与所有输入都不连通：strict 时报错，否则返回同形零块，
        # 避免落入 _grad 的 "Only a tensor require grad" 错误。
        if strict:
            raise RuntimeError(
                "Output is independent of input (0), "
                "Set allow_unused=True if this is the desired behavior.")
        return [zeros_like(inp, requires_grad=create_graph) for inp in inputs]
    grads = _grad(out, inputs, grad_outputs=cotangent,
                  create_graph=create_graph, allow_unused=True)
    if strict:
        for j, gv in enumerate(grads):
            if gv is None:
                raise RuntimeError(
                    f"Output is independent of input ({j}), "
                    f"Set allow_unused=True if this is the desired behavior.")
    return [zeros_like(inp, requires_grad=create_graph) if gv is None else gv
            for inp, gv in zip(inputs, grads)]

def _make_shadow(node: TN, leaf_map: dict) -> TN:
    """构造计算图节点的"局部影子"。

    影子节点与真实节点共享底层数据，但其 fromvars 全是叶子副本，
    因此在影子上求局部 VJP 的转置时只会遍历当前算子，而不会回溯整张图。

    参数:
        node: 前向图中的真实节点。
        leaf_map: id(原始张量) -> 叶子副本 的映射，保证同一张量在多处被引用时
            共用同一个叶子副本。

    返回:
        影子节点：共享 node._array 与 node.parms，fromvars 均为无计算图的叶子。
    """
    shadow = TN(node._array, requires_grad=False)
    shadow.parms = node.parms
    fromvars = []
    for var in node.fromvars:
        var_shadow = leaf_map.get(id(var))
        if var_shadow is None:
            var_shadow = TN(var._array, requires_grad=False)
            leaf_map[id(var)] = var_shadow
        fromvars.append(var_shadow)
    shadow.fromvars = tuple(fromvars)
    return shadow


def _local_jvp_by_basis(node: TN, i: int, target: TN, tangent: TN) -> TN:
    """gradfunc 未建立计算图时的回退：用输出基向量显式构造局部 Jacobian。

    全程使用张量运算（不经过 numpy），因此 CPU/CUDA 都适用：对输出的每个
    基向量 k 求局部 VJP 得到 J_i 的第 k 列，堆叠为 (numel_in, numel_out) 的
    转置矩阵，再与 tangent 相乘得到 J_i @ tangent。

    参数:
        node: 当前算子的输出节点，提供形状、dtype、device 与 numel。
        i: 入边下标，对应 node.gradfuncs[i]。
        target: 调用 gradfunc 的目标节点（真实节点或影子节点），其 grad_value
            会被临时改写为各基向量。
        tangent: 与第 i 个输入同形的切向量。

    返回:
        与输出同形的张量：J_i @ tangent。
    """
    numel_out = node.numel()
    basis = eye(numel_out, dtype=node.dtype, device=node.device)
    cols = []
    saved = target.grad_value
    try:
        for k in range(numel_out):
            target.grad_value = basis[k].reshape(node.shape)
            cols.append(node.gradfuncs[i](target, i).reshape(-1))
    finally:
        target.grad_value = saved
    mat_t = stack(cols, dim=-1)
    # 局部 VJP 登记的是共轭转置形式（gradfunc 中对局部导数取 conj），
    # 故此处用共轭转置还原 J；实数情形与 T 等价。
    out = mat_t.mH @ tangent.reshape(-1)
    return out.reshape(node.shape)


def _local_jvp(node: TN, shadow: Optional[TN], i: int, tangent: TN,
               create_graph: bool, cache: Optional[dict] = None) -> TN:
    """计算单个节点的局部 JVP：J_i @ tangent。

    背景：每个算子只登记了反向规则 gradfunc，它把 node 的余切 g_out（与输出
    同形）映射为 g_in = J_iᵀ g_out（与输入同形），并没有前向规则。这里用
    "对局部 VJP 再做一次反向"补出前向结果：

    1. 以零值 g_out 调 gradfunc，只为得到 g_in = J_iᵀ g_out 这张小图；
    2. 小图对应线性映射 L(g) = J_iᵀ g，线性映射的 VJP 就是其转置，
       (J_iᵀ)ᵀ = J_i；
    3. 以 tangent 作为 g_in 的余切对 L 反向，得到 Lᵀ tangent = J_i @ tangent。

    注意角色互换：node 的余切始终是 g_out；tangent 只在内层反向里充当 g_in
    的余切，因此形状与输入一致，与输出同形的 g_out 无关。复数情形 gradfunc
    登记的是 J_iᴴ，两次共轭随两次转置一起抵消，结果同样是 J_i。

    参数:
        node: 前向图中的当前算子输出节点，node.gradfuncs[i] 是第 i 条入边的
            局部 VJP 规则。
        shadow: node 的局部影子节点（共享底层数据、fromvars 换成叶子副本），
            让内层反向只遍历当前算子；为 None 时退化为 node 本身。
        i: 入边下标，同时索引 node.fromvars[i] 与 node.gradfuncs[i]。
        tangent: 沿该入边传入的切向量，形状与第 i 个输入一致。
        create_graph: 为 True 时保留内层反向的计算图以支持高阶导数，此时必须
            使用真实节点（影子节点会与原始输入断链）。
        cache: 可选缓存，键为 (id(node), i)，值为 (g_in, g_out) 的局部 VJP 图；
            jacfwd 的多列因此可复用同一图结构，只重复反传、不重复构图。

    返回:
        与第 i 个输入同形的张量，即该条边上的局部 JVP：J_i @ tangent。
    """
    if cache is not None:
        entry = cache.get((id(node), i))
        if entry is not None:
            g_in, g_out = entry
            return _grad(g_in, [g_out], grad_outputs=tangent,
                         create_graph=create_graph, allow_unused=True)[0]

    # create_graph=True 时必须真实节点上建图：若用影子节点，g_in 与原始输入
    # 断开连接，高阶导数会丢失。其余情况用影子节点，使局部反传只遍历当前算子，
    # 不回溯整张计算图。
    target = node if create_graph else shadow
    if target is None:
        target = node
    fn = node.gradfuncs[i]
    # 局部 VJP 关于输出余切是线性的，这里用零值 seed 只为拿到 g_in 的图结构
    g_out = zeros_like(node, requires_grad=True)
    saved = target.grad_value
    target.grad_value = g_out
    try:
        g_in = fn(target, i)
    finally:
        target.grad_value = saved

    if not isinstance(g_in, TN):
        raise RuntimeError("gradfunc must return a tensor")
    if g_in.requires_grad:
        # 常规路径：gradfunc 建了图，再反传一次即得 J_i @ tangent
        if cache is not None:
            cache[(id(node), i)] = (g_in, g_out)
        return _grad(g_in, [g_out], grad_outputs=tangent,
                     create_graph=create_graph, allow_unused=True)[0]
    # 回退路径：gradfunc 未建图（如梯度开关关闭），只能显式构造局部 Jacobian
    return _local_jvp_by_basis(node, i, target, tangent)


def _graph_topological_order(outputs):
    """按拓扑序（叶子在前、输出在后）返回从 outputs 可达的计算图节点。

    用迭代式 DFS 的后序输出实现，避免递归深度限制。每个节点入栈两次：
    先以 expanded=False 入栈（用于展开子节点），再以 expanded=True 入栈
    （子节点全部处理完后才追加到 order），从而保证输出一定排在其输入之后。

    参数:
        outputs: 前向输出节点序列。

    返回:
        list[TN]：从 outputs 可达的全部节点，按拓扑序排列（叶子在前、输出在后）。
    """
    order = []
    visited = set()
    for out in outputs:
        stack = [(out, False)]
        while stack:
            node, expanded = stack.pop()
            node_id = id(node)
            if expanded:
                if node_id not in visited:
                    visited.add(node_id)
                    order.append(node)
                continue
            if node_id in visited:
                continue
            stack.append((node, True))
            # reversed 让 fromvars 中靠前的输入先被展开，保持与正向求值一致的顺序
            for var in reversed(node.fromvars):
                if id(var) not in visited:
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


def _jvp_with_prepared(prepared, tangents, create_graph: bool, strict: bool,
                       cache: Optional[dict] = None):
    """在已准备好的前向计算图上传播一组 tangent。

    tangents 与 prepared['primals'] 一一对应，元素为 None 表示该输入的
    tangent 为零（跳过，不做任何传播）。cache 用于跨列复用局部 VJP 图。

    参数:
        prepared: _prepare_forward 的返回值（前向输出、拓扑序、影子节点等）。
        tangents: 切向量序列，与 prepared['primals'] 一一对应；None 表示零方向。
        create_graph: 是否为高阶导数保留内层反向的计算图。
        strict: 为 True 时，若某输出与某输入完全不连通则报错。
        cache: 可选的局部 VJP 图缓存，jacfwd 多列共享。

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
                var_id = id(var)
                mask |= node_mask.get(var_id, 0)
                tangent = node_tangent.get(var_id)
                if tangent is None:
                    # 该入边 tangent 为零：跳过，不必构造 zeros_like 参与运算
                    continue
                local = _local_jvp(node, shadow, i, tangent, create_graph, cache)
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


def jvp(func: Callable[..., Union[TN, tuple[TN, ...]]],
        primals: Union[TN, list[TN], tuple[TN, ...]],
        tangents: Union[TN, list[TN], tuple[TN, ...]],
        create_graph: bool = False,
        strict: bool = False) -> tuple[Union[TN, tuple[TN, ...]], Union[TN, tuple[TN, ...]]]:
    """Jacobian-vector 积（前向模式）。

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


def vjp(func: Callable[..., Union[TN, tuple[TN, ...]]],
        primals: Union[TN, list[TN], tuple[TN, ...]],
        cotangents: Optional[Union[TN, list[TN], tuple[TN, ...]]] = None,
        create_graph: bool = False,
        strict: bool = False) -> tuple[Union[TN, tuple[TN, ...]], Union[TN, tuple[TN, ...]]]:
    """vector-Jacobian 积（反向模式）。

    先执行 func 的前向计算，再从输出到输入反向传播 cotangent，
    得到 vᵀ·J(x)。返回 (输出值, VJP 结果)。

    参数:
        func: 待求 VJP 的函数，输入为 TN，输出为单个 TN 或 TN 序列。
        primals: 求值点输入，单个 TN 或 TN 序列，顺序与 func 的入参一致。
        cotangents: 输出侧余切，结构须与输出一致；为 None 时取全 1，等价于
            对输出求和后求梯度。多输出时各输出的贡献会累加到同一输入梯度上。
        create_graph: 为 True 时保留计算图，VJP 结果可继续求导。
        strict: 为 True 时，若某个输出与某个输入完全不连通则抛 RuntimeError。

    返回:
        (输出值, VJP 结果)。输出值结构与 func 的返回一致；VJP 结果为各输入的
        余切传播结果——单输入时为单个 TN，多输入时为 tuple，每个元素形状与
        对应输入相同。
    """
    is_single_input = not isinstance(primals, (list, tuple))
    inputs = [primals] if is_single_input else list(primals)
    for i, inp in enumerate(inputs):
        if not isinstance(inp, TN):
            raise RuntimeError(f"Input {i} must be a tensor")
    inputs = _ensure_traceable(inputs)

    out_list, is_single_output = _call_and_split(func, inputs)

    if cotangents is None:
        cot_list = [ones_like(out, requires_grad=create_graph) for out in out_list]
    else:
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

    results = [None] * len(inputs)
    for out, cot in zip(out_list, cot_list):
        grads = _vjp_with_output(out, inputs, cot, create_graph, strict)
        for j, gv in enumerate(grads):
            results[j] = gv if results[j] is None else results[j] + gv

    out_values = out_list[0] if is_single_output else tuple(out_list)
    # 单个输入（含只含 1 个元素的序列）直接返回梯度张量，多输入返回元组
    vjp_values = results[0] if len(results) == 1 else tuple(results)
    return out_values, vjp_values


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


def _jacobian_via_vjp(func, args, positions, create_graph: bool, strict: bool):
    """用 vjp 原语计算 Jacobian：对输出的每个基向量做一次 VJP。

    func 只执行一次，所有列共享同一张前向计算图；每列仅做一次独立反向遍历
    （见 _vjp_with_output）。只为 positions 选中的输入收集梯度，未选中的参数
    既不参与 strict 检查，也不产生多余的反向遍历。

    参数:
        func: 待求 Jacobian 的函数。
        args: 完整实参序列（含非张量参数，非张量原样透传）。
        positions: 需要求导的参数位置序列。
        create_graph: 是否为高阶导数保留计算图。
        strict: 输出与输入不连通时是否报错。

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

        cot_batch = _identity_basis(numel, out.shape, out.dtype, out.device)

        def _single_vjp(cot, _out=out):
            # 复用已建立的前向计算图：VJP 关于 cotangent 线性，
            # 无需为每个输出基向量重跑 func。
            return _vjp_with_output(_out, tuple(inputs), cot, create_graph, strict)

        stacked = _as_list(vmap(_single_vjp)(cot_batch))
        for k, (blk, inp) in enumerate(zip(stacked, inputs)):
            # 每个输入的分量形状为 (numel,)，还原成 Jacobian 块
            # out.shape + inp.shape；dtype 统一向输出对齐
            blk = blk.reshape(out.shape + inp.shape)
            if blk.dtype != out.dtype:
                blk = blk.to(dtype=out.dtype)
            blocks[k][j] = blk

    return blocks, is_single_output


def _basis_row(basis: TN, k: int) -> TN:
    """取出基向量张量的第 k 行，构造共享底层数据、无计算图的叶子张量。

    tangent 只作为余切参与局部 JVP，从不被原地修改，因此无需视图元数据。

    参数:
        basis: _identity_basis 返回的基向量张量，形状 (numel,) + shape。
        k: 行下标，取值 [0, numel)。

    返回:
        与 basis[0] 同形、共享底层数据且 requires_grad=False 的叶子张量。
    """
    data = basis._array[k]
    # 输出为 0 维时基向量只有 1 维，整数索引会返回 numpy/cupy 标量而非
    # 0 维数组；统一转成数组，避免后续算子里 _get_array_lib/ndim 失效
    arrlib = basis._get_array_lib()
    if not isinstance(data, arrlib.ndarray):
        data = arrlib.asarray(data)
    return TN(data, requires_grad=False)


def _jacobian_via_jvp(func, args, positions, create_graph: bool):
    """用 jvp 原语计算 Jacobian：对每个输入基向量做一次 JVP。

    等价于 vmap(jvp)，但不经 vmap：前向计算图只建立一次，所有基向量共享
    同一张图、同一批影子节点以及同一份局部 VJP 图缓存（local_cache），
    避免每个基向量都重复执行 func 或重复构建局部梯度图。

    参数:
        func: 待求 Jacobian 的函数。
        args: 完整实参序列（含非张量参数，非张量原样透传）。
        positions: 需要求导的参数位置序列。
        create_graph: 是否为高阶导数保留计算图。

    返回:
        (blocks, is_single_output)。blocks[k][j] 是第 k 个被选中输入对第 j 个
        输出的 Jacobian，形状为 output_shape + input_shape。
    """
    call_args, primals, slot_of = _traceable_args(args)

    prepared = _prepare_forward(lambda: func(*call_args), primals, create_graph)
    out_list = prepared['out_list']
    # 局部 VJP 图按 (节点, 边) 缓存一次，所有基向量（列）共享
    local_cache = {}

    blocks = []
    for pos in positions:
        slot = slot_of.get(pos)
        if slot is None:
            raise RuntimeError(f"Input {pos} must be a tensor")
        inp = primals[slot]
        numel = inp.numel()
        if numel == 0:
            blocks.append([zeros(out.shape + inp.shape, dtype=out.dtype,
                                 device=out.device) for out in out_list])
            continue

        basis = _identity_basis(numel, inp.shape, inp.dtype, inp.device)

        # 逐列传播 tangent：直接遍历基向量的行（_basis_row 生成与基向量
        # 共享数据的无图叶子），省去 vmap 每列的切片、in_dims/out_dims
        # 归一化与参数容器重建开销。
        cols = None
        for k in range(numel):
            tangents = [None] * len(primals)
            tangents[slot] = _basis_row(basis, k)
            _, jvp_res = _jvp_with_prepared(prepared, tangents, create_graph, False,
                                            local_cache)
            res_list = _as_list(jvp_res)
            if cols is None:
                cols = [[] for _ in res_list]
            for j, res in enumerate(res_list):
                cols[j].append(res)

        # 每列结果的形状与 out 相同；沿最后一维堆叠得到 (*out.shape, numel)，
        # 再 reshape 成 (*out.shape, *inp.shape)，即第 k 个输入维度在末尾。
        blocks.append([stack(col, dim=-1).reshape(out.shape + inp.shape)
                       for col, out in zip(cols, out_list)])

    return blocks, prepared['is_single_output']


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


def _make_jacobian(compute_blocks, func, argnums, allow_none: bool, **opts):
    """构造 jacrev / jacfwd 共用的导函数：校验 argnums、算 blocks、统一格式化。

    参数:
        compute_blocks: 底层原语，签名为 (func, args, positions, **opts)，
            返回 (blocks, is_single_output)。
        func: 待求 Jacobian 的函数。
        argnums: 需要求导的参数位置，int、tuple[int, ...]；allow_none 时可为 None。
        allow_none: 是否允许 None 表示全部参数（jacfwd 支持，与 torch.func 一致）。
        **opts: 透传给 compute_blocks 的选项（create_graph / strict）。

    返回:
        Callable：接受与 func 相同的参数，返回 _format_jacobian 整理后的结果。
    """
    def _wrapper(*args, **kwargs):
        an = _check_argnums(argnums, allow_none=allow_none)
        positions = tuple(range(len(args))) if an is None else an
        blocks, is_single_output = compute_blocks(
            lambda *a: func(*a, **kwargs), args, positions, **opts)
        result = _format_jacobian(blocks, is_single_output)
        # argnums=None 表示“对全部参数求导”，此时即使只有 1 个参数也返回元组
        if an is None and not isinstance(result, tuple):
            return (result,)
        return result

    return _wrapper


def jacrev(func: Callable[..., Union[TN, tuple[TN, ...]]],
           argnums: Union[int, tuple[int, ...]] = 0,
           create_graph: bool = False,
           strict: bool = True) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """通过反向模式 (reverse-mode autodiff) 计算 Jacobian 矩阵。

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
        create_graph: 为 True 时 Jacobian 张量保留计算图，可继续求导。
        strict: 为 True 时，若某输出与某输入完全不连通则抛 RuntimeError（默认 True）。

    返回:
        Callable：func 的导函数，接受与 func 相同的参数并返回 Jacobian。
        结果结构与输出结构对应：单输出单输入时返回单个矩阵，形状为
        output_shape + input_shape；单输出多输入时返回 tuple，顺序与 argnums
        一致；多输出单输入时返回 tuple，顺序与输出顺序一致；多输出多输入时
        返回 tuple-of-tuples（输出在外层、输入在内层），内层顺序与 argnums 一致。
    """
    return _make_jacobian(_jacobian_via_vjp, func, argnums,
                          allow_none=False, create_graph=create_graph, strict=strict)


def jacfwd(func: Callable[..., Union[TN, tuple[TN, ...]]],
           argnums: Union[int, tuple[int, ...]] = 0,
           create_graph: bool = False) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """通过正向模式 (forward-mode autodiff) 计算 Jacobian 矩阵。

    实现等价于 vmap(jvp)：对每个输入元素构造 one-hot tangent，沿计算图正向传播
    得到 Jacobian 的一列，最后堆叠成完整 Jacobian。实际用逐列循环而非 vmap，
    且只做一次前向求值，所有列共享同一张计算图与同一批局部 VJP 图（见
    _jacobian_via_jvp）。

    参数:
        func: 待求 Jacobian 的函数，输入为 TN，输出为单个 TN 或 TN 序列。
        argnums: 需要计算 Jacobian 的输入位置，int、tuple[int, ...] 或 None，
                 默认 0。与 torch.func.jacfwd 一致：None 表示对全部参数求导
                 （此时即使只有 1 个参数也返回元组）；() 和含非 int 元素的元组、
                 list 等非法取值在调用导函数时抛 RuntimeError，错误信息与
                 torch.func 一致
        create_graph: 为 True 时 Jacobian 张量保留计算图，可继续求导。

    返回:
        Callable：func 的导函数，接受与 func 相同的参数并返回 Jacobian。
        结果结构与输出结构对应：单输出单输入时返回单个矩阵，形状为
        output_shape + input_shape（argnums=None 时即使只有 1 个参数也返回
        元组）；单输出多输入时返回 tuple，顺序与 argnums 一致；多输出单输入时
        返回 tuple，顺序与输出顺序一致；多输出多输入时返回 tuple-of-tuples
        （输出在外层、输入在内层），内层顺序与 argnums 一致。
    """
    return _make_jacobian(_jacobian_via_jvp, func, argnums,
                          allow_none=True, create_graph=create_graph)


# ---------------------------------------------------------------------------
# hessian
# ---------------------------------------------------------------------------

def hessian(func: Callable[..., TN],
            argnums: Union[int, tuple[int, ...]] = 0,
            create_graph: bool = False,
            strict: bool = True) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """计算标量值函数的 Hessian 矩阵，返回包裹函数。

    实现方式为 jacrev(grad(func))：先由反向模式 grad 得到梯度张量，
    再由反向模式 jacrev 对梯度求 Jacobian，即 Hessian。

    参数:
        func: 标量值函数，输出必须是 0 维 TN（ndim == 0），否则抛 RuntimeError。
        argnums: 需要计算 Hessian 的输入位置，int 或 tuple[int, ...]，默认 0；
                 非法取值（()、None、含非 int 元素的元组、list 等）在调用导函数时
                 抛 RuntimeError，错误信息与 torch.func.hessian 一致
        create_graph: 为 True 时 Hessian 张量保留计算图，可继续求导。
        strict: 为 True 时，若梯度与 argnums 选中的输入完全不连通则抛
                 RuntimeError（未被选中的参数不参与检查）。

    返回:
        Callable：func 的导函数，接受与 func 相同的参数并返回 Hessian。
        单个 argnums 位置时返回形状 input_shape + input_shape 的矩阵；多个位置时
        返回 tuple，第 j 项是第 argnums[j] 个输入的梯度对该输入的 Hessian 块。
    """
    def _wrapper(*args, **kwargs):
        an = _check_argnums(argnums)
        f = lambda *a: func(*a, **kwargs)

        output = f(*args)
        if not isinstance(output, TN) or output.ndim != 0:
            raise RuntimeError("Hessian only supports scalar-valued functions")

        # hessian = jacrev(grad(func))：反向叠反向。
        # 内层用 create_graph=True 求出标量输出梯度的张量（对 argnums 所选输入），
        # 外层 jacrev 再对梯度求 Jacobian，即得 Hessian。
        # 相比 jacrev(jacfwd(func))，梯度图只构建一次：本库的前向模式由局部 VJP
        # 转置模拟，jacfwd(create_graph=True) 会为每个基向量生成一条独立的列子图，
        # 外层每次反向都要遍历这些子图，导致 O(n^2) 放大；反向叠反向则避免该开销。
        # argnums 未覆盖全部参数时放宽 strict：部分 Hessian 块为零（如 ∂²(xy²)/∂x²）
        # 是正常结果，不应视为“未连接”。
        effective_strict = strict if len(an) == len(args) else False

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

        # 只对 argnums 所选输入求 Jacobian（未选中的参数既不参与 strict 检查，
        # 也不产生多余反向遍历）。blocks[j][j] 即第 j 个梯度对第 an[j] 个输入的
        # Hessian 块，因此非连续 argnums（如 (0, 2)）也能直接定位。
        blocks, _ = _jacobian_via_vjp(_grad_vector, args, an,
                                      create_graph=create_graph,
                                      strict=effective_strict)

        if len(an) == 1:
            return blocks[0][0]
        return tuple(blocks[j][j] for j in range(len(an)))

    return _wrapper


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------

def vmap(func: Callable[..., Union[TN, tuple[TN, ...]]],
         in_dims: Union[int, tuple[int, ...], list[Optional[int]]] = 0,
         out_dims: Union[int, tuple[int, ...], list[Optional[int]]] = 0) -> Callable[..., Union[TN, tuple[TN, ...]]]:
    """向量化映射 (vectorized map)。

    将原本作用于单个样本的函数扩展到批次维度上并行（通过 loop + 堆叠实现，
    不是真正的向量化，但接口与 torch.func.vmap 对齐）。

    参数:
        in_dims: 每个输入的批次维度。int 表示所有输入共用该维度；
                 tuple[int,...] 表示每个输入各自的维度位置；None 表示该输入不批处理。
        out_dims: 每个输出的批次维度，语义同 in_dims。

    返回:
        包裹函数，接受批次输入，返回批次输出。

    注意:
        - 切片使用内部 _batch_slice：等价于 inputs[batch_idx]，但跳过通用索引分派
        - 批次维通常是 dim=0；若指定 dim>0，切片索引为 (slice(None),)*d + (b,) + ...
        - 非批处理输入 (in_dims=None) 直接透传给底层 func
        - 输出需要梯度时用 stack 保留拼接的梯度边；纯前向时预分配缓冲写入
    """
    def _normalize_dims(dims, n_expected):
        if isinstance(dims, int):
            return [dims] * n_expected
        if isinstance(dims, (tuple, list)):
            return list(dims) + [None] * builtins.max(0, n_expected - len(dims))
        return [None] * n_expected

    def _wrapper(*args, **kwargs):
        n_inputs = len(args)
        in_dims_list = _normalize_dims(in_dims, n_inputs)

        # 预计算每个输入的切片计划（ndim, 批维位置 d）；None 表示该输入透传。
        # 循环内不再做 isinstance / 负维归一化 / 逐维生成 idx 元组。
        plans = []
        batch_size = None
        for dim, a in zip(in_dims_list, args):
            if dim is None or not isinstance(a, TN):
                plans.append(None)
                continue
            ndim = a.ndim
            d = dim if dim >= 0 else ndim + dim
            plans.append((ndim, d))
            if batch_size is None:
                batch_size = a.shape[d]

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

        # 堆叠结果
        first_out = outputs_list[0]
        is_single = isinstance(first_out, TN)
        out_dims_list = _normalize_dims(out_dims, 1 if is_single else len(first_out))

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

    # 归一化 args 为 tuple，避免单个 TN 被解包
    if isinstance(args, TN):
        args = (args,)
    elif isinstance(args, list):
        args = tuple(args)
    # args 已经是 tuple 时保持不变

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

    # 临时替换 module 的参数值，调用 forward，然后恢复。
    # 注意：不能用 setattr 替换普通 TN，因为 Module.__setattr__ 会把
    # 普通 TN 注册到 _buffers，而 Module.__getattr__ 查找顺序是
    # _parameters → _buffers → _modules，最终仍取到旧的 Parameter。
    # 直接操作 _parameters 字典才能让 self.weight 读到新值。
    orig_values = []
    _params = getattr(module, "_parameters", None)
    try:
        if _params is not None:
            for (name, orig), new_val in zip(module_params, parameter_values):
                orig_values.append(_params.get(name))
                _params[name] = new_val
        # 回退：没有 _parameters 字典的 module，退回到 setattr
        if _params is None:
            for (name, orig), new_val in zip(module_params, parameter_values):
                orig_values.append(orig)
                setattr(module, name, new_val)
        forward = getattr(module, "forward", None) or module
        return forward(*args, **kwargs)
    finally:
        if _params is not None:
            for (name, _), orig in zip(module_params, orig_values):
                _params[name] = orig
        else:
            for (name, _), orig in zip(module_params, orig_values):
                setattr(module, name, orig)
