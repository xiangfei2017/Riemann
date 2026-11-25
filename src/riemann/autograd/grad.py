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
- `backward`: Multi-root reverse-mode autodiff, compatible with torch.autograd.backward
- `grad`: Computes gradients of one or more outputs w.r.t. inputs, returning gradient tuples directly
- `gradcheck`: Numerical gradient verification via finite differences
- `Function`: Base class for custom autograd.Function-style forward/backward definitions

All functions support Riemann Tensor (TN) type and automatic differentiation.
"""

from __future__ import annotations

from typing import Callable, Any
import numpy as np
# 显式导入所需符号：星号导入会把 abs/round 等内置名遮蔽成张量版本，
# 容易在误差比较等纯数值场景中误用（历史上 gradcheck 就踩过这个坑）
from ..core import TN, stack, set_grad_enabled, tensor, zeros_like, api_registry
from ..core.tensordef import (_same_device,
                            _init_calc_graph,
                            _backward_with_hooks,
                            _backward_without_hooks,
                            _propagate_grad_to_sources
)
from ..core.ad_context import forward_ad_trace_active, is_grad_enabled


def _normalize_roots(outputs, arg_name):
    """把单个 TN 或 TN 序列统一规范化为 list，并校验可作为反向根。

    规则（与 torch 对齐）：
      - 单个 TN 视为只含一个根的序列；
      - 序列不能为空（没有根就无法进行反向传播）；
      - 每个元素都必须是 requires_grad=True 的 TN。

    统一返回 list：后续 _merge_seeds_and_ready_roots 需要按位置与 seeds 对齐。
    arg_name 仅用于拼接错误信息。
    """
    roots = [outputs] if not isinstance(outputs, (list, tuple)) else list(outputs)
    if not roots:
        raise RuntimeError(f'{arg_name} must contain at least one tensor')
    if any(not isinstance(out, TN) or not out.requires_grad for out in roots):
        raise RuntimeError(f'all elements of {arg_name} must be TN and require grad')
    return roots


def _normalize_inputs(inputs, *, require_grad):
    """规范化 inputs，返回 (input_list, input_ids)；inputs 为 None 时返回 (None, None)。

    返回的 id 集合用于反向遍历时 O(1) 判断某节点是否属于 inputs。
    inputs 为 None 表示“不过滤”，由 _save_grad 走叶子/retains_grad 默认规则。

    require_grad 区分两个调用方：
      - grad():    True，逐项要求 TN 且 requires_grad=True（要对它求导）；
      - backward(): False，只要求是 TN（仅作为 .grad 的累积目标过滤）。

    用 id 而不是张量本身建集合：TN.__eq__ 是逐元素比较，不能用于成员判断。
    """
    if inputs is None:
        return None, None
    input_list = [inputs] if not isinstance(inputs, (list, tuple)) else list(inputs)
    if require_grad:
        for i, inp in enumerate(input_list):
            if not isinstance(inp, TN) or not inp.requires_grad:
                raise RuntimeError(f"Input {i} must be a tensor that requires grad")
    elif any(not isinstance(inp, TN) for inp in input_list):
        raise TypeError('inputs must contain only TN tensors')
    return input_list, {id(t) for t in input_list}


def _build_seeds(roots, grad_outputs, *, grads_name, create_graph,
                 scalar_error, shape_error, type_error):
    """生成与 roots 按位置对齐的初始余切（seed）列表。

    grad_outputs 支持三种形态：
      - None         : 每个根都必须能提供隐式种子（值为 1.0）；
      - 单个 TN      : 仅当只有一个根时合法；
      - TN/None 序列 : 长度必须与 roots 相等，逐位置对应。

    scalar_error / shape_error / type_error 是带 {i}/{shape}/{expected}/
    {got}/{type} 命名占位符的模板，用于保留 backward 与 grad 各自文案。

    关键点：
      - 隐式种子只在根是实标量时合法；create_graph=True 时该 1.0 参与建图；
      - 显式余切不 detach（与 torch 一致，VJP 关于余切线性）；
      - 显式余切必须与对应根同形状、同设备，否则直接抛错。
    """
    if grad_outputs is None:
        grad_list = [None] * len(roots)
    elif isinstance(grad_outputs, TN):
        if len(roots) != 1:
            raise RuntimeError(
                f'{grads_name} must have length {len(roots)}, but got 1')
        grad_list = [grad_outputs]
    else:
        grad_list = list(grad_outputs)
        if len(grad_list) != len(roots):
            raise RuntimeError(
                f'{grads_name} must have length {len(roots)}, '
                f'but got {len(grad_list)}')

    seeds = []
    for i, (out, g) in enumerate(zip(roots, grad_list)):
        if g is None:
            if out._array.ndim > 0:
                raise RuntimeError(scalar_error.format(i=i, shape=out.shape))
            if out.is_complex():
                raise RuntimeError(
                    'grad can be implicitly created only for real scalar outputs '
                    f'but got {out.dtype}')
            seeds.append(tensor(1.0, dtype=out.dtype, device=out.device,
                                requires_grad=create_graph))
        elif isinstance(g, TN):
            if g._array.shape != out._array.shape:
                raise RuntimeError(
                    shape_error.format(i=i, expected=out.shape, got=g.shape))
            if not _same_device(g, out):
                raise RuntimeError(
                    f'invalid gradient at index {i} - expected device {out.device} '
                    f'but got {g.device}')
            # 与 torch 一致，不对余切做 detach
            seeds.append(g)
        else:
            raise TypeError(type_error.format(i=i, type=type(g)))
    return seeds


def _merge_seeds_and_ready_roots(roots, seeds):
    """写入根的初始余切，并返回当前即可执行反向传播的根列表（栈初值）。

    两个关键处理：
      1. 去重合并：同一张量在 roots 中多次出现时按 id 合并其 seed，
         与 torch 一致（重复根等价于 seed 相加，而非各自独立传播）；
      2. 就绪判定：只有 _rcv_grad_count == 0 的根才能立即入栈。
         被其它根依赖的根计数 > 0，必须等上游根的梯度传到后再入栈。

    前置条件：_init_calc_graph 已重置所有可达节点的 _grad_value 并重建
    _rcv_grad_count。本函数只做赋值与入栈，不重复初始化图。
    """
    merged = {}
    for root, seed in zip(roots, seeds):
        key = id(root)
        entry = merged.get(key)
        merged[key] = (root, seed) if entry is None else (root, entry[1] + seed)

    stack = []
    for root, seed in merged.values():
        root._grad_value = seed
        if root._rcv_grad_count == 0:
            stack.append(root)
    return stack


def backward(tensors,
             grad_tensors=None,
             retain_graph: bool|None = None,
             create_graph: bool = False,
             inputs=None) -> None:
    """计算给定张量对图叶子节点的梯度之和，并累积到 .grad。

    与 torch.autograd.backward 的接口一致（去掉了已废弃的 grad_variables）：

    参数：
        tensors: 求导的起点，单个 TN 或 TN 序列。
        grad_tensors: 各输出对应的“向量”，单个 TN、TN/None 序列或 None。
                      为 None 时要求每个输出都是实标量，隐式使用 1.0。
        retain_graph: 仅为与 PyTorch 兼容保留。Riemann 的反向传播不依赖
                      此参数，始终支持多次调用。
        create_graph: 是否为梯度计算建图，用于高阶导数。
        inputs: 单个 TN 或 TN 序列。提供时只向这些张量累积 .grad，
                其余叶子节点不接收梯度。

    返回：
        None: 梯度结果直接累积到相关张量的 .grad 属性中。

    关键点：
        - grad_tensors 是各根的初始余切，直接赋给根的 _grad_value，不做累加；
          .grad 的累加发生在反向遍历的 _save_grad 阶段。
        - 多根时若某根同时是另一个根的前驱，其 _rcv_grad_count > 0，
          要等上游根传播到达后才入栈，由 _merge_seeds_and_ready_roots 处理。
        - 重复根等价于 seed 相加（由 _merge_seeds_and_ready_roots 去重合并）。
        - inputs=None 时：叶子总是累积 .grad，中间节点仅 retains_grad=True 时累积；
          提供 inputs 时只向这些张量累积，其余节点不写 .grad。
        - 与 grad 共用 _init_calc_graph / _merge_seeds_and_ready_roots，
          区别只是用 _save_grad 累积 .grad，而不是返回梯度元组。
    """
    roots = _normalize_roots(tensors, 'tensors')
    seeds = _build_seeds(
        roots, grad_tensors,
        grads_name='grad_tensors',
        create_graph=create_graph,
        scalar_error='grad can be implicitly created only for scalar outputs, '
                     'but tensors[{i}] has shape {shape}',
        shape_error='invalid gradient at index {i} - expected shape {expected} '
                    'but got {got}',
        type_error='grad_tensors[{i}] must be a TN or None, but got {type}',
    )

    # inputs 只用于过滤 .grad，因此不要求 requires_grad
    _, input_ids = _normalize_inputs(inputs, require_grad=False)

    need_hook_processing = _init_calc_graph(roots)
    stack = _merge_seeds_and_ready_roots(roots, seeds)

    # create_graph 决定梯度计算本身是否被追踪
    with set_grad_enabled(create_graph):
        if need_hook_processing:
            _backward_with_hooks(stack, input_ids)
        else:
            _backward_without_hooks(stack, input_ids)


def grad(outputs:TN|list[TN]|tuple[TN],
         inputs:TN|list[TN]|tuple[TN],
         retain_graph: bool = False,
         grad_outputs:TN|list[TN|None]|tuple[TN|None]|None=None,
         create_graph : bool = False,
         allow_unused: bool = False)->tuple[TN,...]:
    """
    计算给定输出（可多根）相对于输入的梯度，直接返回梯度张量而不修改原始张量。
    
    这是Riemann框架中的核心梯度计算函数。与backward()方法类似，
    但它直接返回计算出的梯度张量，而不是将它们存储在输入张量的.grad属性中。
    这使其更适合于高级梯度计算场景，如计算雅可比矩阵、海森矩阵等。
    
    工作原理：
        1. 规范化 outputs 为根列表并校验 inputs，初始化计算图（_init_calc_graph）
        2. 为每个根生成初始余切（标量隐式 1.0 或用户提供的 grad_outputs），
           重复根的余切相加
        3. 使用 list 作为栈（pop(-1) O(1)）对计算图进行 DFS 反向遍历
        4. 在 with set_grad_enabled(create_graph) 上下文中执行梯度传播，
           确保 create_graph 参数能正确控制中间张量的 autograd 追踪
        5. 复用 _propagate_grad_to_sources 向来源节点传播梯度，
           与 _backward_without_hooks 使用完全相同的传播逻辑
        6. 节点出栈时先检查是否是 inputs 之一，若是则保存梯度到返回列表；
           所有 inputs 的梯度一旦全部收集（O(1) 计数器检测），立即 break 提前退出
    
    与backward()方法的区别：
        1. backward(): 将梯度存储在张量的.grad属性中，适合参数更新
        2. grad(): 返回梯度张量而不修改原始张量，适合高级梯度计算
        3. grad() 在计算完所需梯度后立即停止（lazy evaluation），而backward()遍历整个计算图
        4. grad() 不累积梯度，每次调用都重新计算；backward() 的 .grad 会累加

    关键点：
        - 重复输入：预先把 inputs 整理成 id(tensor) → 槽位列表的映射，
          出栈节点只需一次字典查找就能填满它的全部槽位，返回元组中重复项
          都是同一个梯度值，且顺序不变。
        - 图状态不主动清理：下次 grad/backward 的 _init_calc_graph 会覆盖所有
          可达节点的状态，因此无需在遍历后逐个清理。
        - 多根：outputs 可为单个 TN 或序列。重复根等价于余切相加；某个根若
          同时是另一个根的前驱，会等上游根的梯度传播到达后才入栈。
    
    参数：
        outputs (TN | list[TN] | tuple[TN]): 一个或多个需要计算梯度的输出张量
        inputs (TN | list[TN] | tuple[TN]): 需要计算梯度的输入张量或张量列表
        grad_outputs (TN | list[TN|None] | tuple[TN|None], 可选): 各输出的梯度
            - None: 默认值，要求每个输出都是标量，梯度将设置为1.0
            - 单个 TN: 仅当只有一个输出时合法，必须与输出形状相同
            - 序列: 与 outputs 按位置对应，长度必须一致，元素可为 TN 或 None
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
            1. outputs 为空，或 outputs/inputs 含非 TN、requires_grad=False 的元素
            2. 当未提供grad_outputs时，某个输出不是标量
            3. 当未提供grad_outputs时，某个输出是复标量
            4. grad_outputs长度与输出根数不一致，或某个梯度与对应输出形状/设备不匹配
            5. 在allow_unused=False模式下，当检测到输出与某个输入无关时
    
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
        
        >>> # 多输出：grad_outputs 与 outputs 按位置对应
        >>> x = tensor([1.0, 2.0], requires_grad=True)
        >>> y1, y2 = x.sum(), (x * x).sum()
        >>> gx, = grad([y1, y2], x, grad_outputs=[tensor(1.0), tensor(1.0)])
        >>> print(gx)  # [3. 5.]
        
        >>> # 高阶导数计算
        >>> x = tensor(2.0, requires_grad=True)
        >>> y = x * x * x  # y = x^3
        >>> dy_dx = grad(y, x, create_graph=True)[0]  # dy/dx = 3x^2
        >>> d2y_dx2 = grad(dy_dx, x)[0]  # d²y/dx² = 6x
        >>> print(d2y_dx2)  # 输出: [12.]
        
    注意：
        设置 create_graph=True 会创建额外的计算图，可能增加内存占用。
    """
    roots = _normalize_roots(outputs, 'outputs')
    inputs, _ = _normalize_inputs(inputs, require_grad=True)

    seeds = _build_seeds(
        roots, grad_outputs,
        grads_name='grad_outputs',
        create_graph=create_graph,
        scalar_error='Only a scalar can call grad() without argument grad_outputs',
        shape_error='shape of grad_outputs need to be same as the shape of outputs',
        type_error='grad_outputs can be either tensor or None, but got {type}',
    )

    _init_calc_graph(roots)
    stack = _merge_seeds_and_ready_roots(roots, seeds)

    grad_list = [None] * len(inputs)

    # id(tensor) → grad_list 槽位，一次查找即可填满重复输入的全部槽位
    input_slots: dict[int, list[int]] = {}
    for i, var in enumerate(inputs):
        input_slots.setdefault(id(var), []).append(i)

    saved_count = 0   # 已填充槽位数，用于 O(1) 判断是否收集完成

    with set_grad_enabled(create_graph):
        while stack:
            item: TN = stack.pop(-1)

            slots = input_slots.get(id(item))
            if slots is not None:
                grad_val = item._grad_value if create_graph else item._grad_value.detach()
                for i in slots:
                    grad_list[i] = grad_val
                saved_count += len(slots)

            if saved_count == len(inputs):
                break

            # 只有收齐全部上游贡献（计数归零）的来源才会被加入待入栈集合
            vars_received_grad = set()
            _propagate_grad_to_sources(item, vars_received_grad)
            item._grad_value = None
            stack.extend(vars_received_grad)

    if not allow_unused:
        for i, grad_val in enumerate(grad_list):
            if grad_val is None:
                raise RuntimeError(f"Output is independent of input ({i}), Set allow_unused=True if this is the desired behavior.")

    return tuple(grad_list)  # type: ignore

@api_registry.register
def diff(tensor, *args, create_graph=False):
    r"""计算标量张量对指定变量的高阶偏导数或混合偏导数。

    实现 TN.d 的核心逻辑，由 TN.d 委托调用。

    变量可以是任意形状（标量、向量、矩阵等）。同一变量可以多次出现，
    阶数自动累加。按 args 顺序逐变量求导，输出形状 = 各变量形状按顺序扁平拼接。

    参数:
        tensor: 标量张量，必须 requires_grad=True
        *args: 导数规格，两种形式可混用:
            - TN                对该变量求一阶
            - (TN, k)           对该变量求 k 阶 (k >= 1)
        create_graph: 是否追踪结果计算图以支持继续高阶微分。默认 False。

    返回:
        张量，形状为各变量形状的扁平拼接。

    异常:
        ValueError: tensor 非标量或无梯度；args 含非法项
    """
    
    # ---- 1. 前置校验 ----
    if tensor.ndim > 0 or not tensor.requires_grad:
        raise ValueError("tensor must be a scalar tensor with requires_grad=True")
    if not args:
        raise ValueError("At least one variable must be provided")

    # ---- 2. 解析 args → [(var, 总阶数), ...] ----
    # 先累加同一变量的阶数，再从 dict 构造 specs
    vid_to_order: dict[int, int] = {}   # id(var) → 累计阶数
    var_order: list[TN] = []             # 保持首次出现的顺序

    for item in args:
        if isinstance(item, TN):
            var, k = item, 1
        elif isinstance(item, (tuple, list)) and len(item) == 2 \
                and isinstance(item[0], TN) and isinstance(item[1], int):
            var, k = item
        else:
            raise ValueError(
                f"Each arg must be a TN or (TN, int), got {type(item).__name__}"
            )
        if k < 1:
            raise ValueError(f"Derivative order k must be >= 1, got {k}")
        if not var.requires_grad:
            raise ValueError("Variable must have requires_grad=True")

        vid = id(var)
        if vid in vid_to_order:
            vid_to_order[vid] += k
        else:
            vid_to_order[vid] = k
            var_order.append(var)

    specs: list[tuple[TN, int]] = [(var, vid_to_order[id(var)]) for var in var_order]

    # ---- 3. 预计算 ----
    total_steps = 0
    for _, k in specs:
        total_steps += k
    final_shape: tuple[int, ...] = ()
    for var, k in specs:
        final_shape += var.shape * k

    # ---- 4. 核心: total_steps 次顺序 grad ----
    current: TN = tensor
    step = 0

    for var, order in specs:
        for _ in range(order):
            is_last = (step == total_steps - 1)
            need_graph = create_graph if is_last else True  # 中间步骤必须追踪图

            if current.ndim == 0:
                # 标量 → 一次 grad
                g = grad(current, var,
                         create_graph=need_graph,
                         allow_unused=True)[0]
                if g is None:
                    return zeros_like(tensor, requires_grad=False)
                current = g
            else:
                # 张量 → 逐元素标量 grad，再 stack 保持计算图
                # 注：未提供 grad_outputs 时 grad 要求每个输出根是标量，
                #     因此这里对张量 outputs 必须逐个元素求导再拼回形状
                flat = current.reshape(-1)
                elem_grads: list[TN] = []
                for ei in range(flat.shape[0]):
                    g_elem = grad(flat[ei], var,
                                  create_graph=need_graph,
                                  allow_unused=True)[0]
                    if g_elem is None:
                        elem_grads.append(zeros_like(var, requires_grad=False))
                    else:
                        elem_grads.append(g_elem)

                stacked = stack(elem_grads, dim=0)
                target_shape = current.shape + var.shape
                current = stacked if stacked.shape == target_shape \
                    else stacked.reshape(*target_shape)

            step += 1

    # ---- 5. reshape 回最终形状 ----
    if current.shape != final_shape:
        current = current.reshape(*final_shape)
    return current


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
                    invalids = np.abs(num_col) > (atol + nondet_tol)
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
            abs_error = np.abs(num_val - an_val)
            rel_error = abs_error / (np.abs(num_val) + np.abs(an_val) + 1e-10)

            if abs_error > effective_atol + rtol * np.abs(an_val) and abs_error > nondet_tol:
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
    - jvp（可选）：接收各输入的前向 tangent（无 tangent 的输入为 None），
      返回输出的 tangent；实现后该函数即可参与前向自动微分(jvp)

    示例用法：
    class MyFunction(Function):
        @staticmethod
        def forward(ctx, input1, input2):
            ctx.save_for_backward(input1, input2)
            ctx.save_for_forward(input1, input2)
            output = input1 * input2
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            input1, input2 = ctx.saved_tensors
            grad_input1 = grad_output * input2
            grad_input2 = grad_output * input1
            return grad_input1, grad_input2

        @staticmethod
        def jvp(ctx, tangent1, tangent2):
            input1, input2 = ctx.saved_tensors
            total = None
            if tangent1 is not None:
                total = tangent1 * input2
            if tangent2 is not None:
                term = input1 * tangent2
                total = term if total is None else total + term
            return total
    """
    
    class _Context:
        """
        上下文类，用于在forward、backward与jvp之间传递信息

        saved_tensors 的语义与 torch.autograd.Function 完全对齐：
        - backward 中读取 save_for_backward 保存的张量
        - jvp 中只读取 save_for_forward 保存的张量（未调用时为空元组），
          不会看到 save_for_backward 的内容
        """
        def __init__(self):
            self.saved_tensors = ()
            self.saved_for_forward = None
            self.needs_input_grad = ()
        
        def save_for_backward(self, *tensors):
            """保存张量，以便在backward中使用"""
            self.saved_tensors = tensors

        def save_for_forward(self, *tensors):
            """保存张量，以便在jvp中使用；非张量应直接保存为 ctx 属性"""
            for tensor in tensors:
                if not (isinstance(tensor, TN) or tensor is None):
                    raise AssertionError(
                        "save_for_forward expects all arguments to be tensors; "
                        "you should save non-tensors as attributes on ctx.")
            self.saved_for_forward = tensors
    
    @staticmethod
    def forward(ctx, *inputs):
        """
        前向传播函数，需要被子类实现，支持与 torch 相同的两种写法：

        用法一（forward 接收 ctx）::

            @staticmethod
            def forward(ctx, *args):
                ctx.save_for_backward(...)
                return ...

        用法二（forward 不接收 ctx，另实现 setup_context）::

            @staticmethod
            def forward(*args):
                return ...

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.save_for_backward(...)

        参数：
            ctx: 上下文对象，用于保存信息供backward使用（仅用法一）
            *inputs: 输入张量

        返回：
            输出张量
        """
        raise NotImplementedError("forward() must be implemented in subclass")

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        新式 forward 的上下文构建函数，可选实现。

        实现本方法后，forward 不再接收 ctx，由 apply() 在 forward 返回后调用本方法；
        ctx 上的内容供 backward 与 jvp 使用。未实现时采用老式
        forward(ctx, ...) 写法。

        参数：
            ctx: 上下文对象
            inputs: forward 的输入元组（原样传入）
            output: forward 的返回值（转换为 TN 之前）
        """
        raise NotImplementedError("setup_context is not implemented.")
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播函数，需要被子类实现
        
        参数：
            ctx: 上下文对象，包含forward中保存的信息，以及
                ctx.needs_input_grad（每个输入是否需要梯度的布尔元组）
            grad_output: 输出张量的梯度
        
        返回：
            输入张量的梯度
        """
        raise NotImplementedError("backward() must be implemented in subclass")

    @staticmethod
    def jvp(ctx, *grad_inputs):
        """
        前向传播的 JVP 规则，可选实现；实现后 apply() 会登记对应的 _jvp_rule。

        参数：
            ctx: 上下文对象；读取 ctx.saved_tensors 只能拿到 forward 中通过
                ctx.save_for_forward(...) 保存的张量（与 torch 一致，未保存时
                为空元组），save_for_backward 的内容在此不可见；
                其余属性与 forward 中写入的一致
            *grad_inputs: 与 forward 的 inputs 一一对应的输入 tangent，
                没有 tangent 的输入为 None

        返回：
            输出张量的 tangent；若输出不可微可返回 None。
            支持高阶求导时，应使用 TN 算子由输入 tangent 与原始输入构造结果。
        """
        raise NotImplementedError(
            "jvp() is optional; implement it in subclass to support forward AD")
    
    @classmethod
    def apply(cls, *inputs):
        """
        应用函数到输入张量，执行前向计算并设置梯度/前向规则
        
        参数：
            *inputs: 输入张量，也可包含非张量参数
        
        返回：
            输出张量
        """
        ctx = cls._Context()
        # 与 torch 一致：每个输入是否需要梯度；梯度模式关闭时全为 False
        grad_enabled = is_grad_enabled()
        ctx.needs_input_grad = tuple(
            grad_enabled and isinstance(item, TN) and item.requires_grad
            for item in inputs)

        # 与 torch 一致：实现 setup_context 时 forward 不收 ctx（新式），
        # 否则由 forward 自己接收 ctx（老式）
        if cls.setup_context is not Function.setup_context:
            outputs = cls.forward(*inputs)
            cls.setup_context(ctx, inputs, outputs)
        else:
            outputs = cls.forward(ctx, *inputs)

        first_tn = next((item for item in inputs if isinstance(item, TN)), None)
        # 如果输出不是张量，将其转换为张量
        if not isinstance(outputs, TN):
            outputs = tensor(outputs,
                             device=first_tn.device if first_tn is not None else None)
        
        # 检查是否需要计算梯度（与 torch 一致：梯度模式关闭时不建图）
        requires_grad = grad_enabled and any(
            isinstance(item, TN) and item.requires_grad for item in inputs)
        trace = forward_ad_trace_active()

        if requires_grad or trace:
            # 自定义Function会覆盖forward内部算子的计算图，旧融合钩子不再有效
            outputs._grad_rule = None
            if requires_grad:
                # 设置输出张量的梯度跟踪信息
                outputs.requires_grad = True

            # fromvars 同时服务反向与图边：前向跟踪时包含全部 TN 输入，
            # 保证没有 requires_grad、但带 tangent 的输入也能沿边传播；
            # 其余情况保持原有“只含 requires_grad 输入”的反向语义
            if trace:
                source_indices = tuple(idx for idx, item in enumerate(inputs)
                                       if isinstance(item, TN))
            else:
                source_indices = tuple(idx for idx, item in enumerate(inputs)
                                       if isinstance(item, TN) and item.requires_grad)

            def _apply_grad(result_tensor: TN):
                # 执行一次反向传播
                grad_inputs = cls.backward(ctx, result_tensor._grad_value)
                
                # 确保返回值是元组
                if not isinstance(grad_inputs, tuple):
                    grad_inputs = (grad_inputs,)
                
                # 按图来源顺序返回所需的输入梯度
                return tuple(grad_inputs[idx] for idx in source_indices)
            
            # 设置fromvars和梯度规则
            outputs.fromvars = tuple(inputs[idx] for idx in source_indices)
            outputs._grad_rule = _apply_grad
            outputs.parms = ()

            jvp_impl = cls.jvp
            if trace and jvp_impl is not Function.jvp:
                def _apply_jvp(frame, result_tensor: TN):
                    input_tangents = tuple(
                        frame.node_tangent.get(id(item)) if isinstance(item, TN) else None
                        for item in inputs
                    )
                    if all(tangent is None for tangent in input_tangents):
                        return None
                    # 与 torch 完全对齐：jvp 执行期间 saved_tensors 只暴露
                    # save_for_forward 保存的张量（未调用时为空元组）
                    saved = ctx.saved_tensors
                    ctx.saved_tensors = () if ctx.saved_for_forward is None else ctx.saved_for_forward
                    try:
                        output_tangent = jvp_impl(ctx, *input_tangents)
                    finally:
                        ctx.saved_tensors = saved
                    if output_tangent is None:
                        return None
                    if not isinstance(output_tangent, TN):
                        output_tangent = tensor(output_tangent,
                                                device=result_tensor.device)
                    return output_tangent

                outputs._jvp_rule = _apply_jvp
            elif trace:
                # 未实现 jvp：保持 None；jvp 传播遇到该节点会报错并提示实现
                # Function.jvp
                outputs._jvp_rule = None

        return outputs
# end of Function class