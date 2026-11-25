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
Riemann Library AD Context Module: Automatic Differentiation Control

This module provides automatic differentiation context control functionality for the
Riemann automatic differentiation framework. It implements context managers and decorators
to enable or disable gradient tracking during computation, similar to PyTorch's
corresponding functionality. The module uses thread-local storage to ensure
thread-safe operations in multi-threaded environments.

Main features:
    - Gradient calculation control: Enable or disable gradient tracking during computation
    - Nested mode management: Support for nested gradient calculation mode contexts
    - Multiple usage patterns: Context managers and decorators for flexible usage
    - Thread-safe operations: Each thread maintains its own gradient calculation mode stack
    - Memory optimization: Disable gradient tracking during inference to reduce memory usage
    - Performance optimization: Accelerate computation when gradients are not needed

Using this module, you can optimize memory usage and computation speed during model
inference by disabling gradient tracking, while maintaining full gradient computation
during training phases.

Example usage:
    >>> import riemann as rm
    >>> # Disable gradient tracking for inference
    >>> with rm.no_grad():
    ...     output = model(input_data)
    >>> # Enable gradient tracking for training
    >>> with rm.enable_grad():
    ...     output = model(input_data)
    ...     loss = loss_fn(output, target)
    ...     loss.backward()
"""

import contextvars
import functools
import threading

# 创建线程局部存储，用于控制梯度计算栈
_grad_mode_stack = threading.local()


def init_grad_mode_stack():
    """
    初始化梯度计算模式栈
    
    为当前线程创建梯度计算模式栈，如果栈不存在的话，
    并设置默认值为 True（启用梯度计算）。
    
    注意：
        此函数通常不需要手动调用，其他梯度控制函数会自动确保栈已初始化。
    """
    try:
        _grad_mode_stack.stack
    except AttributeError:
        _grad_mode_stack.stack = [True]  # 默认为启用梯度计算


def is_grad_enabled():
    """
    获取当前线程的梯度计算状态
    
    返回栈顶元素，表示当前的梯度计算模式（True 表示启用，False 表示禁用）。
    如果梯度模式栈尚未初始化，则会自动调用 init_grad_mode_stack() 进行初始化。
    
    返回：
        bool: 当前梯度计算模式，True 表示启用，False 表示禁用
    """
    try:
        return _grad_mode_stack.stack[-1]
    except AttributeError:
        init_grad_mode_stack()  # 确保栈已初始化
        return _grad_mode_stack.stack[-1]


class _GradContext:
    """
    梯度计算模式上下文管理器（模块级别，可复用）

    支持通过 mode 参数控制是否启用梯度计算，
    同时实现了 __call__ 方法以支持装饰器语法。
    """

    __slots__ = ('mode',)

    def __init__(self, mode_val):
        """
        初始化上下文管理器

        参数：
            mode_val (bool): 梯度计算模式，True 启用，False 禁用
        """
        self.mode = mode_val

    def __enter__(self):
        """
        进入上下文，应用新模式

        返回：
            self: 上下文管理器实例
        """
        # 内联：确保栈已初始化并压入新模式
        try:
            stack = _grad_mode_stack.stack
        except AttributeError:
            _grad_mode_stack.stack = [True]
            stack = _grad_mode_stack.stack
        stack.append(self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文，恢复之前的梯度状态

        参数：
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常追踪信息

        返回：
            bool: False 表示不抑制异常
        """
        # 内联：弹出栈顶模式（保留至少一个默认模式）
        stack = _grad_mode_stack.stack
        if len(stack) > 1:
            stack.pop()
        return False

    def __call__(self, func_to_wrap):
        """
        使上下文管理器可以作为装饰器使用

        参数：
            func_to_wrap: 要包装的函数

        返回：
            包装后的函数
        """
        @functools.wraps(func_to_wrap)
        def wrapper(*args, **kwargs):
            with _GradContext(self.mode):
                return func_to_wrap(*args, **kwargs)
        return wrapper


def no_grad(func=None):
    """
    上下文管理器，用于暂时禁用梯度计算

    在这个上下文中，所有计算将不会追踪梯度，类似于 PyTorch 的 no_grad()。
    适用于推理阶段，可显著减少内存使用并加速计算。

    也可以作为函数装饰器使用，禁用被装饰函数内所有计算的梯度追踪。

    参数：
        func: 可选，如果提供，则将 no_grad 作为装饰器应用于该函数

    返回：
        如果未提供 func，则返回上下文管理器实例
        如果提供了 func，则返回装饰后的函数

    示例：
        # 作为上下文管理器使用
        with no_grad():
            # 这段代码中的计算不会追踪梯度
            y = model(x)

        # 作为装饰器使用
        @no_grad
        def inference(x):
            # 函数内的计算不会追踪梯度
            return model(x)
    """
    return set_grad_enabled(False, func)


def enable_grad(func=None):
    """
    上下文管理器，用于暂时启用梯度计算

    在这个上下文中，所有计算将追踪梯度，类似于 PyTorch 的 enable_grad()。
    可用于在 no_grad 上下文中临时启用梯度计算。

    也可以作为函数装饰器使用，确保被装饰函数内的计算追踪梯度。

    参数：
        func: 可选，如果提供，则将 enable_grad 作为装饰器应用于该函数

    返回：
        如果未提供 func，则返回上下文管理器实例
        如果提供了 func，则返回装饰后的函数

    示例：
        # 作为上下文管理器使用
        with no_grad():
            # 这里禁用了梯度
            with enable_grad():
                # 这里临时启用了梯度
                y = model(x)
                y.backward()
            # 回到禁用梯度的状态

        # 作为装饰器使用
        @enable_grad
        def train_step(x, y):
            # 函数内的计算会追踪梯度
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            return loss
    """
    return set_grad_enabled(True, func)


def set_grad_enabled(mode=True, func=None):
    """
    上下文管理器，用于设置梯度计算模式

    类似于 PyTorch 的 set_grad_enabled()，可以显式地启用或禁用梯度计算。
    支持作为上下文管理器或装饰器使用，提供最灵活的梯度控制方式。

    参数：
        mode (bool): 如果为 True，则启用梯度计算；如果为 False，则禁用梯度计算
        func: 可选，当作为装饰器使用时传入的函数

    返回：
        如果 func 为 None，返回上下文管理器实例
        如果提供了 func 参数，返回包装后的函数

    示例：
        # 作为上下文管理器使用
        with set_grad_enabled(False):
            # 这段代码中的计算不会追踪梯度
            y = model(x)

        with set_grad_enabled(True):
            # 这段代码中的计算会追踪梯度
            y = model(x)
            y.backward()

        # 作为装饰器使用
        @set_grad_enabled(False)
        def inference(x):
            return model(x)

        @set_grad_enabled(True)
        def train(x, y):
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            return loss
    """
    # 如果func为None，返回上下文管理器实例
    if func is None:
        return _GradContext(mode)

    # 如果提供了func参数，直接返回包装后的函数
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _GradContext(mode):
            return func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# 前向 AD（JVP）运行时上下文
#
# 三个概念
# --------
#   ContextVar _forward_ad_state
#            │ 当前线程/协程持有
#            ▼
#   _ForwardADState（不可变快照）
#     └─ frames: (frame0, frame1, ...)   栈本体，末尾为栈顶；空表示无前向请求
#            │ 每个元素是一层
#            ▼
#   JVPFrame（一次 JVP / 一个切方向的运行时状态）
#     ├─ node_tangent / dependency_mask  本次传播的数据
#     └─ strict / phase                  本次传播的配置与阶段
#
# 关系
# ----
# - ContextVar ↔ 快照：set() 整体替换并返回 token，reset(token) 精确回滚；
#   快照不可变，线程/协程之间不会串状态。
# - 快照 ↔ 栈：push/pop 各生成新快照（末尾追加/截断），旧快照由 token 持有。
# - 栈 ↔ frame：每层独立，查 tangent 只查当前 frame，嵌套 JVP 互不覆盖。
# - frame ↔ primal 图：frame 不持有图；_jvp_rule 常驻 TN，同一张图可多方向复用。
#
# 生命周期
# --------
#   PRIMAL ：执行原函数，记录 primal 图并挂 _jvp_rule。
#   TANGENT：沿拓扑序调用规则，读 frame.tangent(...) 并写 node_tangent。
#   phase 由 set_jvp_phase 推进，并触发计数重建。
#
# 为什么栈用 tuple：token 回滚与上下文复制都要求快照内容不变，详见
# _ForwardADState 的说明；反向模式由 requires_grad + is_grad_enabled() 独立控制，
# 与前向栈并存。
# ---------------------------------------------------------------------------

PRIMAL = 'primal'      # 阶段：原函数前向执行，记录 primal 图
TANGENT = 'tangent'    # 阶段：沿拓扑序传播 tangent


class JVPFrame:
    """一次 JVP 调用（一个切方向）的传播状态，是前向 AD 栈中的一层。

    tangent 数据放在 frame 而不是 TN 上，因此同一张 primal 图可对多个方向
    反复求 JVP；嵌套调用时每层各有一张表，互不覆盖。

    属性:
        node_tangent:    当前方向的 tangent 表：id(node) -> TN | None。
                         None 表示零方向/无贡献；键为 id(node)，只在对应图节点
                         存活期间有效。
        dependency_mask: 当前方向的输入依赖位掩码：id(node) -> int。
                         第 k 位为 1 表示 node 依赖第 k 个输入，仅供 strict 判定。
        前向请求存在期间始终记录计算图，不再有 create_graph 配置项。
        strict:          输出与所有输入都不连通时是否报错。
        phase:           PRIMAL（记录 primal 图）或 TANGENT（传播 tangent）。
    """

    __slots__ = ('node_tangent', 'dependency_mask', 'strict', 'phase')

    def __init__(self, strict: bool = False, phase: str = PRIMAL):
        self.node_tangent = {}
        self.dependency_mask = {}
        self.strict = strict
        self.phase = phase

    def tangent(self, var):
        """返回节点 var 在当前方向上的 tangent；未播种或零方向返回 None。

        适用于任意节点（输入、中间结果或原地步骤的右值），不要求 var 是函数输入。
        非 TN 对象不会出现在 node_tangent 中，id() 在对象存活期间唯一，因此直接
        查表即可返回 None，无需 isinstance 检查。
        """
        return self.node_tangent.get(id(var))


class _ForwardADState:
    """前向 AD 栈的不可变快照：只保存栈的形状。

    它是什么
    --------
    不是整个栈的深拷贝，而是一个轻量“版本对象”：frames 是不可变元组，新快照
    与旧快照共享 frame 对象及其 node_tangent，只按需新建元组本身（push 为
    frames + (frame,)，pop 为 frames[:-1]）。因此快照成本是 O(栈深) 的指针复制，
    而不是 O(传播数据) 的拷贝。

    为什么必须不可变
    ----------------
    1. 可回滚：ContextVar.set() 返回 token，reset(token) 恢复的是“当时那个对象”。
       只有对象内容不变，恢复才等价于回滚栈；若原地 append/pop，token 指向的
       对象已被改写，reset 会静默失效。
    2. 上下文隔离：asyncio 任务/线程池复制上下文时复制的是对象引用。原地修改会
       让一个任务看到另一个任务的栈变化；不可变对象被共享时天然只读、无竞争。
       不会出现中间态被其他上下文观察到。

    反例（若改成可变 list / 可变对象）
    ---------------------------------
    A. 回滚失效：push F1 拿到 token1；再 push F2；finally 中 reset(token1)。
       若 frames 是同一个 list，reset 后内容仍是 (F1, F2)，F2 泄漏。
    B. 协程串状态：主协程 push 后创建子任务，子任务原地 push/pop 会改写同一个
       list，主协程的栈随之变化，两个 JVP 的 node_tangent 互相干扰。

    职责边界
    --------
    创建：push/pop；
    读取：forward_ad_trace_active（判断栈是否为空）与 active_primal_frames；
    不负责：tangent 数据、图结构、frame 内部可变状态。

    不变性的边界
    ------------
    冻结的只有“栈结构”：frames 元组不可变；JVPFrame 本身是可变工作区——phase
    由 set_jvp_phase 原地推进，node_tangent 在传播期持续写入。这样既有快照级的
    安全回滚，又避免热路径反复复制传播数据。

    生命周期示例
    ------------
    S0=((),) → push F1(PRIMAL) 得 S1=((F1,),)
              → 原地切 TANGENT（快照对象不变）
              → reset(token1) 回到 S0；frame 在各快照之间共享。

    字段
    ----
        frames: 按入栈顺序排列的 frame 元组，末尾为栈顶；空元组表示当前没有
                前向请求。
    """

    __slots__ = ('frames',)

    def __init__(self, frames=()):
        self.frames = frames


# 每个线程/协程上下文各自持有当前快照；默认值为空栈。
_forward_ad_state = contextvars.ContextVar(
    'riemann_forward_ad_state', default=_ForwardADState())


def push_jvp_frame(frame: JVPFrame):
    """把 frame 压入栈顶，返回供 pop_jvp_frame 精确恢复的 token。

    追加操作生成新快照，旧快照保持不变并被返回的 token 引用，
    因此后续 pop 可以精确恢复，不受中途再 push 的影响。
    """
    state = _forward_ad_state.get()
    return _forward_ad_state.set(
        _ForwardADState(state.frames + (frame,)))


def pop_jvp_frame(token=None):
    """弹出栈顶 frame。

    传入 push 返回的 token 时，恢复到压栈前的快照：即使之后又压入了其他
    frame 也会一并丢弃，适合 try/finally 场景。
    未传 token 时生成“截掉栈顶”的新快照，仅适用于确定栈顶就是目标 frame 的
    简单场景。
    """
    if token is None:
        state = _forward_ad_state.get()
        _forward_ad_state.set(_ForwardADState(state.frames[:-1]))
    else:
        _forward_ad_state.reset(token)


def set_jvp_phase(frame: JVPFrame, phase: str):
    """切换 frame 阶段（PRIMAL -> TANGENT）。

    唯一原地修改 frame 的入口；栈快照不重建，trace 判定只看栈是否非空。
    """
    frame.phase = phase


def forward_ad_trace_active() -> bool:
    """当前是否存在前向 AD 请求，供算子热路径做 O(1) 判断。

    栈非空即为 True：PRIMAL 阶段记录 primal 图，TANGENT 阶段记录 tangent
    表达式自身的图（使 tangent 可继续求导）。

    算子侧统一用它决定“是否写 fromvars 并挂 _jvp_rule”。
    没有前向请求时只读一个 ContextVar 并检查元组是否为空。
    """
    return len(_forward_ad_state.get().frames) > 0


def active_primal_frames():
    """返回栈中所有 PRIMAL 阶段 frame 的元组（按入栈顺序，最内层在最后）。

    原地算子记录的每个步骤都附带这些 frame 的身份；_propagate_jvp 只重放与
    当前 frame 身份相交的步骤。这样同一张量连续求 jvp（或嵌套 jvp）时，不会
    把上一次调用记录的原地步骤重复施加到本次 tangent 上。
    """
    return tuple(f for f in _forward_ad_state.get().frames if f.phase == PRIMAL)
