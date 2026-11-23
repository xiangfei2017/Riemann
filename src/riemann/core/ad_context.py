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
# 与反向梯度模式（threading.local 的 _grad_mode_stack）并存，互不覆盖：
#   - 反向：requires_grad + is_grad_enabled() 决定是否建反向图；
#   - 前向：本模块的 frame 栈决定是否记录 primal 图并传播 tangent。
# 使用 ContextVar 而非 threading.local，可同时满足线程与 asyncio 任务隔离，
# 并支持 token 式压栈/出栈恢复（异常路径可安全回滚）。
# ---------------------------------------------------------------------------

PRIMAL = 'primal'      # 阶段：原函数前向执行，记录 primal 图
TANGENT = 'tangent'    # 阶段：沿拓扑序传播 tangent


class JVPFrame:
    """一次 JVP 调用独立的切向传播状态。

    状态不写入 TN 的持久字段，因此同一张 primal 图可对不同方向反复求 JVP，
    嵌套调用也不会相互覆盖（见设计文档 §11/§12）。

    属性:
        seed_tangents:   id(primal) -> 输入 tangent，仅作调用期参考/调试
        node_tangent:    本次调用的中间 tangent：id(node) -> TN 或 None
        dependency_mask: id(node) -> 依赖位掩码，供 strict 判定
        create_graph:    是否保留 tangent 规则产生的图（高阶）
        strict:          输出与输入完全不连通时是否报错
        phase:           PRIMAL 或 TANGENT
    """

    __slots__ = ('seed_tangents', 'node_tangent', 'dependency_mask',
                 'create_graph', 'strict', 'phase')

    def __init__(self, seed_tangents=None, create_graph: bool = False,
                 strict: bool = False, phase: str = PRIMAL):
        self.seed_tangents = {} if seed_tangents is None else seed_tangents
        self.node_tangent = {}
        self.dependency_mask = {}
        self.create_graph = create_graph
        self.strict = strict
        self.phase = phase


class _ForwardADState:
    """ContextVar 中保存的前向 AD 状态快照（不可变，整对象替换）。"""

    __slots__ = ('frames', 'primal_count', 'tangent_keep_count')

    def __init__(self, frames=(), primal_count: int = 0,
                 tangent_keep_count: int = 0):
        self.frames = frames
        self.primal_count = primal_count
        self.tangent_keep_count = tangent_keep_count


def _make_forward_ad_state(frames):
    """由 frame 元组重算两个 O(1) 计数（帧数量很小，非热路径）。"""
    primal_count = 0
    tangent_keep_count = 0
    for frame in frames:
        if frame.phase == PRIMAL:
            primal_count += 1
        elif frame.create_graph:
            tangent_keep_count += 1
    return _ForwardADState(frames, primal_count, tangent_keep_count)


_forward_ad_state = contextvars.ContextVar(
    'riemann_forward_ad_state', default=_ForwardADState())


def push_jvp_frame(frame: JVPFrame):
    """把 frame 压入前向 AD 栈，返回供 pop_jvp_frame 使用的 token。"""
    state = _forward_ad_state.get()
    return _forward_ad_state.set(
        _make_forward_ad_state(state.frames + (frame,)))


def pop_jvp_frame(token=None):
    """弹出前向 AD frame。

    token 由 push_jvp_frame 返回时按 token 精确恢复（可安全用于 try/finally）；
    未提供 token 时直接弹出栈顶。
    """
    if token is None:
        state = _forward_ad_state.get()
        _forward_ad_state.set(_make_forward_ad_state(state.frames[:-1]))
    else:
        _forward_ad_state.reset(token)


def current_jvp_frame():
    """返回当前栈顶的 JVPFrame；未处于前向 AD 上下文时返回 None。"""
    frames = _forward_ad_state.get().frames
    return frames[-1] if frames else None


def set_jvp_phase(frame: JVPFrame, phase: str):
    """切换 frame 阶段并刷新 O(1) 计数（PRIMAL -> TANGENT）。"""
    frame.phase = phase
    state = _forward_ad_state.get()
    _forward_ad_state.set(_make_forward_ad_state(state.frames))


def forward_ad_trace_active() -> bool:
    """O(1) 判定当前是否需要记录前向图（算子热路径只读一个计数）。

    满足任一条件即为 True：
      - 存在 PRIMAL 阶段的 frame（正在为某次 JVP 记录 primal 图）；
      - 栈顶 frame 已进入 TANGENT 且 create_graph=True（保留 tangent 图）。
    没有前向请求时只多一次属性读取，不触碰 frame 内容。
    """
    state = _forward_ad_state.get()
    return state.primal_count > 0 or state.tangent_keep_count > 0


def active_primal_frames():
    """返回当前栈中所有 PRIMAL 阶段的 frame（最内层在最后）。

    原地算子据此给本次记录的前向步骤打标签：只有标签与本 frame 有交集的
    步骤才会在 `_propagate_jvp` 中生效，避免复用同一张量连续调用 jvp 时
    把上一次调用的原地步骤重复施加到本次 tangent 上。
    """
    return tuple(f for f in _forward_ad_state.get().frames if f.phase == PRIMAL)


def outer_ad_trace_active() -> bool:
    """排除栈顶 frame 后，是否仍有 PRIMAL 阶段的外层 frame。

    用于“内层 create_graph=False 不能切断外层高阶图”的判定（§14.3）。
    """
    state = _forward_ad_state.get()
    count = state.primal_count
    frames = state.frames
    if frames and frames[-1].phase == PRIMAL:
        count -= 1
    return count > 0
