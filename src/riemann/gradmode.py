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
Riemann Library Gradient Mode Module: Automatic Differentiation Control

This module provides gradient calculation mode control functionality for the Riemann
automatic differentiation framework. It implements context managers and decorators
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
    if not hasattr(_grad_mode_stack, 'stack'):
        _grad_mode_stack.stack = [True]  # 默认为启用梯度计算


def is_grad_enabled():
    """
    获取当前线程的梯度计算状态
    
    返回栈顶元素，表示当前的梯度计算模式（True 表示启用，False 表示禁用）。
    如果梯度模式栈尚未初始化，则会自动调用 init_grad_mode_stack() 进行初始化。
    
    返回：
        bool: 当前梯度计算模式，True 表示启用，False 表示禁用
    """
    if not hasattr(_grad_mode_stack, 'stack'):
        init_grad_mode_stack()  # 确保栈已初始化
    return _grad_mode_stack.stack[-1]


def _push_grad_mode(mode):
    """
    将新的梯度计算模式压入栈
    
    内部辅助函数，用于在进入梯度控制上下文时保存新的梯度计算模式。
    如果梯度模式栈尚未初始化，则会自动进行初始化。
    
    参数：
        mode (bool): 新的梯度计算模式，True 表示启用，False 表示禁用
    """
    if not hasattr(_grad_mode_stack, 'stack'):
        init_grad_mode_stack()
    _grad_mode_stack.stack.append(mode)


def _pop_grad_mode():
    """
    弹出栈顶的梯度计算模式
    
    内部辅助函数，用于在退出梯度控制上下文时恢复之前的梯度计算模式。
    只有当栈中元素数量大于 1 时才会弹出，确保始终保留一个默认模式。
    
    返回：
        bool 或 None: 被弹出的梯度计算模式，如果无法弹出则返回 None
    """
    if hasattr(_grad_mode_stack, 'stack') and len(_grad_mode_stack.stack) > 1:
        return _grad_mode_stack.stack.pop()


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
    class _NoGradContext:
        """禁用梯度计算的上下文管理器"""
        
        def __enter__(self):
            """进入上下文，保存当前梯度状态并禁用梯度"""
            self.prev = is_grad_enabled()
            _push_grad_mode(False)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            """退出上下文，恢复之前的梯度状态"""
            _pop_grad_mode()
            return False
    
    # 如果func为None，返回上下文管理器实例
    if func is None:
        return _NoGradContext()
    
    # 否则，作为装饰器使用
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _NoGradContext():
            return func(*args, **kwargs)
    return wrapper


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
    class _EnableGradContext:
        """启用梯度计算的上下文管理器"""
        
        def __enter__(self):
            """进入上下文，保存当前梯度状态并启用梯度"""
            self.prev = is_grad_enabled()
            _push_grad_mode(True)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            """退出上下文，恢复之前的梯度状态"""
            _pop_grad_mode()
            return False
    
    # 如果func为None，返回上下文管理器实例
    if func is None:
        return _EnableGradContext()
    
    # 否则，作为装饰器使用
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _EnableGradContext():
            return func(*args, **kwargs)
    return wrapper


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
    class _SetGradContext:
        """
        可配置的梯度计算上下文管理器
        
        支持通过 mode 参数控制是否启用梯度计算，
        同时实现了 __call__ 方法以支持装饰器语法。
        """
        
        def __init__(self, mode_val):
            """
            初始化上下文管理器
            
            参数：
                mode_val (bool): 梯度计算模式，True 启用，False 禁用
            """
            self.mode = mode_val
            
        def __enter__(self):
            """
            进入上下文，保存当前梯度状态并应用新模式
            
            返回：
                self: 上下文管理器实例
            """
            self.prev = is_grad_enabled()
            _push_grad_mode(self.mode)
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
            _pop_grad_mode()
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
                with _SetGradContext(self.mode):
                    return func_to_wrap(*args, **kwargs)
            return wrapper
    
    # 如果func为None，返回上下文管理器实例
    # 这个实例现在同时支持上下文管理器和装饰器语法
    if func is None:
        return _SetGradContext(mode)
    
    # 如果提供了func参数，直接返回包装后的函数
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _SetGradContext(mode):
            return func(*args, **kwargs)
    return wrapper
