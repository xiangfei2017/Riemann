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

"""Riemann core package.

本包聚合 Riemann 框架的核心构建模块：

    - tensordef      : TN 张量类、动态计算图、反向传播引擎、张量创建及核心运算
    - tensorfuncs    : 张量函数式运算（仅单向依赖 tensordef）
    - dtype        : dtype 常量与辅助函数
    - ad_context   : 自动微分上下文（no_grad / enable_grad / set_grad_enabled 等）
    - serialization: 模型与张量的保存 / 加载

外部模块与子包应整体导入本包，而不是单独导入包内文件，例如::

    from riemann.core import TN, tensor, zeros
    from ..core import *          # riemann 子包内部使用

注意：只有 `import *` 能导出的公共名才经由本包重导出；下划线开头的内部辅助函数
（如 _apply_out、_getitem_backward、_parameter_constructor）请直接从定义模块导入，
例如 `from ..core.tensordef import _apply_out`，避免依赖本包的初始化顺序。

导入顺序：tensordef -> tensorfuncs -> serialization。dtype 与 ad_context 由
tensordef 先行导入，并随其 `import *` 传递导出。serialization 依赖 nn.module
（Parameter），必须放在最后，以保证 core.tensordef 先完成加载，避免循环导入。
"""

from .tensordef import *
from .tensorfuncs import *
from .serialization import *

# `import *` 会把子模块依赖的模块对象一并带入本命名空间，
# 这里统一清理，避免 riemann.core.np / riemann.os 之类的意外公共名。
del np, cp, os, sys, builtins, functools, inspect, warnings
del namedtuple, contextmanager, Callable, Any, Optional
