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

    - tensordef    : TN 张量类、动态计算图与自动微分、张量创建及核心运算
    - tensorfuncs  : 张量函数式运算（仅单向依赖 tensordef）
    - dtype        : dtype 常量与辅助函数
    - gradmode     : 梯度开关上下文（no_grad / enable_grad / set_grad_enabled 等）
    - serialization: 模型与张量的保存 / 加载

外部模块与子包应整体导入本包，而不是单独导入包内文件，例如::

    from riemann.core import TN, tensor, zeros
    from ..core import *          # riemann 子包内部使用

注意：导入顺序为 tensordef -> tensorfuncs -> dtype -> gradmode -> serialization。
serialization 依赖 nn.module（Parameter），必须放在最后，以保证
core.tensordef 先完成加载，避免循环导入。
"""

from .tensordef import *
from .tensorfuncs import *
# from .dtype import *
# from .gradmode import *

# tensordef 的内部辅助函数必须在 serialization 之前重导出：
# serialization -> nn.module -> nn.kan -> linalg 的导入链会在本包尚未完成
# 初始化时通过 `from ..core import _apply_out` 等取用这些名字
from .tensordef import _batch_slice, _apply_out, _apply_out_tuple

from .serialization import *
from .serialization import _parameter_constructor
