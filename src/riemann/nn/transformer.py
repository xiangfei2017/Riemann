# BSD 3-Clause License
#
# Copyright (c) 2025, Fei Xiang
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
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
Transformer Module for the Riemann Library

This module provides Transformer-related components for the Riemann library,
including MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer,
TransformerEncoder, TransformerDecoder, and Transformer. All components are designed to be
compatible with PyTorch's interface and behavior.

Implemented Classes:
- MultiheadAttention: Multi-head attention mechanism, compatible with torch.nn.MultiheadAttention
- TransformerEncoderLayer: Single layer of Transformer encoder
- TransformerDecoderLayer: Single layer of Transformer decoder
- TransformerEncoder: Stack of N TransformerEncoderLayer layers
- TransformerDecoder: Stack of N TransformerDecoderLayer layers
- Transformer: Complete Transformer model with encoder and decoder
"""

import math
import copy
from ..tensordef import *
from .module import Module, Parameter, Linear, Dropout, LayerNorm, ModuleList
from .activation import ReLU, GELU
from .functional import linear, softmax, dropout


class MultiheadAttention(Module):
    """
    多头注意力机制 (Multi-head Attention Mechanism)
    
    实现多头注意力机制，允许模型同时关注来自不同表示子空间的信息。
    接口和行为与 torch.nn.MultiheadAttention 完全一致。
    
    内部结构:
        - in_proj: Q/K/V 输入线性投影
        - 注意力权重计算
        - 应用注意力掩码
        - dropout
        - 计算输出的上下文向量
        - 多头合并
        - out_proj: 输出投影线性投影
        
    前向计算流程:
        1. 将 Query、Key、Value 通过输入投影层得到 q、k、v
        2. 可选：添加 bias_k/bias_v（当 add_bias_kv=True）
        3. 可选：添加零注意力（当 add_zero_attn=True）
        4. 将 q、k、v 分头 reshape: (seq_len, bsz, embed_dim) -> (bsz*num_heads, seq_len, head_dim)
        5. 计算注意力分数：attn_scores = q @ k.T / sqrt(head_dim)
        6. 应用因果掩码（当 is_causal=True）
        7. 应用注意力掩码 attn_mask（如果提供，支持2D/3D）
        8. 应用 key_padding_mask（如果提供，支持bool/float）
        9. 计算 softmax 得到注意力权重，应用 dropout
        10. 计算输出：output = attn_weights @ v
        11. 多头合并: (bsz*num_heads, tgt_len, head_dim) -> (tgt_len, bsz, embed_dim)
        12. 通过输出投影层 out_proj
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device=None,
        dtype=None
    ):
        """
        初始化 MultiheadAttention 模块
        
        Args:
            embed_dim (int): 输入和输出向量的维度大小，必须能被 num_heads 整除
            num_heads (int): 多头注意力中使用的头部数量，每个头部负责不同的表示子空间
            dropout (float, optional): 训练过程中对注意力权重应用的 dropout 概率，默认为 0.0
            bias (bool, optional): 是否在投影层中添加偏置项，默认为 True
            add_bias_kv (bool, optional): 是否在 key 和 value 序列的末尾添加可学习的偏置项，默认为 False
            add_zero_attn (bool, optional): 是否在注意力权重中添加一列零，默认为 False
            kdim (int, optional): key 向量的维度，默认为 None（使用 embed_dim）
            vdim (int, optional): value 向量的维度，默认为 None（使用 embed_dim）
            batch_first (bool, optional): 输入输出的形状格式，默认为 False（seq_len, batch_size, embed_dim）
            device: 张量设备，默认为 None
            dtype: 张量数据类型，默认为 None
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first
        
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, "
                f"got embed_dim={self.embed_dim}, num_heads={self.num_heads}"
            )
        
        self.head_dim = self.embed_dim // self.num_heads
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self._qkv_same_embed_dim = self.kdim == self.embed_dim and self.vdim == self.embed_dim
        
        if self._qkv_same_embed_dim:
            self.in_proj_weight = Parameter(
                empty((3 * self.embed_dim, self.embed_dim), **factory_kwargs)
            )
            if self.bias:
                self.in_proj_bias = Parameter(empty((3 * self.embed_dim,), **factory_kwargs))
            else:
                self.register_parameter('in_proj_bias', None)
        else:
            self.q_proj_weight = Parameter(
                empty((self.embed_dim, self.embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = Parameter(
                empty((self.embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = Parameter(
                empty((self.embed_dim, self.vdim), **factory_kwargs)
            )
            if self.bias:
                self.q_proj_bias = Parameter(empty((self.embed_dim,), **factory_kwargs))
                self.k_proj_bias = Parameter(empty((self.embed_dim,), **factory_kwargs))
                self.v_proj_bias = Parameter(empty((self.embed_dim,), **factory_kwargs))
            else:
                self.register_parameter('q_proj_bias', None)
                self.register_parameter('k_proj_bias', None)
                self.register_parameter('v_proj_bias', None)
        
        self.out_proj = Linear(self.embed_dim, self.embed_dim, bias=self.bias, **factory_kwargs)
        
        if self.add_bias_kv:
            self.bias_k = Parameter(empty((1, 1, self.embed_dim), **factory_kwargs))
            self.bias_v = Parameter(empty((1, 1, self.embed_dim), **factory_kwargs))
        else:
            self.register_parameter('bias_k', None)
            self.register_parameter('bias_v', None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        重置模型参数 (Reset Model Parameters)
        
        使用均匀分布初始化权重参数，偏置参数初始化为 0。
        初始化范围为 [-sqrt(1/embed_dim), sqrt(1/embed_dim)]。
        """
        bound = math.sqrt(1.0 / self.embed_dim)
        if self._qkv_same_embed_dim:
            arrlib = self.in_proj_weight._get_array_lib()
            with device_context(self.in_proj_weight):
                self.in_proj_weight.data[:] = arrlib.random.uniform(-bound, bound, self.in_proj_weight.shape)
            if self.in_proj_bias is not None:
                self.in_proj_bias.data.fill(0.)
        else:
            # q、k、v 权重都在同一设备上，只需获取一次 arrlib 和一个上下文
            arrlib = self.q_proj_weight._get_array_lib()
            with device_context(self.q_proj_weight):
                self.q_proj_weight.data[:] = arrlib.random.uniform(-bound, bound, self.q_proj_weight.shape)
                self.k_proj_weight.data[:] = arrlib.random.uniform(-bound, bound, self.k_proj_weight.shape)
                self.v_proj_weight.data[:] = arrlib.random.uniform(-bound, bound, self.v_proj_weight.shape)
            if self.q_proj_bias is not None:
                self.q_proj_bias.data.fill(0.)
                self.k_proj_bias.data.fill(0.)
                self.v_proj_bias.data.fill(0.)
        
        if self.bias_k is not None:
            self.bias_k.data.fill(0.)
        if self.bias_v is not None:
            self.bias_v.data.fill(0.)
    
    def forward(
        self,
        query: TN,
        key: TN,
        value: TN,
        key_padding_mask: TN | None = None,
        need_weights: bool = True,
        attn_mask: TN | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False
    ) -> tuple[TN, TN | None]:
        """
        多头注意力前向传播 (Multi-head Attention Forward Pass)
        
        Args:
            query (TN): 查询（Query）张量，用于确定关注哪些位置。
                形状: (seq_len, batch_size, embed_dim) 或 (batch_size, seq_len, embed_dim)（取决于 batch_first）
            
            key (TN): 键（Key）张量，用于与 query 计算相似度。
                形状: (key_len, batch_size, kdim) 或 (batch_size, key_len, kdim)
            
            value (TN): 值（Value）张量，包含实际要聚合的信息。
                形状: (key_len, batch_size, vdim) 或 (batch_size, key_len, vdim)
            
            key_padding_mask (TN, optional): Key 序列的填充掩码，用于忽略填充的 token。
                形状: (batch_size, key_len)
                填充位置为 True，有效位置为 False，默认为 None
                使用场景: 处理变长序列时，忽略 padding 位置
                
            need_weights (bool, optional): 是否返回注意力权重，默认为 True。
                设为 False 可以节省计算和内存
                
            attn_mask (TN, optional): 注意力掩码，用于控制注意力模式。
                支持两种形状（与PyTorch一致）:
                - 2D: (tgt_len, src_len) - 对所有批次和头部使用相同的掩码
                - 3D: (batch_size * num_heads, tgt_len, src_len) - 对不同批次/头部使用不同掩码
                使用场景: 因果掩码、自定义注意力模式等
                默认为 None
                
            average_attn_weights (bool, optional): 是否对多头的注意力权重进行平均，默认为 True。
                - True: 返回 (batch_size, seq_len, key_len)，各头平均
                - False: 返回 (batch_size, num_heads, seq_len, key_len)，保留各头信息
                
            is_causal (bool, optional): 是否使用因果掩码，用于自回归任务。
                启用后，每个位置只能关注当前和之前的位置（上三角被 mask）。
                使用场景: GPT 等自回归语言模型
                如果同时提供 attn_mask，两者会叠加使用
                默认为 False
        
        Returns:
            tuple[TN, TN | None]:
                - attn_output: 注意力输出，形状与 query 相同
                - attn_output_weights: 注意力权重，形状取决于 average_attn_weights 和 need_weights
                    如果 need_weights=False，则返回 None
        """
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        
        assert embed_dim == self.embed_dim
        assert key.shape == (src_len, bsz, self.kdim)
        assert value.shape == (src_len, bsz, self.vdim)
        
        q, k, v = self._in_proj_qkv(query, key, value)
        
        if self.add_bias_kv:
            k = concatenate([k, self.bias_k.expand(1, bsz, self.embed_dim)], dim=0)
            v = concatenate([v, self.bias_v.expand(1, bsz, self.embed_dim)], dim=0)
            if key_padding_mask is not None:
                key_padding_mask = concatenate(
                    [key_padding_mask, zeros((bsz, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)],
                    dim=1
                )
            src_len += 1  # 更新src_len
        
        if self.add_zero_attn:
            k = concatenate([k, zeros((1, bsz, self.embed_dim), dtype=k.dtype, device=k.device)], dim=0)
            v = concatenate([v, zeros((1, bsz, self.embed_dim), dtype=v.dtype, device=v.device)], dim=0)
            if key_padding_mask is not None:
                key_padding_mask = concatenate(
                    [key_padding_mask, zeros((bsz, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)],
                    dim=1
                )
            src_len += 1  # 更新src_len
        
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # 计算注意力分数
        attn_scores = q @ k.mT / math.sqrt(self.head_dim)
        
        # 应用因果掩码：直接在attn_scores上操作，避免创建完整掩码矩阵
        if is_causal:
            attn_scores = attn_scores.view(bsz, self.num_heads, tgt_len, src_len)
            # 创建因果掩码：上三角（不包括对角线）为True，需要被mask
            causal_mask = ones((tgt_len, src_len), dtype=bool, device=attn_scores.device).triu(diagonal=1)
            attn_scores.masked_fill_(causal_mask, float('-inf'))
            attn_scores = attn_scores.view(bsz * self.num_heads, tgt_len, src_len)
        
        # 处理注意力掩码（与因果掩码叠加）
        # 只支持2D或3D mask，与PyTorch保持一致
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                # 2D mask: (tgt_len, src_len) -> (1, tgt_len, src_len)
                attn_scores += attn_mask.unsqueeze(0)
            elif attn_mask.ndim == 3:
                # 3D mask: (bsz * num_heads, tgt_len, src_len)
                attn_scores += attn_mask.view(bsz * self.num_heads, tgt_len, src_len)
            else:
                raise ValueError(f"attn_mask must be 2D or 3D, got {attn_mask.ndim}D")
        
        if key_padding_mask is not None:
            # 支持bool或float类型的key_padding_mask
            attn_scores = attn_scores.view(bsz, self.num_heads, tgt_len, src_len)
            if key_padding_mask.dtype == bool:
                # bool类型：使用masked_fill_
                attn_scores.masked_fill_(
                    key_padding_mask.unsqueeze((1, 2)),
                    float('-inf')
                )
            else:
                # float类型：直接加到attn_scores上（与attn_mask处理方式一致）
                attn_scores += key_padding_mask.unsqueeze((1, 2))
            attn_scores = attn_scores.view(bsz * self.num_heads, tgt_len, src_len)
        
        if need_weights:
            # 需要返回权重时，计算完整的注意力权重矩阵并保存
            attn_weights = softmax(attn_scores, dim=-1)
            attn_weights = dropout(attn_weights, p=self.dropout, training=self.training)
            
            attn_output = attn_weights @ v
            
            # 处理返回的权重形状
            attn_output_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
        else:
            # 不需要权重时，直接计算输出，不保存中间权重矩阵以节省内存
            attn_weights = softmax(attn_scores, dim=-1)
            attn_weights = dropout(attn_weights, p=self.dropout, training=self.training)
            
            attn_output = attn_weights @ v
            attn_output_weights = None
        
        attn_output = attn_output.transpose(0, 1).view(tgt_len, bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        return attn_output, attn_output_weights
    
    def _in_proj_qkv(self, query, key, value):
        """
        对 Query、Key、Value 进行线性投影 (Linear Projection for Q, K, V)
        
        当 query=key=value 且 _qkv_same_embed_dim=True 时，使用单次矩阵乘法优化，
        然后通过切片得到 q、k、v，提高计算效率。
        
        Args:
            query (TN): 查询张量
            key (TN): 键张量
            value (TN): 值张量
            
        Returns:
            tuple[TN, TN, TN]: 投影后的 q、k、v 张量
        """
        if self._qkv_same_embed_dim and query is key and key is value:
            projected = linear(query, self.in_proj_weight, self.in_proj_bias)
            q = projected[..., :self.embed_dim]
            k = projected[..., self.embed_dim:2*self.embed_dim]
            v = projected[..., 2*self.embed_dim:]
            return q, k, v
        
        if self._qkv_same_embed_dim:
            q_proj_weight = self.in_proj_weight[:self.embed_dim, :]
            k_proj_weight = self.in_proj_weight[self.embed_dim:2*self.embed_dim, :]
            v_proj_weight = self.in_proj_weight[2*self.embed_dim:, :]
            
            if self.in_proj_bias is not None:
                q_proj_bias = self.in_proj_bias[:self.embed_dim]
                k_proj_bias = self.in_proj_bias[self.embed_dim:2*self.embed_dim]
                v_proj_bias = self.in_proj_bias[2*self.embed_dim:]
            else:
                q_proj_bias = k_proj_bias = v_proj_bias = None
        else:
            q_proj_weight = self.q_proj_weight
            k_proj_weight = self.k_proj_weight
            v_proj_weight = self.v_proj_weight
            
            if self.q_proj_bias is not None:
                q_proj_bias = self.q_proj_bias
                k_proj_bias = self.k_proj_bias
                v_proj_bias = self.v_proj_bias
            else:
                q_proj_bias = k_proj_bias = v_proj_bias = None
        
        q = linear(query, q_proj_weight, q_proj_bias)
        k = linear(key, k_proj_weight, k_proj_bias)
        v = linear(value, v_proj_weight, v_proj_bias)
        
        return q, k, v


class TransformerEncoderLayer(Module):
    """
    Transformer 编码器层 (Transformer Encoder Layer)
    
    实现 Transformer 编码器的单个层，由自注意力机制和前馈网络组成。
    
    内部结构:
        - self_attn: 多头自注意力层 (MultiheadAttention)
        - linear1: 前馈网络第一层线性变换 (d_model -> dim_feedforward)
        - activation_fn: 激活函数 (ReLU 或 GELU)
        - linear2: 前馈网络第二层线性变换 (dim_feedforward -> d_model)
        - norm1: 第一层的层归一化 (LayerNorm)
        - norm2: 第二层的层归一化 (LayerNorm)
        - dropout1: 自注意力输出的 dropout
        - dropout2: 前馈网络第一层的 dropout
        - dropout3: 前馈网络第二层的 dropout
        
    前向计算流程（两种模式）:
        Post-LN 模式（原始 Transformer 论文）:
            1. 自注意力 -> dropout1 -> 残差连接 -> norm1
            2. 前馈网络 (linear1 -> activation -> dropout2 -> linear2) -> dropout3 -> 残差连接 -> norm2
            
        Pre-LN 模式（更稳定的训练）:
            1. norm1 -> 自注意力 -> dropout1 -> 残差连接
            2. norm2 -> 前馈网络 (linear1 -> activation -> dropout2 -> linear2) -> dropout3 -> 残差连接
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        """
        初始化 TransformerEncoderLayer 模块
        
        Args:
            d_model (int): 输入和输出特征的维度大小
            nhead (int): 多头注意力中使用的头部数量
            dim_feedforward (int, optional): 前馈网络中隐藏层的维度大小，默认为 2048
            dropout (float, optional): 训练过程中对各层输出应用的 dropout 概率，默认为 0.1
            activation (str, optional): 前馈网络中使用的激活函数类型，'relu' 或 'gelu'，默认为 'relu'
            layer_norm_eps (float, optional): 层归一化中使用的 epsilon 值，默认为 1e-05
            batch_first (bool, optional): 输入输出的形状格式，默认为 False（seq_len, batch_size, d_model）
            norm_first (bool, optional): 是否使用 Pre-LN 模式，默认为 False（Post-LN 模式）
            bias (bool, optional): 是否在所有线性层中添加偏置项，默认为 True
            device: 张量设备，默认为 None
            dtype: 张量数据类型，默认为 None
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.bias = bias
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias,
            batch_first=batch_first, **factory_kwargs
        )
        
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        if activation == "relu":
            self.activation_fn = ReLU()
        elif activation == "gelu":
            self.activation_fn = GELU()
        else:
            raise ValueError(f"activation should be 'relu' or 'gelu', got {activation}")
    
    def forward(
        self,
        src: TN,
        src_mask: TN | None = None,
        src_key_padding_mask: TN | None = None,
        is_causal: bool = False
    ) -> TN:
        """
        Transformer 编码器层前向传播 (Transformer Encoder Layer Forward Pass)
        
        Args:
            src (TN): 编码器输入序列，形状为 (seq_len, batch_size, d_model) 
                或 (batch_size, seq_len, d_model)（取决于 batch_first）
            src_mask (TN, optional): 自注意力掩码，形状为 (seq_len, seq_len) 
                或 (batch_size, nhead, seq_len, seq_len)，默认为 None
            src_key_padding_mask (TN, optional): key 序列的填充掩码，
                形状为 (batch_size, seq_len)，默认为 None
            is_causal (bool, optional): 是否使用因果掩码，默认为 False
        
        Returns:
            TN: 编码器层输出，形状与 src 相同
        """
        if self.norm_first:
            src2 = self.norm1(src)
            src2, _ = self.self_attn(
                src2, src2, src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                need_weights=False
            )
            src = src + self.dropout1(src2)
            
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout2(self.activation_fn(self.linear1(src2))))
            src = src + self.dropout3(src2)
        else:
            src2, _ = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                need_weights=False
            )
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            src2 = self.linear2(self.dropout2(self.activation_fn(self.linear1(src))))
            src = src + self.dropout3(src2)
            src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(Module):
    """
    Transformer 解码器层 (Transformer Decoder Layer)
    
    实现 Transformer 解码器的单个层，由自注意力、交叉注意力和前馈网络组成。
    接口和行为与 torch.nn.TransformerDecoderLayer 完全一致。
    
    内部结构:
        - self_attn: 多头自注意力层 (MultiheadAttention)，用于目标序列内部的注意力
        - multihead_attn: 多头交叉注意力层 (MultiheadAttention)，连接解码器和编码器
        - linear1: 前馈网络第一层线性变换 (d_model -> dim_feedforward)
        - activation_fn: 激活函数 (ReLU 或 GELU)
        - linear2: 前馈网络第二层线性变换 (dim_feedforward -> d_model)
        - norm1: 自注意力的层归一化 (LayerNorm)
        - norm2: 交叉注意力的层归一化 (LayerNorm)
        - norm3: 前馈网络的层归一化 (LayerNorm)
        - dropout1: 自注意力输出的 dropout
        - dropout2: 交叉注意力输出的 dropout
        - dropout3: 前馈网络第一层的 dropout
        - dropout4: 前馈网络第二层的 dropout
        
    前向计算流程（两种模式）:
        Post-LN 模式（原始 Transformer 论文）:
            1. 自注意力 (tgt) -> dropout1 -> 残差连接 -> norm1
            2. 交叉注意力 (tgt, memory, memory) -> dropout2 -> 残差连接 -> norm2
            3. 前馈网络 (linear1 -> activation -> dropout3 -> linear2) -> dropout4 -> 残差连接 -> norm3
            
        Pre-LN 模式（更稳定的训练）:
            1. norm1 -> 自注意力 (tgt) -> dropout1 -> 残差连接
            2. norm2 -> 交叉注意力 (tgt, memory, memory) -> dropout2 -> 残差连接
            3. norm3 -> 前馈网络 (linear1 -> activation -> dropout3 -> linear2) -> dropout4 -> 残差连接
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        """
        初始化 TransformerDecoderLayer 模块
        
        Args:
            d_model (int): 输入和输出特征的维度大小
            nhead (int): 多头注意力中使用的头部数量
            dim_feedforward (int, optional): 前馈网络中隐藏层的维度大小，默认为 2048
            dropout (float, optional): 训练过程中对各层输出应用的 dropout 概率，默认为 0.1
            activation (str, optional): 前馈网络中使用的激活函数类型，'relu' 或 'gelu'，默认为 'relu'
            layer_norm_eps (float, optional): 层归一化中使用的 epsilon 值，默认为 1e-05
            batch_first (bool, optional): 输入输出的形状格式，默认为 False（seq_len, batch_size, d_model）
            norm_first (bool, optional): 是否使用 Pre-LN 模式，默认为 False（Post-LN 模式）
            bias (bool, optional): 是否在所有线性层中添加偏置项，默认为 True
            device: 张量设备，默认为 None
            dtype: 张量数据类型，默认为 None
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.bias = bias
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias,
            batch_first=batch_first, **factory_kwargs
        )
        
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias,
            batch_first=batch_first, **factory_kwargs
        )
        
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)
        
        if activation == "relu":
            self.activation_fn = ReLU()
        elif activation == "gelu":
            self.activation_fn = GELU()
        else:
            raise ValueError(f"activation should be 'relu' or 'gelu', got {activation}")
    
    def forward(
        self,
        tgt: TN,
        memory: TN,
        tgt_mask: TN | None = None,
        memory_mask: TN | None = None,
        tgt_key_padding_mask: TN | None = None,
        memory_key_padding_mask: TN | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False
    ) -> TN:
        """
        Transformer 解码器层前向传播 (Transformer Decoder Layer Forward Pass)
        
        Args:
            tgt (TN): 解码器输入序列（目标序列），
                形状为 (tgt_len, batch_size, d_model) 或 (batch_size, tgt_len, d_model)
            memory (TN): 编码器输出（记忆），
                形状为 (src_len, batch_size, d_model) 或 (batch_size, src_len, d_model)
            tgt_mask (TN, optional): 自注意力掩码，形状为 (tgt_len, tgt_len)，
                默认为 None
            memory_mask (TN, optional): 交叉注意力掩码，形状为 (tgt_len, src_len)，
                默认为 None
            tgt_key_padding_mask (TN, optional): 目标序列的填充掩码，
                形状为 (batch_size, tgt_len)，默认为 None
            memory_key_padding_mask (TN, optional): 编码器输出的填充掩码，
                形状为 (batch_size, src_len)，默认为 None
            tgt_is_causal (bool, optional): 是否对自注意力使用因果掩码（自回归模式），
                默认为 False
            memory_is_causal (bool, optional): 是否对交叉注意力使用因果掩码，
                默认为 False
        
        Returns:
            TN: 解码器层输出，形状与 tgt 相同
        """
        if self.norm_first:
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.self_attn(
                tgt2, tgt2, tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=tgt_is_causal,
                need_weights=False
            )
            tgt = tgt + self.dropout1(tgt2)
            
            tgt2 = self.norm2(tgt)
            tgt2, _ = self.multihead_attn(
                tgt2, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False
            )
            tgt = tgt + self.dropout2(tgt2)
            
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout3(self.activation_fn(self.linear1(tgt2))))
            tgt = tgt + self.dropout4(tgt2)
        else:
            tgt2, _ = self.self_attn(
                tgt, tgt, tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=tgt_is_causal,
                need_weights=False
            )
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            
            tgt2, _ = self.multihead_attn(
                tgt, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False
            )
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            
            tgt2 = self.linear2(self.dropout3(self.activation_fn(self.linear1(tgt))))
            tgt = tgt + self.dropout4(tgt2)
            tgt = self.norm3(tgt)
        
        return tgt


def _get_clones(module, N):
    """
    克隆模块 (Clone Module)
    
    创建一个包含 N 个模块克隆的 ModuleList。
    
    Args:
        module: 需要克隆的模块
        N: 克隆的数量
    
    Returns:
        ModuleList: 包含 N 个克隆模块的 ModuleList
    """
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(Module):
    """
    Transformer 编码器 (Transformer Encoder)
    
    由 N 个 TransformerEncoderLayer 层堆叠而成。
    接口和行为与 torch.nn.TransformerEncoder 完全一致。
    
    内部结构:
        - layers: ModuleList，包含 num_layers 个 TransformerEncoderLayer 实例
        - num_layers: 编码器层的数量
        - norm: 可选的最终层归一化 (LayerNorm)
        - enable_nested_tensor: 是否启用嵌套张量优化（Riemann 暂不支持，仅接口兼容）
        - use_nested_tensor: 是否实际使用嵌套张量（Riemann 暂不支持）
        - mask_check: 是否进行掩码检查（Riemann 暂不支持，仅接口兼容）
        
    前向计算流程:
        1. 将输入 src 依次通过所有 layers（编码器层）
           每个层：output = layer(output, src_mask, src_key_padding_mask, is_causal)
        2. 如果提供了 norm，则对最终输出进行层归一化
    """
    
    def __init__(
        self,
        encoder_layer: 'TransformerEncoderLayer',
        num_layers: int,
        norm: Module | None = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True
    ):
        """
        初始化 TransformerEncoder 模块
        
        Args:
            encoder_layer (TransformerEncoderLayer): 单个编码器层实例，将被克隆 num_layers 次
            num_layers (int): 编码器层的数量
            norm (Module, optional): 最后的层归一化层，默认为 None
            enable_nested_tensor (bool, optional): 是否启用嵌套张量优化（Riemann 暂不支持，仅接口兼容），默认为 True
            mask_check (bool, optional): 是否进行掩码检查（Riemann 暂不支持，仅接口兼容），默认为 True
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        
        why_not_sparsity_fast_path = ""
        if enable_nested_tensor and why_not_sparsity_fast_path:
            self.use_nested_tensor = False
    
    def forward(
        self,
        src: TN,
        mask: TN | None = None,
        src_key_padding_mask: TN | None = None,
        is_causal: bool | None = None
    ) -> TN:
        """
        Transformer 编码器前向传播 (Transformer Encoder Forward Pass)
        
        Args:
            src (TN): 编码器输入序列，形状为 (seq_len, batch_size, d_model)
                或 (batch_size, seq_len, d_model)（取决于 batch_first）
            mask (TN, optional): 源序列的注意力掩码，默认为 None
            src_key_padding_mask (TN, optional): 源序列的填充掩码，默认为 None
            is_causal (bool, optional): 是否使用因果掩码，默认为 None
        
        Returns:
            TN: 编码器输出，形状与 src 相同
        """
        output = src
        
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal if is_causal is not None else False
            )
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


class TransformerDecoder(Module):
    """
    Transformer 解码器 (Transformer Decoder)
    
    由 N 个 TransformerDecoderLayer 层堆叠而成。
    接口和行为与 torch.nn.TransformerDecoder 完全一致。
    
    内部结构:
        - layers: ModuleList，包含 num_layers 个 TransformerDecoderLayer 实例
        - num_layers: 解码器层的数量
        - norm: 可选的最终层归一化 (LayerNorm)
        
    前向计算流程:
        1. 将输入 tgt 依次通过所有 layers（解码器层）
           每个层：
               output = layer(
                   output, memory,
                   tgt_mask, memory_mask,
                   tgt_key_padding_mask, memory_key_padding_mask,
                   tgt_is_causal, memory_is_causal
               )
        2. 如果提供了 norm，则对最终输出进行层归一化
    """
    
    def __init__(
        self,
        decoder_layer: 'TransformerDecoderLayer',
        num_layers: int,
        norm: Module | None = None
    ):
        """
        初始化 TransformerDecoder 模块
        
        Args:
            decoder_layer (TransformerDecoderLayer): 单个解码器层实例，将被克隆 num_layers 次
            num_layers (int): 解码器层的数量
            norm (Module, optional): 最后的层归一化层，默认为 None
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        tgt: TN,
        memory: TN,
        tgt_mask: TN | None = None,
        memory_mask: TN | None = None,
        tgt_key_padding_mask: TN | None = None,
        memory_key_padding_mask: TN | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False
    ) -> TN:
        """
        Transformer 解码器前向传播 (Transformer Decoder Forward Pass)
        
        Args:
            tgt (TN): 解码器输入序列（目标序列），
                形状为 (tgt_len, batch_size, d_model) 或 (batch_size, tgt_len, d_model)
            memory (TN): 编码器输出（记忆），
                形状为 (src_len, batch_size, d_model) 或 (batch_size, src_len, d_model)
            tgt_mask (TN, optional): 目标序列的自注意力掩码，默认为 None
            memory_mask (TN, optional): 记忆序列的交叉注意力掩码，默认为 None
            tgt_key_padding_mask (TN, optional): 目标序列的填充掩码，默认为 None
            memory_key_padding_mask (TN, optional): 记忆序列的填充掩码，默认为 None
            tgt_is_causal (bool, optional): 是否对目标序列使用因果掩码，默认为 None
            memory_is_causal (bool, optional): 是否对记忆序列使用因果掩码，默认为 False
        
        Returns:
            TN: 解码器输出，形状与 tgt 相同
        """
        output = tgt
        
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal if tgt_is_causal is not None else False,
                memory_is_causal=memory_is_causal
            )
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


class Transformer(Module):
    """
    Transformer 模型 (Transformer Model)
    
    实现完整的 Transformer 架构，包含编码器和解码器。
    接口和行为与 torch.nn.Transformer 完全一致。
    
    内部结构:
        - encoder: Transformer 编码器 (TransformerEncoder 模块)
        - decoder: Transformer 解码器 (TransformerDecoder 模块)
        
    前向计算流程:
        1. 将 src 通过 encoder 得到 memory
        2. 将 tgt 和 memory 通过 decoder 得到输出
    """
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        custom_encoder: Module | None = None,
        custom_decoder: Module | None = None,
        layer_norm_eps: float = 1e-05,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        """
        初始化 Transformer 模块
        
        Args:
            d_model (int, optional): 编码器/解码器输入的特征维度，默认为 512
            nhead (int, optional): 多头注意力模型中的头数，默认为 8
            num_encoder_layers (int, optional): 编码器中子编码器层的数量，默认为 6
            num_decoder_layers (int, optional): 解码器中子解码器层的数量，默认为 6
            dim_feedforward (int, optional): 前馈网络模型的维度，默认为 2048
            dropout (float, optional): dropout 值，默认为 0.1
            activation (str, optional): 编码器/解码器中间层的激活函数，'relu' 或 'gelu'，默认为 'relu'
            custom_encoder (Module, optional): 自定义编码器，默认为 None
            custom_decoder (Module, optional): 自定义解码器，默认为 None
            layer_norm_eps (float, optional): 层归一化组件中的 eps 值，默认为 1e-05
            batch_first (bool, optional): 输入输出张量是否为 (batch, seq, feature) 格式，默认为 False（seq, batch, feature）
            norm_first (bool, optional): 是否在其他注意力和前馈操作之前执行 LayerNorm，默认为 False（之后）
            bias (bool, optional): Linear 和 LayerNorm 层是否学习加性偏置，默认为 True
            device: 张量设备，默认为 None
            dtype: 张量数据类型，默认为 None
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation,
                layer_norm_eps, batch_first, norm_first, bias, **factory_kwargs
            )
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation,
                layer_norm_eps, batch_first, norm_first, bias, **factory_kwargs
            )
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
    
    def forward(
        self,
        src: TN,
        tgt: TN,
        src_mask: TN | None = None,
        tgt_mask: TN | None = None,
        memory_mask: TN | None = None,
        src_key_padding_mask: TN | None = None,
        tgt_key_padding_mask: TN | None = None,
        memory_key_padding_mask: TN | None = None,
        src_is_causal: bool | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False
    ) -> TN:
        """
        Transformer 前向传播 (Transformer Forward Pass)
        
        Args:
            src (TN): 源序列，形状为 (seq_len, batch_size, d_model) 或 (batch_size, seq_len, d_model)
            tgt (TN): 目标序列，形状为 (tgt_len, batch_size, d_model) 或 (batch_size, tgt_len, d_model)
            src_mask (TN, optional): 源序列的注意力掩码，默认为 None
            tgt_mask (TN, optional): 目标序列的注意力掩码，默认为 None
            memory_mask (TN, optional): 编码器输出的注意力掩码，默认为 None
            src_key_padding_mask (TN, optional): 源序列的填充掩码，默认为 None
            tgt_key_padding_mask (TN, optional): 目标序列的填充掩码，默认为 None
            memory_key_padding_mask (TN, optional): 编码器输出的填充掩码，默认为 None
            src_is_causal (bool, optional): 源序列是否使用因果掩码，默认为 None
            tgt_is_causal (bool, optional): 目标序列是否使用因果掩码，默认为 None
            memory_is_causal (bool, optional): 编码器输出是否使用因果掩码，默认为 False
        
        Returns:
            TN: Transformer 输出，形状与 tgt 相同
        """
        memory = self.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal
        )
        
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal
        )
        
        return output
    
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> TN:
        """
        生成后续序列的方形掩码 (Generate Square Subsequent Mask)
        
        生成一个方形掩码，用于自回归解码，防止每个位置关注后面的位置。
        
        Args:
            sz (int): 序列长度
        
        Returns:
            TN: 形状为 (sz, sz) 的掩码矩阵，上三角为 -inf，下三角为 0
        """
        mask = full((sz, sz), fill_value=float('-inf'))
        mask = mask.triu(diagonal=1)
        return mask
