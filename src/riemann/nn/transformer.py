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
including MultiheadAttention, which is a core component of Transformer models.
All components are designed to be compatible with PyTorch's interface and behavior.

Implemented Classes:
- MultiheadAttention: Multi-head attention mechanism, compatible with torch.nn.MultiheadAttention
"""

import math
from typing import Optional, Tuple
from ..tensordef import *
from .module import Module, Parameter, Linear
from .functional import linear, softmax, dropout


class MultiheadAttention(Module):
    """
    多头注意力机制 (Multi-head Attention Mechanism)
    
    实现多头注意力机制，允许模型同时关注来自不同表示子空间的信息。
    接口和行为与torch.nn.MultiheadAttention完全一致。
    
    Args:
        embed_dim (int): 输入和输出向量的维度大小
        num_heads (int): 多头注意力中使用的头部数量
        dropout (float, optional): 训练过程中应用的dropout概率，默认值为0.0
        bias (bool, optional): 是否在投影层中添加偏置项，默认为True
        add_bias_kv (bool, optional): 是否在key和value序列的末尾添加偏置项，默认为False
        add_zero_attn (bool, optional): 是否在注意力权重中添加一列零，默认为False
        kdim (int, optional): key的维度，如果为None则使用embed_dim，默认为None
        vdim (int, optional): value的维度，如果为None则使用embed_dim，默认为None
        batch_first (bool, optional): 如果为True，输入输出形状为(batch, seq, feature)，
            否则为(seq, batch, feature)，默认为False
        
    Examples:
        >>> # 创建多头注意力层
        >>> multihead_attn = MultiheadAttention(embed_dim=512, num_heads=8)
        >>> 
        >>> # 创建输入
        >>> query = rm.randn(10, 32, 512)  # (seq_len, batch_size, embed_dim)
        >>> key = rm.randn(10, 32, 512)
        >>> value = rm.randn(10, 32, 512)
        >>> 
        >>> # 前向传播
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
        >>> print(attn_output.shape)  # (10, 32, 512)
        >>> print(attn_output_weights.shape)  # (32, 10, 10)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None
    ):
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
        """重置参数"""
        bound = math.sqrt(1.0 / self.embed_dim)
        if self._qkv_same_embed_dim:
            arrlib = self.in_proj_weight._get_array_lib()
            self.in_proj_weight.data[:] = arrlib.random.uniform(-bound, bound, self.in_proj_weight.shape)
            if self.in_proj_bias is not None:
                self.in_proj_bias.data.fill(0.)
        else:
            arrlib = self.q_proj_weight._get_array_lib()
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
        key_padding_mask: Optional[TN] = None,
        need_weights: bool = True,
        attn_mask: Optional[TN] = None,
        average_attn_weights: bool = True
    ) -> Tuple[TN, Optional[TN]]:
        """
        前向传播 (Forward Pass)
        
        Args:
            query (TN): 查询张量，形状为(seq_len, batch_size, embed_dim)或(batch_size, seq_len, embed_dim)
            key (TN): 键张量，形状为(key_len, batch_size, kdim)或(batch_size, key_len, kdim)
            value (TN): 值张量，形状为(key_len, batch_size, vdim)或(batch_size, key_len, vdim)
            key_padding_mask (TN, optional): key序列的填充掩码，形状为(batch_size, key_len)，
                其中填充位置为True，默认为None
            need_weights (bool, optional): 是否返回注意力权重，默认为True
            attn_mask (TN, optional): 注意力掩码，形状为(seq_len, key_len)或(batch_size, num_heads, seq_len, key_len)，
                默认为None
            average_attn_weights (bool, optional): 是否对注意力权重进行平均，默认为True
        
        Returns:
            Tuple[TN, Optional[TN]]: 
                - attn_output: 注意力输出，形状与query相同
                - attn_output_weights: 注意力权重，形状为(batch_size, seq_len, key_len)，
                    如果need_weights为False则为None
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
                    [key_padding_mask, zeros((bsz, 1), dtype=key_padding_mask.dtype)],
                    dim=1
                )
        
        if self.add_zero_attn:
            k = concatenate([k, zeros((1, bsz, self.embed_dim))], dim=0)
            v = concatenate([v, zeros((1, bsz, self.embed_dim))], dim=0)
            if key_padding_mask is not None:
                key_padding_mask = concatenate(
                    [key_padding_mask, zeros((bsz, 1), dtype=key_padding_mask.dtype)],
                    dim=1
                )
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        attn_output_weights = q @ k.mT / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_output_weights += attn_mask.unsqueeze(0)
            elif attn_mask.ndim == 3:
                attn_output_weights += attn_mask.view(bsz * self.num_heads, tgt_len, -1)
        
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, -1)
            attn_output_weights.masked_fill_(
                key_padding_mask.unsqueeze((1, 2)),
                float('-inf')
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, -1)
        
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        
        if self.dropout > 0:
            attn_output_weights = dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        attn_output = attn_output_weights @ v
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, -1)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
        else:
            attn_output_weights = None
        
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        return attn_output, attn_output_weights
    
    def _in_proj_qkv(self, query, key, value):
        """投影query、key和value"""
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
