"""
iTransformer 子模块：反转嵌入、多头自注意力、编码器层。

核心思想: 将每个变量（variates）视为 token，
时间步作为 token 的特征维度，实现变量间注意力。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbeddingInverted(nn.Module):
    """
    反转嵌入: [B, L, N] → [B, N, d_model]

    每个变量（N）的完整时间序列作为 token，
    通过线性映射将时间维度 L 投影到 d_model。
    """

    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, L, N]

        Returns
        -------
        [B, N, d_model]
        """
        x = x.permute(0, 2, 1)  # [B, N, L]
        x = self.value_embedding(x)  # [B, N, d_model]
        return self.dropout(x)


class FullAttention(nn.Module):
    """标准缩放点积注意力。"""

    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        """
        Parameters
        ----------
        queries, keys, values : [B, L, H, E]

        Returns
        -------
        tuple[Tensor, Tensor]
            (context, attn_weights)
        """
        B, L, H, E = queries.shape
        scale = 1.0 / math.sqrt(E)

        scores = torch.einsum(
            "blhe,bshe->bhls", queries, keys
        )
        attn = torch.softmax(scores * scale, dim=-1)
        attn = self.dropout(attn)

        context = torch.einsum(
            "bhls,bshd->blhd", attn, values
        )
        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """注意力层: 投影 → 多头注意力 → 输出投影。"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = FullAttention(
            attention_dropout=dropout
        )
        self.query_projection = nn.Linear(
            d_model, d_keys * n_heads
        )
        self.key_projection = nn.Linear(
            d_model, d_keys * n_heads
        )
        self.value_projection = nn.Linear(
            d_model, d_values * n_heads
        )
        self.out_projection = nn.Linear(
            d_values * n_heads, d_model
        )
        self.n_heads = n_heads

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, L, d_model]

        Returns
        -------
        tuple[Tensor, Tensor]
        """
        B, L, _ = x.shape
        H = self.n_heads

        queries = self.query_projection(x).view(
            B, L, H, -1
        )
        keys = self.key_projection(x).view(
            B, L, H, -1
        )
        values = self.value_projection(x).view(
            B, L, H, -1
        )

        out, attn = self.inner_attention(
            queries, keys, values
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """Transformer 编码器层: 注意力 + FFN + 残差 + LayerNorm。"""

    def __init__(self, d_model, n_heads, d_ff=None,
                 dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = AttentionLayer(
            d_model, n_heads, dropout=dropout
        )
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, L, d_model]

        Returns
        -------
        tuple[Tensor, Tensor]
        """
        new_x, attn = self.attention(x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        y = x.transpose(-1, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(-1, 1)

        return self.norm2(x + y), attn
