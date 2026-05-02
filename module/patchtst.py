"""
PatchTST 子模块：多头注意力、编码器层。

Channel-independent 架构: 所有变量共享权重，
但每个变量独立通过 Transformer 编码器。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """缩放点积多头注意力。"""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        d_kv = d_model // n_heads
        self.n_heads = n_heads
        self.scale = 1.0 / math.sqrt(d_kv)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, L, D]

        Returns
        -------
        tuple[Tensor, Tensor]
            (output, attn_weights)
        """
        B, L, _ = x.shape
        H = self.n_heads

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, H, -1).transpose(1, 2)
        k = k.view(B, L, H, -1).transpose(1, 2)
        v = v.view(B, L, H, -1).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(attn * self.scale, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out), attn


class Transpose(nn.Module):
    """维度转置工具层。"""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class EncoderLayer(nn.Module):
    """
    PatchTST 编码器层。

    注意力 + FFN + 残差 + LayerNorm/BatchNorm。
    """

    def __init__(self, d_model, n_heads, d_ff=256,
                 dropout=0.1, activation="gelu"):
        super().__init__()
        self.attn = MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.BatchNorm1d(d_ff),
            Transpose(1, 2),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            Transpose(1, 2),
            nn.Conv1d(d_ff, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, L, D]

        Returns
        -------
        [B, L, D]
        """
        attn_out, _ = self.attn(x)
        x = self.norm1(x + self.dropout1(attn_out))
        x = self.norm2(x + self.ff(x))
        return x
