"""
PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers (ICLR 2023).

核心思想: 将时间序列切分为 patch，每个 patch 作为
Transformer 的 token，channel-independent 处理每个变量。
"""

import torch
import torch.nn as nn

from module.patchtst import EncoderLayer


class PatchTST(nn.Module):
    """
    PatchTST 预测模型。

    Patching → 线性投影+位置编码 → Transformer 编码器
    → Flatten head → 预测。

    接口兼容: forward(x, y, is_training)。
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            模型配置，包含:
            - seq_len, pred_len: 序列长度
            - enc_in: 变量数
            - patch_len: patch 长度
            - stride: patch 步长
            - d_model, n_heads, e_layers, d_ff: Transformer 参数
            - dropout: Dropout 率
            - revin: 是否使用 RevIN 实例归一化
        """
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.enc_in = config["enc_in"]
        self.patch_len = config["patch_len"]
        self.stride = config.get("stride", config["patch_len"])

        # RevIN
        self.revin = config.get("revin", True)
        if self.revin:
            self._revin_affine = config.get("revin_affine", False)
            self._revin_mean = None
            self._revin_stdev = None

        # Patch 数量
        self.patch_num = (
            (self.seq_len - self.patch_len) // self.stride + 1
        )

        # 线性投影: patch_len → d_model
        d_model = config["d_model"]
        self.patch_proj = nn.Linear(self.patch_len, d_model)

        # 可学习位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_num, d_model) * 0.02
        )
        self.pos_drop = nn.Dropout(config["dropout"])

        # Transformer 编码器（channel-independent: 共享权重）
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_heads=config["n_heads"],
                d_ff=config.get("d_ff", 4 * d_model),
                dropout=config["dropout"],
            )
            for _ in range(config["e_layers"])
        ])

        # Flatten head: d_model × patch_num → pred_len
        self.head_flatten = nn.Flatten(start_dim=-2)
        self.head_linear = nn.Linear(
            d_model * self.patch_num, self.pred_len
        )
        self.head_dropout = nn.Dropout(config["dropout"])

    def _norm(self, x):
        """RevIN 归一化。"""
        self._revin_mean = x.mean(dim=1, keepdim=True).detach()
        self._revin_stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False)
            + 1e-5
        ).detach()
        return (x - self._revin_mean) / self._revin_stdev

    def _denorm(self, x):
        """RevIN 反归一化。"""
        return (
            x * self._revin_stdev[:, 0, :].unsqueeze(1)
            + self._revin_mean[:, 0, :].unsqueeze(1)
        )

    def forward(self, x, y, is_training=True):
        """
        Parameters
        ----------
        x : [B, seq_len, C]
        y : [B, pred_len, C]
            未来序列（仅提取目标真值）
        is_training : bool

        Returns
        -------
        output : [B, pred_len, C]
        None
        float
            0.0（无对齐损失）
        """
        B, L, N = x.shape

        # RevIN 归一化
        if self.revin:
            x = self._norm(x)

        # [B, L, N] → [B, N, L]
        z = x.permute(0, 2, 1)

        # Patching: unfold → [B, N, patch_num, patch_len]
        z = z.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )

        # 投影: [B, N, patch_num, patch_len] → [B*N, patch_num, D]
        z = self.patch_proj(z)
        z = z.reshape(B * N, self.patch_num, -1)

        # 加位置编码
        z = self.pos_drop(z + self.pos_embed)

        # Transformer 编码器
        for layer in self.encoder_layers:
            z = layer(z)

        # [B*N, patch_num, D] → [B, N, D, patch_num]
        z = z.reshape(B, N, self.patch_num, -1)
        z = z.permute(0, 1, 3, 2)  # [B, N, D, patch_num]

        # Flatten head: [B, N, D*patch_num] → [B, N, pred_len]
        z = self.head_flatten(z)
        z = self.head_linear(z)
        z = self.head_dropout(z)

        # [B, N, pred_len] → [B, pred_len, N]
        output = z.permute(0, 2, 1)

        # RevIN 反归一化
        if self.revin:
            output = self._denorm(output)

        return output, None, 0.0
