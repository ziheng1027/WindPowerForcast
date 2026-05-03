"""
iTransformer: 反转 Transformer 用于多变量时序预测。

论文: iTransformer: Inverted Transformers Are Effective
     for Time Series Forecasting (ICLR 2024)

核心创新: 将每个变量视为独立 token，
通过注意力机制捕获变量间依赖关系，
而非传统 Transformer 的时序间注意力。
"""

import torch
import torch.nn as nn

from module.itransformer import (
    DataEmbeddingInverted,
    EncoderLayer,
)


class ITransformer(nn.Module):
    """
    iTransformer 预测模型。

    架构: 反转嵌入 → 堆叠编码器层 → 线性投影
    接口兼容 TimeAlign: forward(x, y, is_training)。
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            模型配置，包含:
            - seq_len, pred_len: 序列长度
            - enc_in: 输入特征维度（变量数）
            - d_model: 模型隐藏维度
            - n_heads: 注意力头数
            - e_layers: 编码器层数
            - d_ff: FFN 中间维度
            - dropout: Dropout 率
            - use_norm: 是否使用实例归一化
        """
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.enc_in = config["enc_in"]
        self.use_norm = config.get("use_norm", True)
        self.dec_out = config.get("dec_out", 1)

        # 反转嵌入: 每个变量的时间序列 → d_model
        self.enc_embedding = DataEmbeddingInverted(
            self.seq_len, config["d_model"],
            dropout=config["dropout"]
        )

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                d_ff=config.get("d_ff", 4 * config["d_model"]),
                dropout=config["dropout"],
            )
            for _ in range(config["e_layers"])
        ])
        self.norm = nn.LayerNorm(config["d_model"])

        # 投影: [B, N, d_model] → [B, N, pred_len]
        self.projector = nn.Linear(
            config["d_model"], self.pred_len
        )

    def forward(self, x, y, is_training=True):
        """
        Parameters
        ----------
        x : [B, seq_len, C]
            输入序列（含所有变量）
        y : [B, pred_len, C]
            未来序列（仅用于提取目标真值）
        is_training : bool
            是否训练模式

        Returns
        -------
        output : [B, pred_len, C]
            预测结果
        None
            iTransformer 无重建分支
        float
            0.0（无对齐损失）
        """
        B, L, N = x.shape

        # 实例归一化（Non-stationary Transformer 风格）
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(
                    x, dim=1, keepdim=True,
                    unbiased=False
                ) + 1e-5
            ).detach()
            x = x / stdev

        # 反转嵌入: [B, L, N] → [B, N, d_model]
        enc_out = self.enc_embedding(x)

        # 编码器
        for layer in self.encoder_layers:
            enc_out, _ = layer(enc_out)
        enc_out = self.norm(enc_out)

        # 投影: [B, N, d_model] → [B, N, pred_len]
        dec_out = self.projector(enc_out)
        # [B, N, pred_len] → [B, pred_len, N]
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]

        # 反归一化
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )

        return dec_out, None, 0.0
