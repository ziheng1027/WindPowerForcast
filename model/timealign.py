"""
TimeAlign 完整模型：双分支编码 + alignment 对齐 + 投影解码。

ICLR 2026 — Bridging Past and Future:
Distribution-Aware Alignment for Time Series Forecasting
"""

import torch
import torch.nn as nn

from module.timealign import (
    GlocalAlignAblation,
    Normalize,
    PositionalEmbedding,
)


class PatchEmbed(nn.Module):
    """
    Patch 嵌入层：将时间序列分 patch 投影到 d_model 维。
    """

    def __init__(self, dim, patch_len, stride=None, pos=True):
        """
        Parameters
        ----------
        dim : int
            嵌入维度 (d_model)
        patch_len : int
            每个 patch 的长度
        stride : int or None
            滑动步长，默认等于 patch_len
        pos : bool
            是否添加位置编码
        """
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)

        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B * C, L]

        Returns
        -------
        [B * C, patch_num, D]
        """
        # x 需要先 reshape 为 [B*C, 1, L] 才能 unfold
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B*C, 1, L]

        x = x.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )
        # [B*C, 1, patch_num, patch_len]
        x = x.squeeze(1)  # [B*C, patch_num, patch_len]
        x = self.patch_proj(x)  # [B*C, patch_num, D]

        if self.pos:
            x += self.pe(x)
        return x


class TimeAlign(nn.Module):
    """
    TimeAlign 预测模型。

    双分支架构:
    - x 分支: 过去序列编码 → 预测
    - y 分支: 未来序列自编码 → 对齐辅助

    训练时同时使用两个分支，推理时仅用 x 分支。
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            模型配置，包含:
            - seq_len, pred_len: 序列长度
            - enc_in: 输入特征维度
            - patch_num: patch 数量
            - d_model, d_ff: 模型维度
            - e_layers: 编码器层数
            - dropout: dropout 率
            - layer_norm: 是否使用 LayerNorm
            - pos: 是否使用位置编码
            - local_margin, global_margin: 对齐 margin
            - loc, glo: 对齐开关
        """
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.patch_num = config["patch_num"]
        self.d_model = config["d_model"]
        self.e_layers = config["e_layers"]
        self.layer_norm = config.get("layer_norm", True)

        # Patch 嵌入
        self.patch_emb_x = PatchEmbed(
            config["d_model"],
            self.seq_len // self.patch_num,
            pos=config.get("pos", True),
        )
        self.patch_emb_y = PatchEmbed(
            config["d_model"],
            self.pred_len // self.patch_num,
            pos=config.get("pos", True),
        )

        # x 分支编码器 (MLP)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config["d_model"], config["d_ff"]),
                nn.GELU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(config["d_ff"], config["d_model"]),
            )
            for _ in range(config["e_layers"])
        ])

        # 对齐模块
        self.align = GlocalAlignAblation(
            local_margin=config.get("local_margin", 0.0),
            global_margin=config.get("global_margin", 0.0),
            loc=config.get("loc", True),
            glo=config.get("glo", True),
        )

        # FFN 映射层（对齐用）
        self.ffn = nn.ModuleList([
            nn.Linear(config["d_model"], config["d_model"])
            for _ in range(config["e_layers"])
        ])

        # y 分支自编码器
        self.autoencoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config["d_model"], config["d_ff"]),
                nn.GELU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(config["d_ff"], config["d_model"]),
            )
            for _ in range(config["e_layers"])
        ])

        # LayerNorm
        if self.layer_norm:
            self.norm_x = nn.ModuleList([
                nn.LayerNorm(config["d_model"])
                for _ in range(config["e_layers"])
            ])
            self.norm_y = nn.ModuleList([
                nn.LayerNorm(config["d_model"])
                for _ in range(config["e_layers"])
            ])

        # 投影解码
        self.proj_x = nn.Linear(
            config["d_model"] * self.patch_num, self.pred_len
        )
        self.proj_y = nn.Linear(
            config["d_model"] * self.patch_num, self.pred_len
        )

        # 实例归一化
        self.normalization_x = Normalize(config["enc_in"], affine=False)
        self.normalization_y = Normalize(config["enc_in"], affine=False)

    def forward(self, x, y, is_training=True):
        """
        Parameters
        ----------
        x : [B, seq_len, C]
            过去序列（含目标列）
        y : [B, pred_len, C]
            未来序列（训练时传入真值）
        is_training : bool
            是否训练模式

        Returns
        -------
        output : [B, pred_len, C]
            预测结果
        y_recon : [B, pred_len, C] 或 None
            重建结果（训练时）
        align_loss : float
            对齐损失
        """
        B, T, C = x.shape
        _, L, _ = y.shape

        # 归一化
        x = self.normalization_x(x, "norm")
        x = self.patch_emb_x(
            x.permute(0, 2, 1).reshape(-1, C * T)
        )

        if is_training:
            y = self.normalization_y(y, "norm")
            y = self.patch_emb_y(
                y.permute(0, 2, 1).reshape(-1, C * L)
            )

        # 编码 + 对齐
        align_loss = 0.0
        for i in range(self.e_layers):
            x = x + self.encoder[i](x)
            if self.layer_norm:
                x = self.norm_x[i](x)

            if is_training:
                x_ = self.ffn[i](x)
                y = y + self.autoencoder[i](y)
                if self.layer_norm:
                    y = self.norm_y[i](y)
                align_loss += self.align(x_, y.detach())

        align_loss /= self.e_layers

        # 投影解码
        x = self.proj_x(
            x.reshape(-1, C, self.patch_num, self.d_model)
            .flatten(start_dim=-2)
        )
        x = x.permute(0, 2, 1)
        x = self.normalization_x(x, "denorm")

        if is_training:
            y = self.proj_y(
                y.reshape(-1, C, self.patch_num, self.d_model)
                .flatten(start_dim=-2)
            )
            y = y.permute(0, 2, 1)
            y = self.normalization_y(y, "denorm")
        else:
            y = None

        return x[:, -self.pred_len :, :], y, align_loss
