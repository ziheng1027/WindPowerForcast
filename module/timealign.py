"""
TimeAlign 子模块：位置编码、实例归一化、对齐损失。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """正弦余弦位置编码。"""

    def __init__(self, d_model, max_len=5000):
        """
        Parameters
        ----------
        d_model : int
            嵌入维度
        max_len : int
            最大序列长度
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, N, D]

        Returns
        -------
        [1, N, D] 位置编码
        """
        return self.pe[:, : x.size(1)]


class Normalize(nn.Module):
    """
    RevIN 风格实例归一化。

    支持 norm（归一化）和 denorm（反归一化）两种模式。
    训练时先 norm，输出时再 denorm 恢复原始分布。
    """

    def __init__(self, num_features, eps=1e-5, affine=False,
                 subtract_last=False, non_norm=False):
        """
        Parameters
        ----------
        num_features : int
            特征维度数
        eps : float
            数值稳定常数
        affine : bool
            是否使用可学习仿射参数
        subtract_last : bool
            是否减去最后一个值而非均值
        non_norm : bool
            是否跳过归一化
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode):
        """
        Parameters
        ----------
        x : [B, L, C]
        mode : str
            "norm" 或 "denorm"

        Returns
        -------
        [B, L, C]
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(
                f"不支持的模式: {mode}"
            )
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(
            torch.ones(self.num_features)
        )
        self.affine_bias = nn.Parameter(
            torch.zeros(self.num_features)
        )

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(
                x, dim=dim2reduce, keepdim=True
            ).detach()
        self.stdev = torch.sqrt(
            torch.var(
                x, dim=dim2reduce, keepdim=True, unbiased=False
            ) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class GlocalAlignAblation(nn.Module):
    """
    local + global 对齐损失（消融版）。

    local: 逐点对齐（余弦相似度）
    global: 分布对齐（Gram 矩阵差异）
    使用动态权重平衡两种损失
    """

    def __init__(self, local_margin=0.0, global_margin=0.0,
                 loc=True, glo=True):
        """
        Parameters
        ----------
        local_margin : float
            local 损失的 margin
        global_margin : float
            global 损失的 margin
        loc : bool
            是否启用 local 损失
        glo : bool
            是否启用 global 损失
        """
        super().__init__()
        self.local_margin = local_margin
        self.global_margin = global_margin
        self.loc = loc
        self.glo = glo

    def _weight_based_dynamic_loss(self, losses):
        """动态权重损失平衡。"""
        n = len(losses)
        w_avg = sum(loss.detach() for loss in losses) / n
        dyn_loss = sum(
            w_avg * loss / loss.detach() for loss in losses
        )
        return dyn_loss

    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred : [B, C, D] 编码器输出
        target : [B, C, D] 自编码器输出

        Returns
        -------
        torch.Tensor
            对齐损失
        """
        pred = nn.functional.normalize(pred, dim=-1)
        target = nn.functional.normalize(target, dim=-1)

        # local: 逐点对齐
        local_loss = torch.mean(
            F.gelu(
                1 - torch.abs(pred * target) - self.local_margin
            )
        )

        # global: Gram 矩阵分布对齐
        global_loss = torch.mean(
            F.gelu(
                torch.abs(
                    torch.matmul(pred, pred.transpose(1, 2))
                    - torch.matmul(target, target.transpose(1, 2))
                )
                - self.global_margin
            )
        )

        if not self.loc and not self.glo:
            return 0.0
        elif self.loc and not self.glo:
            return local_loss
        elif not self.loc and self.glo:
            return global_loss
        else:
            return self._weight_based_dynamic_loss(
                [local_loss, global_loss]
            )
