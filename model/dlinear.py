"""
DLinear: Are Transformers Effective for Time Series Forecasting? (AAAI 2023).

极简基线模型: 移动平均分解 + 两个线性层。
DLinear 用作基准，验证复杂模型的必要性。
"""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    移动平均平滑，用于提取趋势分量。

    对时间序列两端做 replicate padding，
    再通过 AvgPool1d 平滑。
    """

    def __init__(self, kernel_size, stride=1):
        """
        Parameters
        ----------
        kernel_size : int
            移动平均窗口大小
        stride : int
            滑动步长
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, L, C]

        Returns
        -------
        [B, L, C]
        """
        # 两端 replicate padding
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # AvgPool1d 作用于 [B, C, L]
        result = self.avg(x_padded.permute(0, 2, 1))
        return result.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """
    序列分解: 趋势（移动平均） + 季节性（残差）。
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, L, C]

        Returns
        -------
        tuple[Tensor, Tensor]
            (seasonal, trend)
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear 预测模型。

    分解 → 趋势线性映射 + 季节性线性映射 → 合并。
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
            - individual: 是否每个变量独立线性层
            - kernel_size: 移动平均窗口
        """
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.enc_in = config["enc_in"]
        self.dec_out = config.get("dec_out", 1)
        self.individual = config.get("individual", False)

        kernel_size = config.get("kernel_size", 25)
        self.decomp = SeriesDecomp(kernel_size)

        if self.individual:
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len)
                for _ in range(self.dec_out)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len)
                for _ in range(self.dec_out)
            ])
        else:
            self.linear_seasonal = nn.Linear(
                self.seq_len, self.pred_len
            )
            self.linear_trend = nn.Linear(
                self.seq_len, self.pred_len
            )

    def forward(self, x, y, is_training=True):
        """
        Parameters
        ----------
        x : [B, seq_len, C]
        y : [B, pred_len, C]
        is_training : bool

        Returns
        -------
        output : [B, pred_len, C]
        None
        float
            0.0
        """
        # 分解
        seasonal, trend = self.decomp(x)

        # [B, L, C] → [B, C, L]
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            B, C, _ = seasonal.shape
            s_out = torch.zeros(
                B, self.dec_out, self.pred_len,
                device=seasonal.device, dtype=seasonal.dtype
            )
            t_out = torch.zeros_like(s_out)
            for i in range(self.dec_out):
                # 只取 actual_power 通道 (index 0)
                s_out[:, i, :] = self.linear_seasonal[i](
                    seasonal[:, 0, :]
                )
                t_out[:, i, :] = self.linear_trend[i](
                    trend[:, 0, :]
                )
        else:
            s_out = self.linear_seasonal(seasonal)
            t_out = self.linear_trend(trend)

        if self.individual:
            output = s_out + t_out
        else:
            # 只取 actual_power 通道 (index 0)
            output = s_out[:, 0:1, :] + t_out[:, 0:1, :]
        output = output.permute(0, 2, 1)  # [B, pred_len, dec_out]

        return output, None, 0.0
