"""
TCN: Temporal Convolutional Network (Bai et al., 2018).

基于 causal dilated convolution 的时序模型，
使用残差连接和 weight normalization。
参考: https://github.com/locuslab/TCN
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """裁剪右侧 padding，保证因果性。"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    单个时序残差块。

    两层 dilated causal conv + ReLU + Dropout + 残差连接。
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size,
        stride, dilation, padding, dropout=0.2
    ):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    多层 TCN 堆叠。

    每层 dilation 倍增 (1, 2, 4, 8, ...)。
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2,
                 dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, stride=1,
                dilation=dilation_size, padding=padding,
                dropout=dropout,
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNForecaster(nn.Module):
    """
    TCN 预测模型。

    使用 TCN 编码输入序列，
    线性层将最后时间步映射到预测窗口。
    接口兼容: forward(x, y, is_training)。
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            模型配置，包含:
            - seq_len, pred_len: 序列长度
            - enc_in: 输入特征维度
            - dec_out: 输出通道数
            - num_channels: TCN 各层通道数列表
            - kernel_size: 卷积核大小
            - dropout: Dropout 率
        """
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.enc_in = config["enc_in"]
        self.dec_out = config.get("dec_out", 17)

        # TCN 通道配置，默认 [64, 64, 64]
        num_channels = config.get(
            "num_channels", [64, 64, 64]
        )
        kernel_size = config.get("kernel_size", 3)
        dropout = config.get("dropout", 0.2)

        # TCN backbone: 输入 [B, C, L] → [B, last_ch, L]
        self.tcn = TemporalConvNet(
            self.enc_in, num_channels,
            kernel_size=kernel_size, dropout=dropout,
        )

        # 投影: 取最后时间步 → 预测窗口
        last_ch = num_channels[-1]
        self.projection = nn.Linear(
            last_ch, self.pred_len * self.dec_out
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, y, is_training=True):
        """
        Parameters
        ----------
        x : [B, seq_len, C]
        y : [B, pred_len, C]
        is_training : bool

        Returns
        -------
        output : [B, pred_len, dec_out]
        None
        float
            0.0
        """
        B = x.shape[0]

        # [B, L, C] → [B, C, L] (Conv1d 需要 channel-first)
        x = x.permute(0, 2, 1)

        # TCN 编码: [B, C, L] → [B, last_ch, L]
        tcn_out = self.tcn(x)

        # 取最后时间步: [B, last_ch]
        last_step = tcn_out[:, :, -1]

        # 投影: [B, last_ch] → [B, pred_len * dec_out]
        last_step = self.drop(last_step)
        output = self.projection(last_step)

        # [B, pred_len * dec_out] → [B, pred_len, dec_out]
        output = output.view(B, self.pred_len, self.dec_out)

        return output, None, 0.0
