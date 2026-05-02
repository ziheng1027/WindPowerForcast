"""
LSTM / BiLSTM 基线模型用于时序预测。

Encoder-decoder 风格: LSTM 编码输入序列，
线性层将隐状态投影到预测窗口。
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    LSTM / BiLSTM 预测模型。

    支持 forward(x, y, is_training) 统一接口。
    通过 config["bidirectional"] 切换单向/双向。
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            模型配置，包含:
            - seq_len, pred_len: 序列长度
            - enc_in: 输入特征维度
            - hidden_size: LSTM 隐状态维度
            - num_layers: LSTM 层数
            - dropout: Dropout 率
            - bidirectional: 是否双向 LSTM
        """
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.enc_in = config["enc_in"]
        hidden_size = config["hidden_size"]
        num_layers = config.get("num_layers", 2)
        bidirectional = config.get("bidirectional", False)

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=config.get("dropout", 0.0)
            if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # 投影: 隐状态 → 预测
        lstm_out_size = hidden_size * self.num_directions
        self.projection = nn.Linear(
            lstm_out_size, self.pred_len * self.enc_in
        )

        self.dropout = nn.Dropout(config.get("dropout", 0.1))

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
        B = x.shape[0]

        # LSTM 编码
        lstm_out, (h_n, _) = self.lstm(x)

        # 取最后一层隐状态
        # h_n: [num_layers * directions, B, hidden]
        h_last = h_n.view(
            -1, self.num_directions, B,
            h_n.shape[-1]
        )
        # 取最后一层的两个方向拼接
        h_last = h_last[-1]  # [directions, B, hidden]
        h_last = h_last.permute(1, 0, 2).reshape(
            B, -1
        )  # [B, directions * hidden]

        # 投影
        h_last = self.dropout(h_last)
        output = self.projection(h_last)

        # [B, pred_len * C] → [B, pred_len, C]
        output = output.view(B, self.pred_len, self.enc_in)

        return output, None, 0.0
