"""
MetePower-Net: 气象-功率双分支融合网络。

BiLSTM 编码 Power，TCN 共享编码 GFS/Tower，
Cross-Attention 融合后注入未来 GFS，MLP 解码。
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
    """TCN 单个残差块。"""

    def __init__(self, n_inputs, n_outputs, kernel_size,
                 stride, dilation, padding, dropout=0.2):
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
        res = (
            x if self.downsample is None
            else self.downsample(x)
        )
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """多层 TCN 堆叠，每层 dilation 倍增。"""

    def __init__(self, num_inputs, num_channels,
                 kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = (
                num_inputs if i == 0
                else num_channels[i - 1]
            )
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


class MeteoPowerNet(nn.Module):
    """
    气象-功率融合预测模型。

    Power → BiLSTM 编码；
    GFS / Tower → 共享 TCN 编码；
    Cross-Attention: Power 查询 Meteo 全序列；
    未来 GFS → 独立 TCN 注入；
    MLP 解码输出。

    Parameters
    ----------
    config : dict
        模型配置，包含:
        - seq_len, pred_len: 序列长度
        - dec_out: 输出通道数 (17)
        - hidden_size: BiLSTM 隐状态维度
        - tcn_channels: 历史气象 TCN 通道列表
        - tcn_future_channels: 未来 GFS TCN 通道列表
        - kernel_size: TCN 卷积核大小
        - d_attn: Cross-Attention 维度
        - dropout: Dropout 率
    """

    def __init__(self, config):
        super().__init__()
        self.pred_len = config["pred_len"]
        self.dec_out = config.get("dec_out", 17)

        hidden_size = config.get("hidden_size", 64)
        tcn_channels = config.get(
            "tcn_channels", [32, 32, 32]
        )
        tcn_fut_ch = config.get(
            "tcn_future_channels", [32, 32]
        )
        kernel_size = config.get("kernel_size", 3)
        dropout = config.get("dropout", 0.2)
        d_attn = config.get("d_attn", 64)

        # --- Power 编码器 (BiLSTM) ---
        self.power_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.power_drop = nn.Dropout(dropout)
        power_dim = hidden_size * 2  # 128

        # --- 气象编码器 (TCN, GFS 和 Tower 共享权重) ---
        self.meteo_tcn = TemporalConvNet(
            num_inputs=6,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        meteo_dim = tcn_channels[-1]  # 32

        # --- Cross-Attention 融合 ---
        # Q 来自 Power, KV 来自 Meteo 全序列
        self.query_proj = nn.Linear(power_dim, d_attn)
        self.kv_proj = nn.Linear(
            meteo_dim * 2, d_attn
        )
        self.attn_out = nn.Linear(d_attn, d_attn)
        self.attn_scale = d_attn ** -0.5
        fused_dim = power_dim + d_attn  # 192

        # --- 未来 GFS 编码器 (独立 TCN) ---
        self.future_tcn = TemporalConvNet(
            num_inputs=6,
            num_channels=tcn_fut_ch,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        future_dim = tcn_fut_ch[-1]  # 32

        # --- MLP 解码器 ---
        total_dim = fused_dim + future_dim  # 224
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.pred_len * self.dec_out),
        )

    def forward(self, power_hist, gfs_hist, tower_hist,
                gfs_future, is_training=True):
        """
        Parameters
        ----------
        power_hist : [B, seq_len, 1]
        gfs_hist : [B, seq_len, 6]
        tower_hist : [B, seq_len, 6]
        gfs_future : [B, pred_len, 6]
        is_training : bool

        Returns
        -------
        output : [B, pred_len, dec_out]
        None
        float (0.0)
        """
        B = power_hist.shape[0]

        # --- Power 编码 (BiLSTM) ---
        lstm_out, (h_n, _) = self.power_lstm(power_hist)
        # h_n: [2, B, hidden] → [B, hidden*2]
        h_pow = h_n.permute(1, 0, 2).reshape(B, -1)
        h_pow = self.power_drop(h_pow)  # [B, 128]

        # --- 气象编码 (共享 TCN) ---
        # [B, L, C] → [B, C, L]
        gfs_in = gfs_hist.permute(0, 2, 1)
        tow_in = tower_hist.permute(0, 2, 1)
        gfs_enc = self.meteo_tcn(gfs_in)  # [B, 32, L]
        tow_enc = self.meteo_tcn(tow_in)  # [B, 32, L]

        # --- Cross-Attention 融合 ---
        # Q: [B, 1, d_attn] 来自 Power
        Q = self.query_proj(h_pow).unsqueeze(1)
        # KV: [B, seq_len, d_attn] 来自 Meteo 全序列
        meteo_kv = torch.cat(
            [gfs_enc, tow_enc], dim=1
        )  # [B, 64, seq_len]
        meteo_kv = meteo_kv.permute(0, 2, 1)
        KV = self.kv_proj(meteo_kv)

        # 注意力: [B, 1, seq_len]
        attn_w = torch.matmul(
            Q, KV.transpose(-2, -1)
        ) * self.attn_scale
        attn_w = torch.softmax(attn_w, dim=-1)
        attn_out = torch.matmul(
            attn_w, KV
        )  # [B, 1, d_attn]
        attn_out = self.attn_out(
            attn_out
        ).squeeze(1)  # [B, d_attn]

        # 融合: [B, 192]
        fused = torch.cat([h_pow, attn_out], dim=-1)

        # --- 未来 GFS 注入 ---
        fut_in = gfs_future.permute(0, 2, 1)  # [B, 6, 16]
        h_fut = self.future_tcn(fut_in)[:, :, -1]
        # [B, 32]

        # --- 解码 ---
        total = torch.cat(
            [fused, h_fut], dim=-1
        )  # [B, 224]
        output = self.decoder(total)
        output = output.view(
            B, self.pred_len, self.dec_out
        )

        return output, None, 0.0
