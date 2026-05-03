"""
气象-功率融合数据集：拆分 Power/GFS/Tower 三路输入 + 未来 GFS。
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MeteoPowerDataset(Dataset):
    """
    气象-功率融合时序数据集。

    将 23 维特征拆分为 Power/GFS/Tower 三路，
    额外提供未来 pred_len 步的 GFS 数据。

    Returns
    -------
    tuple of 8 tensors:
        power_hist:   [seq_len, 1]    历史 Power
        gfs_hist:     [seq_len, 6]    历史 GFS
        tower_hist:   [seq_len, 6]    历史 Tower
        gfs_future:   [pred_len, 6]   未来 GFS
        power_target: [pred_len, 17]  未来标签 (前17维)
        months:       [pred_len]      月份
        dates:        [pred_len]      日期编码
        minutes:      [pred_len]      分钟数
    """

    # feature_cols 列索引 (见 preprocess.py)
    IDX_POWER = 0          # actual_power
    IDX_GFS = slice(1, 7)  # GFS 6 维
    IDX_TOWER = slice(7, 13)  # Tower 6 维

    def __init__(self, data_dir, split, seq_len, pred_len):
        """
        Parameters
        ----------
        data_dir : str
            预处理数据目录
        split : str
            "train" / "val" / "test"
        seq_len : int
            输入序列长度
        pred_len : int
            预测序列长度
        """
        self.seq_len = seq_len
        self.pred_len = pred_len

        with open(
            os.path.join(data_dir, "features.pkl"), "rb"
        ) as f:
            df = pickle.load(f)

        with open(
            os.path.join(data_dir, "split_info.json"),
            "r", encoding="utf-8",
        ) as f:
            split_info = json.load(f)

        mask = df["split"] == split
        feature_cols = split_info["feature_cols"]
        data = df.loc[mask, feature_cols].values.astype(
            np.float32
        )

        times = pd.to_datetime(df.loc[mask, "time"])
        all_months = times.dt.month.values.astype(np.int64)
        all_dates = (
            times.dt.year * 10000
            + times.dt.month * 100
            + times.dt.day
        ).values.astype(np.int64)
        all_minutes = (
            times.dt.hour * 60 + times.dt.minute
        ).values.astype(np.int64)

        self.samples = []
        self.sample_months = []
        self.sample_dates = []
        self.sample_minutes = []
        total_len = seq_len + pred_len

        for i in range(len(data) - total_len + 1):
            hist = data[i : i + seq_len]
            fut = data[i + seq_len : i + total_len]

            power_hist = hist[
                :, self.IDX_POWER : self.IDX_POWER + 1
            ]
            gfs_hist = hist[:, self.IDX_GFS]
            tower_hist = hist[:, self.IDX_TOWER]
            gfs_future = fut[:, self.IDX_GFS]
            # 标签: 前 17 维
            power_target = fut[:, :17]

            self.samples.append(
                (power_hist, gfs_hist, tower_hist,
                 gfs_future, power_target)
            )
            self.sample_months.append(
                all_months[i + seq_len : i + total_len]
            )
            self.sample_dates.append(
                all_dates[i + seq_len : i + total_len]
            )
            self.sample_minutes.append(
                all_minutes[i + seq_len : i + total_len]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (power_hist, gfs_hist, tower_hist,
         gfs_future, power_target) = self.samples[idx]
        return (
            torch.tensor(power_hist, dtype=torch.float32),
            torch.tensor(gfs_hist, dtype=torch.float32),
            torch.tensor(tower_hist, dtype=torch.float32),
            torch.tensor(gfs_future, dtype=torch.float32),
            torch.tensor(power_target, dtype=torch.float32),
            torch.tensor(
                self.sample_months[idx], dtype=torch.long
            ),
            torch.tensor(
                self.sample_dates[idx], dtype=torch.long
            ),
            torch.tensor(
                self.sample_minutes[idx], dtype=torch.long
            ),
        )


def get_meteo_dataloader(config, flag="train"):
    """
    返回 MeteoPowerDataset 的 DataLoader。

    Parameters
    ----------
    config : dict
        包含 seq_len, pred_len, batch_size, data_dir
    flag : str
        "train" / "valid" / "test"

    Returns
    -------
    DataLoader
    """
    split_map = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }
    split = split_map.get(flag, flag)
    shuffle = flag == "train"

    dataset = MeteoPowerDataset(
        data_dir=config["data_dir"],
        split=split,
        seq_len=config["seq_len"],
        pred_len=config["pred_len"],
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=flag == "train",
    )

    return loader
