"""
风电功率数据集构建：加载预处理数据 + 滑动窗口切分 + DataLoader。
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class WindPowerDataset(Dataset):
    """
    风电功率时序数据集。

    从预处理文件加载指定 split 的数据，
    按滑动窗口切分为 (x_enc, y_enc, months) 对。
    months 为预测目标窗口对应的月份列表，
    用于按月评估准确率。
    """

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

        # 加载预处理数据
        with open(os.path.join(data_dir, "features.pkl"),
                  "rb") as f:
            df = pickle.load(f)

        # 加载划分信息
        with open(os.path.join(data_dir, "split_info.json"),
                  "r", encoding="utf-8") as f:
            split_info = json.load(f)

        # 筛选指定 split
        mask = df["split"] == split
        feature_cols = split_info["feature_cols"]
        data = df.loc[mask, feature_cols].values.astype(
            np.float32
        )

        # 提取时间列的月份
        times = pd.to_datetime(df.loc[mask, "time"])
        all_months = times.dt.month.values

        # 滑动窗口切分
        self.samples = []
        self.sample_months = []
        total_len = seq_len + pred_len

        for i in range(len(data) - total_len + 1):
            x_enc = data[i : i + seq_len]
            y_enc = data[i + seq_len : i + total_len]

            # 预测目标窗口对应的月份
            y_months = all_months[
                i + seq_len : i + total_len
            ]

            self.samples.append((x_enc, y_enc))
            self.sample_months.append(y_months)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_enc, y_enc = self.samples[idx]
        months = self.sample_months[idx]
        return (
            torch.tensor(x_enc, dtype=torch.float32),
            torch.tensor(y_enc, dtype=torch.float32),
            torch.tensor(months, dtype=torch.long),
        )


def get_dataloader(config, flag="train"):
    """
    根据配置返回对应数据集的 DataLoader。

    Parameters
    ----------
    config : dict
        包含 seq_len, pred_len, batch_size, data_dir 等参数
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

    dataset = WindPowerDataset(
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
