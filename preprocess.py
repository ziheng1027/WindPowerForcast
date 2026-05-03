"""
数据预处理脚本：原始数据 → 特征工程 → 归一化 → 持久化。

执行一次即可，所有模型复用处理后的数据。
"""

import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# ========== 路径配置 ==========
RAW_DIR = "data/wind_power/raw"
PROCESSED_DIR = "data/wind_power/processed"
_date_str = datetime.now().strftime("%Y-%m-%d")
LOG_PATH = os.path.join(
    "output", "log", f"preprocess_{_date_str}.log"
)

# ========== 特征配置 ==========
TARGET_COL = "actual_power"
GFS_COLS = [
    "gfs_wind_speed",
    "gfs_wind_direction_sin",
    "gfs_wind_direction_cos",
    "gfs_temperature",
    "gfs_humidity",
    "gfs_pressure",
]
TOWER_COLS = [
    "tower_wind_speed",
    "tower_wind_direction_sin",
    "tower_wind_direction_cos",
    "tower_temperature",
    "tower_humidity",
    "tower_pressure",
]
TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos"]

# 数据划分比例 (8:1:1)
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

def log(msg):
    """同时打印到控制台和日志文件。"""
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_raw_data():
    """
    加载三份原始 CSV 并按时间对齐合并。

    Returns
    -------
    pd.DataFrame
        合并后的 DataFrame
    """
    log("[DATA] 加载原始数据...")

    power = pd.read_csv(os.path.join(RAW_DIR, "power.csv"))
    gfs = pd.read_csv(os.path.join(RAW_DIR, "gfs.csv"))
    tower = pd.read_csv(os.path.join(RAW_DIR, "tower.csv"))

    log(f"  power.csv: {len(power)} 行")
    log(f"  gfs.csv: {len(gfs)} 行")
    log(f"  tower.csv: {len(tower)} 行")

    # 统一时间列名并转换
    for df in [power, gfs, tower]:
        df["time"] = pd.to_datetime(df["time"])

    # 按时间列合并
    merged = power.merge(gfs, on="time", how="outer")
    merged = merged.merge(tower, on="time", how="outer")
    merged = merged.sort_values("time").reset_index(drop=True)

    log(f"  合并后: {len(merged)} 行")
    return merged


def handle_missing(df):
    """
    处理缺失值：前向填充 → 后向填充。

    Parameters
    ----------
    df : pd.DataFrame
        含缺失值的数据

    Returns
    -------
    pd.DataFrame
        无缺失值的数据
    """
    n_missing = df.isnull().sum()
    total_missing = n_missing.sum()

    if total_missing > 0:
        log(f"  缺失值总计: {total_missing}")
        for col in n_missing[n_missing > 0].index:
            log(f"    {col}: {n_missing[col]}")
        log("  处理策略: 前向填充 → 后向填充")
        df = df.ffill().bfill()
    else:
        log("  无缺失值")

    return df


def build_time_features(df):
    """
    构建时间编码特征：hour_sin/cos, month_sin/cos。

    Parameters
    ----------
    df : pd.DataFrame
        含 time 列的 DataFrame

    Returns
    -------
    pd.DataFrame
        添加时间编码后的 DataFrame
    """
    hour = df["time"].dt.hour + df["time"].dt.minute / 60
    month = df["time"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


# ========== 派生特征配置 ==========
DERIVED_COLS = [
    "tower_ws_cube",
    "gfs_ws_cube",
    "ws_bias",
    "delta_tower_ws",
    "delta_power",
    "power_ramp_rate",
]


def build_derived_features(df):
    """
    构建派生特征：风速三次方、偏差、变化率。

    Parameters
    ----------
    df : pd.DataFrame
        含原始特征列的 DataFrame

    Returns
    -------
    pd.DataFrame
        添加派生特征后的 DataFrame
    """
    # 风速三次方（P ∝ v³）
    df["tower_ws_cube"] = df["tower_wind_speed"] ** 3
    df["gfs_ws_cube"] = df["gfs_wind_speed"] ** 3

    # 实测与预报偏差（GFS 订正信号）
    df["ws_bias"] = (
        df["tower_wind_speed"] - df["gfs_wind_speed"]
    )

    # 变化率（15min 差分）
    df["delta_tower_ws"] = (
        df["tower_wind_speed"].diff().fillna(0.0)
    )
    df["delta_power"] = (
        df["actual_power"].diff().fillna(0.0)
    )

    # 功率变化率（归一化，避免除零）
    prev_power = df["actual_power"].shift(1)
    prev_power = prev_power.clip(lower=1.0)
    df["power_ramp_rate"] = (
        df["delta_power"] / prev_power
    ).fillna(0.0)

    return df


def split_data(df):
    """
    按月份划分 train/valid/test。

    1~10月训练，11月验证，12月测试。
    归一化参数仅用 1~10月计算，避免数据泄露。

    Parameters
    ----------
    df : pd.DataFrame
        完整数据

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    train_df = df[df["time"].dt.month <= 10].copy()
    valid_df = df[df["time"].dt.month == 11].copy()
    test_df = df[df["time"].dt.month == 12].copy()

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    log(f"  训练集: {train_df['time'].iloc[0]} ~ "
        f"{train_df['time'].iloc[-1]} ({len(train_df)} 样本)")
    log(f"  验证集: {valid_df['time'].iloc[0]} ~ "
        f"{valid_df['time'].iloc[-1]} ({len(valid_df)} 样本)")
    log(f"  测试集: {test_df['time'].iloc[0]} ~ "
        f"{test_df['time'].iloc[-1]} ({len(test_df)} 样本)")

    return train_df, valid_df, test_df


def compute_norm_params(train_df, feature_cols):
    """
    用训练集计算各特征的 min/max 归一化参数。

    Parameters
    ----------
    train_df : pd.DataFrame
        训练集数据
    feature_cols : list[str]
        需要归一化的特征列

    Returns
    -------
    dict
        各特征的 {col: {min, max}} 字典
    """
    norm_params = {}
    for col in feature_cols:
        norm_params[col] = {
            "min": float(train_df[col].min()),
            "max": float(train_df[col].max()),
        }
    return norm_params


def normalize(df, norm_params):
    """
    min-max 归一化。

    Parameters
    ----------
    df : pd.DataFrame
        原始数据
    norm_params : dict
        归一化参数

    Returns
    -------
    pd.DataFrame
        归一化后的数据
    """
    result = df.copy()
    for col, params in norm_params.items():
        valid_range = params["max"] - params["min"]
        if valid_range < 1e-8:
            result[col] = 0.0
        else:
            result[col] = (result[col] - params["min"]) / valid_range
    return result


def is_processed():
    """
    检查预处理结果是否已存在且最新。

    Returns
    -------
    bool
        True 表示已处理，可跳过
    """
    required = ["features.pkl", "norm_params.json", "split_info.json"]
    for f in required:
        path = os.path.join(PROCESSED_DIR, f)
        if not os.path.exists(path):
            return False

    # 检查 raw 文件是否比 processed 更新
    processed_mtime = os.path.getmtime(
        os.path.join(PROCESSED_DIR, "features.pkl")
    )
    for fname in ["power.csv", "gfs.csv", "tower.csv"]:
        raw_path = os.path.join(RAW_DIR, fname)
        if os.path.getmtime(raw_path) > processed_mtime:
            return False

    return True


def main():
    """主预处理流程。"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    # 清空旧日志
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    log("=" * 50)
    log("数据预处理开始")
    log("=" * 50)

    # 幂等检查
    if is_processed():
        log("预处理结果已存在且为最新，跳过处理。")
        log("如需重新处理，请删除 data/wind_power/processed/ 目录。")
        return

    # 1. 加载原始数据
    df = load_raw_data()

    # 2. 缺失值处理
    log("[DATA] 缺失值处理...")
    df = handle_missing(df)

    # 3. 特征构建
    log("[FEAT] 构建时间编码特征...")
    df = build_time_features(df)

    log("[FEAT] 构建派生特征...")
    df = build_derived_features(df)

    # 特征列顺序: 目标列在第0位
    feature_cols = (
        [TARGET_COL]
        + GFS_COLS
        + TOWER_COLS
        + DERIVED_COLS
        + TIME_FEATURES
    )
    log(f"[FEAT] 特征维度: {len(feature_cols)}")
    log(f"[FEAT] 特征列表: {feature_cols}")
    log(f"[FEAT] 数据形状: ({len(df)}, {len(feature_cols)})")

    # 4. 数据划分
    log("[DATA] 数据划分 (8:1:1)...")
    train_df, valid_df, test_df = split_data(df)

    # 5. 归一化
    log("[DATA] 计算归一化参数（仅用训练集）...")
    norm_params = compute_norm_params(train_df, feature_cols)
    for col, p in norm_params.items():
        log(f"  {col}: min={p['min']:.4f}, max={p['max']:.4f}")

    log("[DATA] 应用 min-max 归一化...")
    train_norm = normalize(train_df, norm_params)
    valid_norm = normalize(valid_df, norm_params)
    test_norm = normalize(test_df, norm_params)

    # 6. 持久化
    log("[DATA] 保存处理结果...")

    # 合并完整数据（带 split 标记）
    full_df = pd.concat([train_norm, valid_norm, test_norm],
                        ignore_index=True)
    full_df["split"] = (
        ["train"] * len(train_norm)
        + ["val"] * len(valid_norm)
        + ["test"] * len(test_norm)
    )

    # 保留 time 列用于按月评估
    full_df["time"] = df["time"].values

    # 保存特征矩阵（含 time 和 split）
    with open(os.path.join(PROCESSED_DIR, "features.pkl"), "wb") as f:
        pickle.dump(
            full_df[["time"] + feature_cols + ["split"]], f
        )

    # 导出 CSV（便于人工查看）
    csv_df = full_df[["time"] + feature_cols + ["split"]].copy()
    csv_path = os.path.join(
        PROCESSED_DIR, "features.csv"
    )
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 保存归一化参数
    with open(os.path.join(PROCESSED_DIR, "norm_params.json"), "w",
              encoding="utf-8") as f:
        json.dump(norm_params, f, indent=2, ensure_ascii=False)

    # 保存划分信息
    split_info = {
        "train_start": str(train_df["time"].iloc[0]),
        "train_end": str(train_df["time"].iloc[-1]),
        "train_size": len(train_df),
        "valid_start": str(valid_df["time"].iloc[0]),
        "valid_end": str(valid_df["time"].iloc[-1]),
        "valid_size": len(valid_df),
        "test_start": str(test_df["time"].iloc[0]),
        "test_end": str(test_df["time"].iloc[-1]),
        "test_size": len(test_df),
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
    }
    with open(os.path.join(PROCESSED_DIR, "split_info.json"), "w",
              encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    log(f"[DATA] 保存到 {PROCESSED_DIR}/")
    log(f"  features.pkl: {len(full_df)} 行 × "
        f"{len(feature_cols)} 特征")
    log(f"  features.csv: 同上（CSV 格式，含时间列）")
    log(f"  norm_params.json: {len(norm_params)} 特征的归一化参数")
    log(f"  split_info.json: 划分信息")
    log("=" * 50)
    log("数据预处理完成")
    log("=" * 50)


if __name__ == "__main__":
    main()
