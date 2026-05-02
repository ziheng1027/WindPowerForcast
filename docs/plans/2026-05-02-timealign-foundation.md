# TimeAlign 基础设施 + 模型实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 TimeAlign 训练链路的最小基础设施，包含数据预处理、工具集、模型、训练器和入口脚本。

**Architecture:** 两阶段数据管线（预处理一次性持久化 + 模型复用），模块化训练器（基类+派生），全流程四阶段日志。TimeAlign 模型采用双分支编码+alignment 对齐机制。

**Tech Stack:** Python 3.10, PyTorch 2.11.0+cu128, pandas, numpy, pyyaml, matplotlib

**Spec:** `docs/2026-05-02-timealign-foundation-design.md`

---

## File Structure

| 操作 | 文件 | 职责 |
|------|------|------|
| Create | `preprocess.py` | 数据预处理入口（一次性） |
| Create | `dataset/wind_power.py` | 加载 processed 数据 + 滑动窗口 |
| Create | `tool/utils.py` | 种子、配置加载、设备、参数统计 |
| Create | `tool/logger.py` | 全流程四阶段日志 |
| Create | `tool/metrics.py` | 基础指标 + 国网评估指标 |
| Create | `tool/early_stopping.py` | 早停 + 保存最佳模型 |
| Create | `tool/visualize.py` | loss 曲线、预测对比图 |
| Create | `module/timealign.py` | PositionalEmbedding + Normalize + GlocalAlignAblation |
| Create | `model/timealign.py` | PatchEmbed + TimeAlign 主模型 |
| Create | `config/timealign.yml` | 模型 + 训练超参配置 |
| Create | `trainer/trainer_base.py` | 训练基类（通用流程） |
| Create | `trainer/trainer_timealign.py` | TimeAlign 训练器（双输入+三损失） |
| Create | `train.py` | 训练入口 |
| Create | `test.py` | 测试入口 |

---

## Task 1: 工具函数 — `tool/utils.py`

**Files:**
- Create: `tool/utils.py`

- [ ] **Step 1: 实现 `tool/utils.py`**

```python
"""
通用工具函数：随机种子、配置加载、设备选择、参数统计。
"""

import random

import numpy as np
import torch
import yaml


def set_seed(seed=42):
    """设置全局随机种子，确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """
    加载 YAML 配置文件。

    Parameters
    ----------
    config_path : str
        配置文件路径

    Returns
    -------
    dict
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_device(config):
    """
    根据配置返回计算设备。

    Parameters
    ----------
    config : dict
        包含 device 字段的配置

    Returns
    -------
    torch.device
    """
    device_str = config.get("device", "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model):
    """
    统计模型可训练参数量。

    Parameters
    ----------
    model : nn.Module
        PyTorch 模型

    Returns
    -------
    int
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from tool.utils import set_seed, load_config, get_device, count_parameters; print('OK')"`

Expected: `OK`

---

## Task 2: 全流程日志 — `tool/logger.py`

**Files:**
- Create: `tool/logger.py`

- [ ] **Step 1: 实现 `tool/logger.py`**

```python
"""
全流程训练日志，支持四阶段标记：[DATA]/[FEAT]/[TRAIN]/[EVAL]。
"""

import os
import sys
import datetime


# ANSI 颜色码
_COLORS = {
    "DATA": "\033[94m",    # 蓝色
    "FEAT": "\033[92m",    # 绿色
    "TRAIN": "\033[93m",   # 黄色
    "EVAL": "\033[96m",    # 青色
    "RESET": "\033[0m",
}


class Logger:
    """
    全流程日志记录器。

    四阶段标记: [DATA] 数据处理 / [FEAT] 特征构建
                [TRAIN] 模型训练 / [EVAL] 测试评估
    """

    def __init__(self, log_dir, model_name):
        """
        初始化日志器，创建日志目录和文件。

        Parameters
        ----------
        log_dir : str
            日志保存目录
        model_name : str
            模型名称，用于日志文件命名
        """
        self.model_name = model_name
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "train.log")
        self._file = open(log_path, "a", encoding="utf-8")

        # 写入分隔线标记新会话
        self._write_raw("=" * 60)
        self._write_raw(
            f"新会话: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._write_raw("=" * 60)

    def _write_raw(self, msg):
        """写入原始消息到文件和控制台。"""
        self._file.write(msg + "\n")
        self._file.flush()
        print(msg)

    def _log(self, tag, msg):
        """
        带标签的日志记录。

        Parameters
        ----------
        tag : str
            阶段标记 (DATA/FEAT/TRAIN/EVAL)
        msg : str
            日志内容
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{tag}] {timestamp} | {msg}"

        # 文件写入（无颜色）
        self._file.write(line + "\n")
        self._file.flush()

        # 控制台（带颜色）
        color = _COLORS.get(tag, "")
        reset = _COLORS["RESET"]
        print(f"{color}{line}{reset}")

    def log_data(self, msg):
        """记录数据处理阶段信息。"""
        self._log("DATA", msg)

    def log_feat(self, msg):
        """记录特征构建阶段信息。"""
        self._log("FEAT", msg)

    def log_train(self, msg):
        """记录训练阶段信息。"""
        self._log("TRAIN", msg)

    def log_eval(self, msg):
        """记录评估阶段信息。"""
        self._log("EVAL", msg)

    def log_config(self, config):
        """
        记录完整配置信息。

        Parameters
        ----------
        config : dict
            配置字典
        """
        self._write_raw("\n--- 配置信息 ---")
        for key, value in sorted(config.items()):
            self._write_raw(f"  {key}: {value}")
        self._write_raw("--- 配置结束 ---\n")

    def close(self):
        """关闭日志文件。"""
        if self._file and not self._file.closed:
            self._file.close()
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from tool.logger import Logger; print('OK')"`

Expected: `OK`

---

## Task 3: 评估指标 — `tool/metrics.py`

**Files:**
- Create: `tool/metrics.py`

- [ ] **Step 1: 实现 `tool/metrics.py`**

```python
"""
评估指标：基础指标 + 国网 GBT+40607-2021 标准。
"""

import numpy as np


# 装机容量 (14台 × 5000kW, EN-171/5.0)
CAPACITY = 70000.0


def calculate_metrics(preds, trues, capacity=CAPACITY):
    """
    计算全部评估指标。

    Parameters
    ----------
    preds : np.ndarray
        预测值，shape [N, pred_len] 或 [N, pred_len, C]
    trues : np.ndarray
        真实值，同 shape
    capacity : float
        装机容量 (kW)

    Returns
    -------
    dict
        包含 mae, mse, rmse, mape, accuracy, qualification_rate
    """
    # 仅取目标列（第0列）
    if preds.ndim == 3:
        preds = preds[:, :, 0]
        trues = trues[:, :, 0]

    error = preds - trues
    abs_error = np.abs(error)

    mae = float(np.mean(abs_error))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))

    # MAPE: 避免除零
    mask = np.abs(trues) > 1e-6
    if mask.sum() > 0:
        mape = float(np.mean(abs_error[mask] / np.abs(trues[mask]))) * 100
    else:
        mape = 0.0

    # 国网准确率
    accuracy = (1 - rmse / capacity) * 100

    # 国网合格率
    threshold = 0.25 * capacity
    qualified = abs_error <= threshold
    qualification_rate = float(np.mean(qualified)) * 100

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "accuracy": accuracy,
        "qualification_rate": qualification_rate,
    }


def calculate_step_metrics(preds, trues, step, capacity=CAPACITY):
    """
    计算指定步长的评估指标。

    Parameters
    ----------
    preds : np.ndarray
        预测值，shape [N, pred_len] 或 [N, pred_len, C]
    trues : np.ndarray
        真实值
    step : int
        步长索引（0-based，如第16步为 step=15）
    capacity : float
        装机容量

    Returns
    -------
    dict
        该步长的全部指标
    """
    if preds.ndim == 3:
        preds_step = preds[:, step, 0]
        trues_step = trues[:, step, 0]
    else:
        preds_step = preds[:, step]
        trues_step = trues[:, step]

    # 扩展为 [N, 1] 复用 calculate_metrics
    preds_exp = preds_step.reshape(-1, 1)
    trues_exp = trues_step.reshape(-1, 1)

    result = calculate_metrics(preds_exp, trues_exp, capacity)
    result["step"] = step + 1
    return result


def calculate_all_step_metrics(preds, trues, capacity=CAPACITY):
    """
    计算所有步长的评估指标。

    Returns
    -------
    list[dict]
        每个步长的指标字典
    """
    if preds.ndim == 3:
        n_steps = preds.shape[1]
    else:
        n_steps = preds.shape[1]

    results = []
    for s in range(n_steps):
        result = calculate_step_metrics(preds, trues, s, capacity)
        results.append(result)
    return results
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from tool.metrics import calculate_metrics, calculate_step_metrics, calculate_all_step_metrics; print('OK')"`

Expected: `OK`

---

## Task 4: 早停机制 — `tool/early_stopping.py`

**Files:**
- Create: `tool/early_stopping.py`

- [ ] **Step 1: 实现 `tool/early_stopping.py`**

```python
"""
早停机制：监控验证集指标，自动保存最佳模型。
"""

import os

import torch


class EarlyStopping:
    """
    早停控制器。

    当验证损失连续 patience 个 epoch 未改善时触发早停，
    并在每次改善时自动保存最佳模型权重。
    """

    def __init__(self, patience=7, verbose=True, delta=0):
        """
        Parameters
        ----------
        patience : int
            容忍的 epoch 数
        verbose : bool
            是否打印改善信息
        delta : float
            最小改善量阈值
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model, path):
        """
        检查是否需要早停并保存最佳模型。

        Parameters
        ----------
        val_loss : float
            当前验证损失
        model : nn.Module
            待保存的模型
        path : str
            模型保存目录
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, path)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: "
                    f"{self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model, path)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model, path):
        """
        保存最佳模型权重。

        Parameters
        ----------
        val_loss : float
            当前验证损失
        model : nn.Module
            待保存的模型
        path : str
            模型保存目录
        """
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, "best_model.pth")
        if self.verbose:
            print(
                f"验证损失改善: "
                f"{self.val_loss_min:.6f} → {val_loss:.6f}，"
                f"保存模型 → {save_path}"
            )
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from tool.early_stopping import EarlyStopping; print('OK')"`

Expected: `OK`

---

## Task 5: 可视化工具 — `tool/visualize.py`

**Files:**
- Create: `tool/visualize.py`

- [ ] **Step 1: 实现 `tool/visualize.py`**

```python
"""
可视化工具：loss 曲线、预测对比图、步长准确率。
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_loss_curve(train_losses, val_losses, save_path):
    """
    绘制训练/验证 loss 曲线。

    Parameters
    ----------
    train_losses : list[float]
        每 epoch 训练损失
    val_losses : list[float]
        每 epoch 验证损失
    save_path : str
        图片保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练/验证损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_prediction(preds, trues, save_path, n_samples=5):
    """
    绘制预测 vs 真值对比图。

    Parameters
    ----------
    preds : np.ndarray
        预测值，shape [N, pred_len]
    trues : np.ndarray
        真实值，shape [N, pred_len]
    save_path : str
        图片保存路径
    n_samples : int
        展示的样本数
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 仅取目标列
    if preds.ndim == 3:
        preds = preds[:, :, 0]
        trues = trues[:, :, 0]

    indices = np.linspace(0, len(preds) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        steps = range(1, preds.shape[1] + 1)
        ax.plot(steps, trues[idx], label="真实值", linewidth=1.5)
        ax.plot(steps, preds[idx], label="预测值", linewidth=1.5, linestyle="--")
        ax.set_title(f"样本 #{idx}")
        ax.set_xlabel("预测步长")
        ax.set_ylabel("功率 (kW)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("预测 vs 真值", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_step_accuracy(preds, trues, capacity, save_path):
    """
    绘制各步长的准确率和合格率。

    Parameters
    ----------
    preds : np.ndarray
        预测值，shape [N, pred_len]
    trues : np.ndarray
        真实值
    capacity : float
        装机容量
    save_path : str
        图片保存路径
    """
    from tool.metrics import calculate_all_step_metrics

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if preds.ndim == 3:
        preds = preds[:, :, 0]
        trues = trues[:, :, 0]

    step_metrics = calculate_all_step_metrics(preds, trues, capacity)
    steps = [m["step"] for m in step_metrics]
    accuracies = [m["accuracy"] for m in step_metrics]
    qual_rates = [m["qualification_rate"] for m in step_metrics]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("预测步长")
    ax1.set_ylabel("准确率 (%)", color=color1)
    ax1.plot(steps, accuracies, color=color1, marker="o", label="准确率")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("合格率 (%)", color=color2)
    ax2.plot(steps, qual_rates, color=color2, marker="s", label="合格率")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("各步长准确率与合格率")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from tool.visualize import plot_loss_curve, plot_prediction, plot_step_accuracy; print('OK')"`

Expected: `OK`

---

## Task 6: 数据预处理 — `preprocess.py`

**Files:**
- Create: `preprocess.py`

- [ ] **Step 1: 实现 `preprocess.py`**

```python
"""
数据预处理脚本：原始数据 → 特征工程 → 归一化 → 持久化。

执行一次即可，所有模型复用处理后的数据。
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd


# ========== 路径配置 ==========
RAW_DIR = "data/wind_power/raw"
PROCESSED_DIR = "data/wind_power/processed"
LOG_PATH = os.path.join(PROCESSED_DIR, "preprocess.log")

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
TRAIN_RATIO = 8 / 10
VAL_RATIO = 1 / 10


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


def split_data(df):
    """
    按时间顺序划分 train/val/test。

    Parameters
    ----------
    df : pd.DataFrame
        完整数据

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    log(f"  训练集: {train_df['time'].iloc[0]} ~ "
        f"{train_df['time'].iloc[-1]} ({len(train_df)} 样本)")
    log(f"  验证集: {val_df['time'].iloc[0]} ~ "
        f"{val_df['time'].iloc[-1]} ({len(val_df)} 样本)")
    log(f"  测试集: {test_df['time'].iloc[0]} ~ "
        f"{test_df['time'].iloc[-1]} ({len(test_df)} 样本)")

    return train_df, val_df, test_df


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
        val_range = params["max"] - params["min"]
        if val_range < 1e-8:
            result[col] = 0.0
        else:
            result[col] = (result[col] - params["min"]) / val_range
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

    # 特征列顺序: 目标列在第0位
    feature_cols = [TARGET_COL] + GFS_COLS + TOWER_COLS + TIME_FEATURES
    log(f"[FEAT] 特征维度: {len(feature_cols)}")
    log(f"[FEAT] 特征列表: {feature_cols}")
    log(f"[FEAT] 数据形状: ({len(df)}, {len(feature_cols)})")

    # 4. 数据划分
    log("[DATA] 数据划分 (8:1:1)...")
    train_df, val_df, test_df = split_data(df)

    # 5. 归一化
    log("[DATA] 计算归一化参数（仅用训练集）...")
    norm_params = compute_norm_params(train_df, feature_cols)
    for col, p in norm_params.items():
        log(f"  {col}: min={p['min']:.4f}, max={p['max']:.4f}")

    log("[DATA] 应用 min-max 归一化...")
    train_norm = normalize(train_df, norm_params)
    val_norm = normalize(val_df, norm_params)
    test_norm = normalize(test_df, norm_params)

    # 6. 持久化
    log("[DATA] 保存处理结果...")

    # 合并完整数据（带 split 标记）
    full_df = pd.concat([train_norm, val_norm, test_norm], ignore_index=True)
    full_df["split"] = (
        ["train"] * len(train_norm)
        + ["val"] * len(val_norm)
        + ["test"] * len(test_norm)
    )

    # 保存特征矩阵
    with open(os.path.join(PROCESSED_DIR, "features.pkl"), "wb") as f:
        pickle.dump(full_df[feature_cols + ["split"]], f)

    # 保存归一化参数
    with open(os.path.join(PROCESSED_DIR, "norm_params.json"), "w",
              encoding="utf-8") as f:
        json.dump(norm_params, f, indent=2, ensure_ascii=False)

    # 保存划分信息
    split_info = {
        "train_start": str(train_df["time"].iloc[0]),
        "train_end": str(train_df["time"].iloc[-1]),
        "train_size": len(train_df),
        "val_start": str(val_df["time"].iloc[0]),
        "val_end": str(val_df["time"].iloc[-1]),
        "val_size": len(val_df),
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
    log(f"  features.pkl: {len(full_df)} 行 × {len(feature_cols)} 特征")
    log(f"  norm_params.json: {len(norm_params)} 特征的归一化参数")
    log(f"  split_info.json: 划分信息")
    log("=" * 50)
    log("数据预处理完成")
    log("=" * 50)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行预处理脚本**

Run: `python preprocess.py`

Expected:
```
[DATA] 加载原始数据...
[DATA] 缺失值处理...
[FEAT] 构建时间编码特征...
[FEAT] 特征维度: 16
[DATA] 数据预处理完成
```

- [ ] **Step 3: 验证输出文件存在**

Run: `python -c "import os; files = os.listdir('data/wind_power/processed'); assert 'features.pkl' in files; assert 'norm_params.json' in files; assert 'split_info.json' in files; print('OK')"`

Expected: `OK`

---

## Task 7: 数据集构建 — `dataset/wind_power.py`

**Files:**
- Create: `dataset/wind_power.py`

- [ ] **Step 1: 实现 `dataset/wind_power.py`**

```python
"""
风电功率数据集构建：加载预处理数据 + 滑动窗口切分 + DataLoader。
"""

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WindPowerDataset(Dataset):
    """
    风电功率时序数据集。

    从预处理文件加载指定 split 的数据，
    按滑动窗口切分为 (x_enc, y_enc) 对。
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
        with open(os.path.join(data_dir, "features.pkl"), "rb") as f:
            df = pickle.load(f)

        # 加载划分信息
        with open(os.path.join(data_dir, "split_info.json"), "r",
                  encoding="utf-8") as f:
            split_info = json.load(f)

        # 筛选指定 split
        mask = df["split"] == split
        feature_cols = split_info["feature_cols"]
        data = df.loc[mask, feature_cols].values.astype(np.float32)

        # 滑动窗口切分
        self.samples = []
        total_len = seq_len + pred_len
        for i in range(len(data) - total_len + 1):
            x_enc = data[i : i + seq_len]          # [seq_len, C]
            y_enc = data[i + seq_len : i + total_len]  # [pred_len, C]
            self.samples.append((x_enc, y_enc))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_enc, y_enc = self.samples[idx]
        return (
            torch.tensor(x_enc, dtype=torch.float32),
            torch.tensor(y_enc, dtype=torch.float32),
        )


def get_dataloader(config, flag="train"):
    """
    根据配置返回对应数据集的 DataLoader。

    Parameters
    ----------
    config : dict
        包含 seq_len, pred_len, batch_size, data_dir 等参数
    flag : str
        "train" / "val" / "test"

    Returns
    -------
    DataLoader
    """
    split_map = {"train": "train", "val": "val", "test": "test"}
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
```

- [ ] **Step 2: 验证数据集可加载**

Run:
```python
python -c "
import sys; sys.path.insert(0, '.')
from dataset.wind_power import get_dataloader
config = {'data_dir': 'data/wind_power/processed', 'seq_len': 96, 'pred_len': 16, 'batch_size': 32}
loader = get_dataloader(config, 'train')
batch = next(iter(loader))
print(f'x_enc: {batch[0].shape}, y_enc: {batch[1].shape}')
print('OK')
"
```

Expected: `x_enc: torch.Size([32, 96, 16]), y_enc: torch.Size([32, 16, 16])` + `OK`

---

## Task 8: TimeAlign 子模块 — `module/timealign.py`

**Files:**
- Create: `module/timealign.py`

- [ ] **Step 1: 实现 `module/timealign.py`**

```python
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
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from module.timealign import PositionalEmbedding, Normalize, GlocalAlignAblation; print('OK')"`

Expected: `OK`

---

## Task 9: TimeAlign 完整模型 — `model/timealign.py`

**Files:**
- Create: `model/timealign.py`

- [ ] **Step 1: 实现 `model/timealign.py`**

```python
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
```

- [ ] **Step 2: 验证模型可实例化**

Run:
```python
python -c "
import sys; sys.path.insert(0, '.')
from model.timealign import TimeAlign
config = {
    'seq_len': 96, 'pred_len': 16, 'enc_in': 16,
    'patch_num': 4, 'd_model': 64, 'd_ff': 256,
    'e_layers': 2, 'dropout': 0.1,
    'layer_norm': True, 'pos': True,
    'local_margin': 0.0, 'global_margin': 0.0,
    'loc': True, 'glo': True,
}
model = TimeAlign(config)
import torch
x = torch.randn(2, 96, 16)
y = torch.randn(2, 16, 16)
out, y_recon, align_loss = model(x, y, is_training=True)
print(f'output: {out.shape}, recon: {y_recon.shape}, align: {align_loss.item():.4f}')
print('OK')
"
```

Expected: `output: torch.Size([2, 16, 16]), recon: torch.Size([2, 16, 16]), align: 0.xxxx` + `OK`

---

## Task 10: 配置文件 — `config/timealign.yml`

**Files:**
- Create: `config/timealign.yml`

- [ ] **Step 1: 创建配置文件**

```yaml
# TimeAlign 模型配置

# 模型参数
model: timealign
seq_len: 96
pred_len: 16
enc_in: 16              # 12原始特征 + 4时间编码
patch_num: 4            # seq_len/patch_num=24步/patch
d_model: 64
d_ff: 256
e_layers: 2
dropout: 0.1
layer_norm: true
pos: true
local_margin: 0.0
global_margin: 0.0
loc: true
glo: true

# 损失权重
w_recon: 1.0
w_align: 0.1

# 训练参数
seed: 42
device: cuda
lr: 0.001
batch_size: 32
epochs: 100
patience: 10
optimizer: adamw
scheduler: cosine
grad_clip: 1.0

# 数据路径
data_dir: data/wind_power/processed
target_col: actual_power
```

- [ ] **Step 2: 验证配置可加载**

Run: `python -c "from tool.utils import load_config; c = load_config('config/timealign.yml'); print(c['model'], c['enc_in']); print('OK')"`

Expected: `timealign 16` + `OK`

---

## Task 11: 训练基类 — `trainer/trainer_base.py`

**Files:**
- Create: `trainer/trainer_base.py`

- [ ] **Step 1: 实现 `trainer/trainer_base.py`**

```python
"""
训练基类：定义通用训练流程框架。

子类通过覆写 forward_batch() 实现模型特有逻辑。
"""

import math

import torch
import torch.nn as nn
from torch import optim

from tool.early_stopping import EarlyStopping


class TrainerBase:
    """
    训练基类，提供标准训练流程。

    支持 Adam/AdamW/SGD 优化器和
    CosineAnnealing/ReduceLROnPlateau/OneCycleLR 调度器。
    """

    def __init__(self, model, config, train_loader, val_loader,
                 test_loader, logger, device):
        """
        Parameters
        ----------
        model : nn.Module
            待训练的模型
        config : dict
            训练配置
        train_loader : DataLoader
            训练数据加载器
        val_loader : DataLoader
            验证数据加载器
        test_loader : DataLoader
            测试数据加载器
        logger : Logger
            日志记录器
        device : torch.device
            计算设备
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger
        self.device = device

        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 10)
        self.grad_clip = config.get("grad_clip", 1.0)

        self.optimizer = self._select_optimizer()
        self.scheduler = self._select_scheduler(self.optimizer)
        self.criterion = nn.L1Loss()

        self.train_losses = []
        self.val_losses = []

    def _select_optimizer(self):
        """
        根据配置选择优化器。

        Returns
        -------
        optimizer
        """
        name = self.config.get("optimizer", "adamw").lower()
        lr = self.config.get("lr", 0.001)
        params = self.model.parameters()

        if name == "adam":
            return optim.Adam(params, lr=lr)
        elif name == "adamw":
            return optim.AdamW(params, lr=lr)
        elif name == "sgd":
            return optim.SGD(params, lr=lr, momentum=0.9)
        else:
            return optim.AdamW(params, lr=lr)

    def _select_scheduler(self, optimizer):
        """
        根据配置选择学习率调度器。

        Parameters
        ----------
        optimizer
            优化器实例

        Returns
        -------
        scheduler or None
        """
        name = self.config.get("scheduler", "cosine").lower()
        epochs = self.config.get("epochs", 100)

        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        elif name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        elif name == "onecycle":
            steps = len(self.train_loader) * epochs
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.config.get("lr", 0.001),
                total_steps=steps
            )
        elif name == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )
        return None

    def forward_batch(self, batch, is_training=True):
        """
        前向传播（子类必须覆写）。

        Parameters
        ----------
        batch : tuple
            数据批次
        is_training : bool
            是否训练模式

        Returns
        -------
        loss : torch.Tensor
            总损失
        metrics : dict
            详细指标（如各分项损失）
        """
        raise NotImplementedError

    def train_epoch(self, epoch):
        """
        执行一个 epoch 的训练。

        Parameters
        ----------
        epoch : int
            当前 epoch 编号

        Returns
        -------
        float
            平均训练损失
        """
        self.model.train()
        total_loss = []
        total_metrics = {}

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            loss, metrics = self.forward_batch(batch, is_training=True)

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            self.optimizer.step()

            total_loss.append(loss.item())
            for k, v in metrics.items():
                total_metrics.setdefault(k, [])
                total_metrics[k].append(
                    v if isinstance(v, float) else v
                )

        avg_loss = sum(total_loss) / len(total_loss)
        avg_metrics = {
            k: sum(v) / len(v) for k, v in total_metrics.items()
        }
        return avg_loss, avg_metrics

    def valid(self):
        """
        在验证集上评估。

        Returns
        -------
        float
            平均验证损失
        """
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for batch in self.val_loader:
                loss, _ = self.forward_batch(
                    batch, is_training=False
                )
                total_loss.append(loss.item())

        return sum(total_loss) / len(total_loss)

    def train(self):
        """
        完整训练流程。

        epoch 循环 + 早停 + 调度器 + 日志记录。
        """
        model_name = self.config.get("model", "unknown")
        save_path = f"checkpoints/{model_name}"
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True
        )

        self.logger.log_train("开始训练...")

        for epoch in range(1, self.epochs + 1):
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)

            # 验证
            val_loss = self.valid()

            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 调度器步进
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            # 日志
            metrics_str = " | ".join(
                f"{k}: {v:.6f}" for k, v in train_metrics.items()
            )
            self.logger.log_train(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"LR: {lr:.2e} | {metrics_str}"
            )

            # 早停检查
            early_stopping(val_loss, self.model, save_path)
            if early_stopping.early_stop:
                self.logger.log_train(
                    f"早停触发于 epoch {epoch}"
                )
                break

        # 加载最佳模型
        best_path = f"{save_path}/best_model.pth"
        self.model.load_state_dict(
            torch.load(best_path, weights_only=True)
        )
        self.logger.log_train("已加载最佳模型权重")

    def test(self):
        """
        在测试集上评估。

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            (predictions, ground_truths)
        """
        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch in self.test_loader:
                x_enc, y_enc = batch
                x_enc = x_enc.to(self.device)
                y_enc = y_enc.to(self.device)

                output, _, _ = self.model(
                    x_enc, y_enc, is_training=False
                )

                # 取目标列（第0列）
                preds.append(
                    output[:, :, 0].detach().cpu().numpy()
                )
                trues.append(
                    y_enc[:, :, 0].detach().cpu().numpy()
                )

        import numpy as np
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        return preds, trues
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from trainer.trainer_base import TrainerBase; print('OK')"`

Expected: `OK`

---

## Task 12: TimeAlign 训练器 — `trainer/trainer_timealign.py`

**Files:**
- Create: `trainer/trainer_timealign.py`

- [ ] **Step 1: 实现 `trainer/trainer_timealign.py`**

```python
"""
TimeAlign 训练器：双输入 + 三部分损失。
"""

import torch
import torch.nn as nn

from trainer.trainer_base import TrainerBase


class TrainerTimeAlign(TrainerBase):
    """
    TimeAlign 专用训练器。

    损失 = pred_loss + w_recon × recon_loss + w_align × align_loss
    """

    def __init__(self, model, config, train_loader, val_loader,
                 test_loader, logger, device):
        """
        Parameters
        ----------
        model : nn.Module
            TimeAlign 模型实例
        config : dict
            训练配置，额外需要 w_recon 和 w_align
        """
        super().__init__(
            model, config, train_loader, val_loader,
            test_loader, logger, device
        )
        self.w_recon = config.get("w_recon", 1.0)
        self.w_align = config.get("w_align", 0.1)
        self.pred_len = config["pred_len"]

    def forward_batch(self, batch, is_training=True):
        """
        TimeAlign 前向传播。

        Parameters
        ----------
        batch : tuple(x_enc, y_enc)
            x_enc: [B, seq_len, C]
            y_enc: [B, pred_len, C]
        is_training : bool
            是否训练模式

        Returns
        -------
        loss : torch.Tensor
            总损失
        metrics : dict
            分项损失详情
        """
        x_enc, y_enc = batch
        x_enc = x_enc.to(self.device)
        y_enc = y_enc.to(self.device)

        # 模型前向
        output, y_recon, align_loss = self.model(
            x_enc, y_enc, is_training=is_training
        )

        # 取目标列（第0列）
        pred_target = output[:, -self.pred_len :, 0]
        true_target = y_enc[:, -self.pred_len :, 0]

        # 预测损失
        pred_loss = self.criterion(pred_target, true_target)

        if is_training:
            # 重建损失
            recon_target = y_recon[:, -self.pred_len :, 0]
            recon_loss = self.criterion(recon_target, true_target)

            # 总损失
            loss = (
                pred_loss
                + self.w_recon * recon_loss
                + self.w_align * align_loss
            )

            metrics = {
                "pred_loss": pred_loss.item(),
                "recon_loss": recon_loss.item(),
                "align_loss": (
                    align_loss.item()
                    if torch.is_tensor(align_loss)
                    else align_loss
                ),
            }
        else:
            # 验证/测试时只计算预测损失
            loss = pred_loss
            metrics = {"val_loss": pred_loss.item()}

        return loss, metrics
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from trainer.trainer_timealign import TrainerTimeAlign; print('OK')"`

Expected: `OK`

---

## Task 13: 训练入口 — `train.py`

**Files:**
- Create: `train.py`

- [ ] **Step 1: 实现 `train.py`**

```python
"""
训练入口：加载数据 → 构建模型 → 训练 → 保存。
"""

import argparse

from tool.logger import Logger
from tool.utils import (
    count_parameters,
    get_device,
    load_config,
    set_seed,
)
from tool.visualize import plot_loss_curve


def get_trainer(model_name):
    """
    根据模型名返回对应的训练器类。

    Parameters
    ----------
    model_name : str
        模型名称

    Returns
    -------
    type
        训练器类
    """
    from trainer.trainer_timealign import TrainerTimeAlign

    trainers = {
        "timealign": TrainerTimeAlign,
    }
    if model_name not in trainers:
        raise ValueError(f"未知模型: {model_name}")
    return trainers[model_name]


def get_model(model_name, config):
    """
    根据模型名实例化模型。

    Parameters
    ----------
    model_name : str
        模型名称
    config : dict
        模型配置

    Returns
    -------
    nn.Module
    """
    from model.timealign import TimeAlign

    models = {
        "timealign": TimeAlign,
    }
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}")
    return models[model_name](config)


def main():
    """主训练流程。"""
    parser = argparse.ArgumentParser(description="WPF 训练入口")
    parser.add_argument(
        "--config", type=str, required=True, help="配置文件路径"
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    model_name = config["model"]

    # 2. 设置种子和设备
    set_seed(config.get("seed", 42))
    device = get_device(config)

    # 3. 初始化日志
    log_dir = f"checkpoints/{model_name}"
    logger = Logger(log_dir, model_name)
    logger.log_config(config)

    # 4. 构建数据
    from dataset.wind_power import get_dataloader

    logger.log_feat("加载数据集...")
    train_loader = get_dataloader(config, "train")
    val_loader = get_dataloader(config, "val")
    test_loader = get_dataloader(config, "test")

    logger.log_feat(f"训练集: {len(train_loader.dataset)} 样本")
    logger.log_feat(f"验证集: {len(val_loader.dataset)} 样本")
    logger.log_feat(f"测试集: {len(test_loader.dataset)} 样本")

    # 5. 构建模型
    model = get_model(model_name, config)
    n_params = count_parameters(model)
    logger.log_train(f"模型: {model_name}, 参数量: {n_params:,}")

    # 6. 构建训练器
    TrainerClass = get_trainer(model_name)
    trainer = TrainerClass(
        model, config, train_loader, val_loader,
        test_loader, logger, device
    )

    # 7. 训练
    trainer.train()

    # 8. 绘制 loss 曲线
    fig_path = f"figures/{model_name}/loss_curve.png"
    plot_loss_curve(trainer.train_losses, trainer.val_losses, fig_path)
    logger.log_train(f"Loss 曲线保存至: {fig_path}")

    # 9. 关闭日志
    logger.close()
    print("训练完成。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证入口可导入**

Run: `python -c "import train; print('OK')"`

Expected: `OK`

---

## Task 14: 测试入口 — `test.py`

**Files:**
- Create: `test.py`

- [ ] **Step 1: 实现 `test.py`**

```python
"""
测试入口：加载模型 → 推理 → 评估 → 可视化。
"""

import argparse

import numpy as np
import torch

from tool.logger import Logger
from tool.metrics import (
    calculate_all_step_metrics,
    calculate_metrics,
    calculate_step_metrics,
)
from tool.utils import get_device, load_config, set_seed
from tool.visualize import plot_prediction, plot_step_accuracy


def get_model(model_name, config):
    """根据模型名实例化模型。"""
    from model.timealign import TimeAlign

    models = {
        "timealign": TimeAlign,
    }
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}")
    return models[model_name](config)


def main():
    """主测试流程。"""
    parser = argparse.ArgumentParser(description="WPF 测试入口")
    parser.add_argument(
        "--config", type=str, required=True, help="配置文件路径"
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    model_name = config["model"]

    # 2. 设置种子和设备
    set_seed(config.get("seed", 42))
    device = get_device(config)

    # 3. 初始化日志
    log_dir = f"checkpoints/{model_name}"
    logger = Logger(log_dir, model_name)

    # 4. 构建数据
    from dataset.wind_power import get_dataloader

    test_loader = get_dataloader(config, "test")
    logger.log_eval(f"测试集: {len(test_loader.dataset)} 样本")

    # 5. 加载模型
    model = get_model(model_name, config)
    weight_path = f"checkpoints/{model_name}/best_model.pth"
    model.load_state_dict(
        torch.load(weight_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    logger.log_eval(f"已加载模型: {weight_path}")

    # 6. 推理
    preds = []
    trues = []
    with torch.no_grad():
        for batch in test_loader:
            x_enc, y_enc = batch
            x_enc = x_enc.to(device)
            y_enc = y_enc.to(device)

            output, _, _ = model(
                x_enc, y_enc, is_training=False
            )

            # 目标列（第0列）
            preds.append(output[:, :, 0].cpu().numpy())
            trues.append(y_enc[:, :, 0].cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    logger.log_eval(f"预测结果: {preds.shape}")

    # 7. 计算指标
    overall = calculate_metrics(preds, trues)
    logger.log_eval("--- 整体指标 ---")
    logger.log_eval(f"  MAE: {overall['mae']:.4f}")
    logger.log_eval(f"  MSE: {overall['mse']:.4f}")
    logger.log_eval(f"  RMSE: {overall['rmse']:.4f}")
    logger.log_eval(f"  MAPE: {overall['mape']:.2f}%")
    logger.log_eval(f"  准确率: {overall['accuracy']:.2f}%")
    logger.log_eval(f"  合格率: {overall['qualification_rate']:.2f}%")

    # 第16步（最后一步）指标
    step_16 = calculate_step_metrics(preds, trues, step=15)
    logger.log_eval("--- 第16步 (4h ahead) ---")
    logger.log_eval(f"  准确率: {step_16['accuracy']:.2f}%")
    logger.log_eval(f"  合格率: {step_16['qualification_rate']:.2f}%")

    # 8. 可视化
    fig_dir = f"figures/{model_name}"
    plot_prediction(preds, trues, f"{fig_dir}/prediction.png")
    plot_step_accuracy(
        preds, trues, 70000, f"{fig_dir}/step_accuracy.png"
    )
    logger.log_eval(f"可视化图表保存至: {fig_dir}/")

    # 9. 关闭日志
    logger.close()
    print("测试完成。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证入口可导入**

Run: `python -c "import test; print('OK')"`

Expected: `OK`

---

## Task 15: 端到端集成验证

**Files:** 无新增

- [ ] **Step 1: 运行预处理**

Run: `python preprocess.py`

Expected: 正常完成，`data/wind_power/processed/` 下生成三个文件。

- [ ] **Step 2: 运行训练（短验证，1 epoch）**

临时修改 `config/timealign.yml` 中 epochs 为 1，然后：

Run: `python train.py --config config/timealign.yml`

Expected:
- 日志输出包含 `[FEAT]` 和 `[TRAIN]` 标记
- 生成 `checkpoints/timealign/best_model.pth`
- 生成 `figures/timealign/loss_curve.png`
- 无报错

- [ ] **Step 3: 运行测试**

Run: `python test.py --config config/timealign.yml`

Expected:
- 日志输出包含 `[EVAL]` 标记
- 输出整体指标和第16步指标
- 生成 `figures/timealign/prediction.png` 和 `step_accuracy.png`
- 无报错

- [ ] **Step 4: 恢复配置**

将 `config/timealign.yml` 中 epochs 改回 100。
