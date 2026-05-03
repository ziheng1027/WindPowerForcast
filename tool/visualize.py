"""
可视化工具：loss 曲线、步长准确率、单步时序对比。
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_loss_curve(train_losses, valid_losses, save_path):
    """
    绘制训练/验证 loss 曲线。

    Parameters
    ----------
    train_losses : list[float]
        每 epoch 训练损失
    valid_losses : list[float]
        每 epoch 验证损失
    save_path : str
        图片保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, valid_losses, label="Valid Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练/验证损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_step_accuracy(preds, trues, capacity, save_path):
    """
    绘制各步长的准确率和合格率 (全局平均)。

    Parameters
    ----------
    preds : np.ndarray, shape [N, pred_len]
    trues : np.ndarray
    capacity : float
    save_path : str
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if preds.ndim == 3:
        preds = preds[:, :, 0]
        trues = trues[:, :, 0]

    n_steps = preds.shape[1]
    steps = []
    accuracies = []
    qual_rates = []

    for s in range(n_steps):
        p_s = preds[:, s]
        t_s = trues[:, s]
        error = p_s - t_s
        rmse = float(np.sqrt(np.mean(error ** 2)))
        acc = (1 - rmse / capacity) * 100
        threshold = 0.25 * capacity
        qual = float(np.mean(np.abs(error) <= threshold)) * 100

        steps.append(s + 1)
        accuracies.append(acc)
        qual_rates.append(qual)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("预测步长")
    ax1.set_ylabel("准确率 (%)", color=color1)
    ax1.plot(
        steps, accuracies, color=color1,
        marker="o", label="准确率"
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("合格率 (%)", color=color2)
    ax2.plot(
        steps, qual_rates, color=color2,
        marker="s", label="合格率"
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("各步长准确率与合格率")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _to_timestamps(dates, minutes):
    """日期编码 + 分钟数 → pd.DatetimeIndex。"""
    d = dates.astype(int)
    m = minutes.astype(int)
    years = d // 10000
    months = (d // 100) % 100
    days = d % 100
    hours = m // 60
    mins = m % 60
    return pd.to_datetime(
        dict(
            year=years, month=months, day=days,
            hour=hours, minute=mins,
        )
    )


def plot_step_series(
    preds, trues, step, save_path,
    dates=None, minutes=None, title=None,
):
    """
    绘制指定步长的逐点预测时序对比图。

    提供 dates/minutes 时使用真实时间戳做 X 轴;
    否则使用样本序号。

    Parameters
    ----------
    preds : np.ndarray, shape [N, pred_len] 或 [N, pred_len, C]
    trues : np.ndarray
    step : int
        步长索引 (0-based)
    save_path : str
    dates : np.ndarray or None
        shape [N, pred_len], YYYYMMDD 日期编码
    minutes : np.ndarray or None
        shape [N, pred_len], 一天内分钟数
    title : str or None
    """
    import matplotlib.dates as mdates

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if preds.ndim == 3:
        p = preds[:, step, 0]
        t = trues[:, step, 0]
    else:
        p = preds[:, step]
        t = trues[:, step]

    has_time = (
        dates is not None and minutes is not None
    )

    if has_time:
        d = dates[:, step]
        m = minutes[:, step]
        x_axis = _to_timestamps(d, m)
    else:
        x_axis = np.arange(len(p))

    fig, ax = plt.subplots(figsize=(24, 6))
    ax.plot(
        x_axis, t,
        label="真实值", linewidth=0.8, alpha=0.9
    )
    ax.plot(
        x_axis, p,
        label="预测值", linewidth=0.8,
        linestyle="--", alpha=0.7
    )

    if has_time:
        ax.xaxis.set_major_locator(
            mdates.DayLocator()
        )
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d")
        )
        ax.xaxis.set_minor_locator(
            mdates.HourLocator(byhour=[0, 6, 12, 18])
        )
        fig.autofmt_xdate(rotation=0, ha="center")

    ax.set_xlabel("时间" if has_time else "样本序号")
    ax.set_ylabel("功率 (kW)")
    title = title or f"第 {step + 1} 步预测 vs 真值"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
