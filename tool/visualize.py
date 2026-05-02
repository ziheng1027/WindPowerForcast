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
