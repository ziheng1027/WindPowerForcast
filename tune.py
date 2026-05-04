"""
Optuna 超参调优：贝叶斯搜索 MeteoPowerNet 最优配置。

使用 TPE 采样器，以最佳验证 loss 为目标函数。
"""

import argparse
import gc
import json
import os
import time

import optuna
import torch
import yaml

from train import train_and_evaluate
from tool.utils import set_seed


# ========== 搜索空间 ==========
def suggest_params(trial):
    """
    定义 MeteoPowerNet 的超参搜索空间。

    Parameters
    ----------
    trial : optuna.Trial
        当前试验

    Returns
    -------
    dict
        采样的超参配置
    """
    # --- 模型结构 ---
    hidden_size = trial.suggest_categorical(
        "hidden_size", [32, 64, 128]
    )

    # TCN 通道数: 选基础宽度 × 深度组合
    tcn_depth = trial.suggest_int("tcn_depth", 2, 4)
    tcn_base = trial.suggest_categorical(
        "tcn_base", [32, 64]
    )
    tcn_channels = [tcn_base] * tcn_depth

    # 未来 GFS TCN
    fut_depth = trial.suggest_int("fut_depth", 1, 3)
    fut_base = trial.suggest_categorical(
        "fut_base", [32, 64]
    )
    tcn_future_channels = [fut_base] * fut_depth

    kernel_size = trial.suggest_categorical(
        "kernel_size", [3, 5, 7]
    )

    d_attn = trial.suggest_categorical(
        "d_attn", [32, 64, 128]
    )

    dropout = trial.suggest_float(
        "dropout", 0.1, 0.4, step=0.1
    )

    # --- 训练策略 ---
    lr = trial.suggest_float(
        "lr", 1e-4, 3e-3, log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128]
    )

    patience = trial.suggest_categorical(
        "patience", [10, 15, 20]
    )

    return {
        "model": "meteo_power",
        "seq_len": 96,
        "pred_len": 16,
        "enc_in": 23,
        "dec_out": 17,
        "hidden_size": hidden_size,
        "tcn_channels": tcn_channels,
        "tcn_future_channels": tcn_future_channels,
        "kernel_size": kernel_size,
        "d_attn": d_attn,
        "dropout": dropout,
        "seed": 42,
        "device": "cuda",
        "lr": lr,
        "batch_size": batch_size,
        "epochs": 100,
        "patience": patience,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "grad_clip": 1.0,
        "dataset_type": "meteo",
        "data_dir": "data/wind_power/processed",
        "target_col": "actual_power",
    }


# ========== 目标函数 ==========
def objective(trial):
    """
    Optuna 目标函数：训练并返回最佳验证 loss。

    Parameters
    ----------
    trial : optuna.Trial
        当前试验

    Returns
    -------
    float
        最佳验证 loss（越小越好）
    """
    config = suggest_params(trial)

    # 记录试验参数
    trial_id = trial.number
    print(f"\n{'='*50}")
    print(f"Trial {trial_id}")
    print(f"  hidden={config['hidden_size']}, "
          f"tcn={config['tcn_channels']}, "
          f"fut={config['tcn_future_channels']}")
    print(f"  kernel={config['kernel_size']}, "
          f"d_attn={config['d_attn']}, "
          f"dropout={config['dropout']}")
    print(f"  lr={config['lr']:.6f}, "
          f"bs={config['batch_size']}, "
          f"patience={config['patience']}")

    t0 = time.time()

    try:
        best_loss, trainer = train_and_evaluate(
            config, silent=True
        )
    except RuntimeError as e:
        # CUDA OOM: 剪枝该试验
        if "out of memory" in str(e):
            print(f"  [OOM] CUDA OOM, pruned")
            torch.cuda.empty_cache()
            gc.collect()
            return float("inf")
        raise

    elapsed = time.time() - t0
    n_params = sum(
        p.numel()
        for p in trainer.model.parameters()
    )

    # 记录辅助指标
    trial.set_user_attr("n_params", n_params)
    trial.set_user_attr("elapsed_sec", round(elapsed))
    trial.set_user_attr(
        "best_epoch",
        len(trainer.valid_losses)
        - trainer.valid_losses[::-1].index(
            min(trainer.valid_losses)
        )
    )

    print(f"  [OK] loss={best_loss:.6f}, "
          f"params={n_params:,}, "
          f"time={elapsed:.0f}s")

    # 清理 GPU 显存
    del trainer, config
    torch.cuda.empty_cache()
    gc.collect()

    return best_loss


# ========== 回调 ==========
def print_best_callback(study, trial):
    """每次试验后打印当前最佳结果。"""
    best = study.best_trial
    print(f"\n[BEST] 当前最佳 (Trial {best.number}):")
    print(f"   loss = {best.value:.6f}")
    print(f"   params = {best.params}")


# ========== 主函数 ==========
def main():
    """调参主流程。"""
    parser = argparse.ArgumentParser(
        description="MeteoPowerNet 超参调优"
    )
    parser.add_argument(
        "--n_trials", type=int, default=30,
        help="调参轮数 (默认 30)"
    )
    parser.add_argument(
        "--study_name", type=str,
        default="meteo_power_tune",
        help="Optuna study 名称"
    )
    args = parser.parse_args()

    # 结果目录
    os.makedirs("output/tune", exist_ok=True)

    print("=" * 60)
    print("MeteoPowerNet 超参调优 (Optuna TPE)")
    print(f"  目标: 最小化验证 loss")
    print(f"  试验轮数: {args.n_trials}")
    print("=" * 60)

    # 创建 study（内存存储，无需 sqlalchemy）
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42, n_startup_trials=5
        ),
    )

    # 已完成的 trial 数
    n_done = len(study.trials)
    n_remaining = max(0, args.n_trials - n_done)
    print(f"  已完成: {n_done}, 剩余: {n_remaining}\n")

    if n_remaining == 0:
        print("已达目标轮数，跳过。")
    else:
        study.optimize(
            objective,
            n_trials=n_remaining,
            callbacks=[print_best_callback],
            show_progress_bar=False,
        )

    # 输出最佳结果
    best = study.best_trial
    print("\n" + "=" * 60)
    print("[DONE] 调参完成，最佳结果:")
    print(f"  Trial #{best.number}")
    print(f"  Valid Loss: {best.value:.6f}")
    print(f"  参数量: {best.user_attrs['n_params']:,}")
    print(f"  耗时: {best.user_attrs['elapsed_sec']}s")
    print(f"  Best Epoch: "
          f"{best.user_attrs['best_epoch']}")
    print("\n  最佳超参:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # 保存最佳配置为 YAML
    best_config = suggest_params_from_best(best)
    save_path = "output/tune/best_config.yml"
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(
            best_config, f, allow_unicode=True,
            default_flow_style=False, sort_keys=False,
        )
    print(f"\n  最佳配置已保存: {save_path}")

    # 保存调参报告
    report_path = "output/tune/tune_report.json"
    report = {
        "best_trial": best.number,
        "best_loss": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "n_trials": len(study.trials),
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  完整报告已保存: {report_path}")
    print("=" * 60)


def suggest_params_from_best(best_trial):
    """
    从 Optuna best trial 重建完整配置 dict。

    Parameters
    ----------
    best_trial : optuna.FrozenTrial
        最佳试验

    Returns
    -------
    dict
        完整的模型配置
    """
    p = best_trial.params
    tcn_depth = p["tcn_depth"]
    tcn_base = p["tcn_base"]
    fut_depth = p["fut_depth"]
    fut_base = p["fut_base"]

    return {
        "model": "meteo_power",
        "seq_len": 96,
        "pred_len": 16,
        "enc_in": 23,
        "dec_out": 17,
        "hidden_size": p["hidden_size"],
        "tcn_channels": [tcn_base] * tcn_depth,
        "tcn_future_channels": [fut_base] * fut_depth,
        "kernel_size": p["kernel_size"],
        "d_attn": p["d_attn"],
        "dropout": p["dropout"],
        "seed": 42,
        "device": "cuda",
        "lr": p["lr"],
        "batch_size": p["batch_size"],
        "epochs": 100,
        "patience": p["patience"],
        "optimizer": "adamw",
        "scheduler": "cosine",
        "grad_clip": 1.0,
        "dataset_type": "meteo",
        "data_dir": "data/wind_power/processed",
        "target_col": "actual_power",
    }


if __name__ == "__main__":
    main()
