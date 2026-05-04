"""
测试入口：加载模型 → 推理 → 评估 → 可视化 + 导出。
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from tool.logger import Logger
from tool.metrics import (
    CAPACITY,
    calculate_global_metrics,
    calculate_gbt_metrics,
)
from tool.utils import get_device, load_config, set_seed
from tool.visualize import plot_step_accuracy, plot_step_series


def get_model(model_name, config):
    """根据模型名实例化模型。"""
    from model.timealign import TimeAlign
    from model.itransformer import ITransformer
    from model.patchtst import PatchTST
    from model.dlinear import DLinear
    from model.lstm import LSTMForecaster
    from model.tcn import TCNForecaster
    from model.meteo_power import MeteoPowerNet

    models = {
        "timealign": TimeAlign,
        "meteo_power": MeteoPowerNet,
        "itransformer": ITransformer,
        "patchtst": PatchTST,
        "dlinear": DLinear,
        "lstm": LSTMForecaster,
        "bilstm": LSTMForecaster,
        "tcn": TCNForecaster,
    }
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}")
    return models[model_name](config)


def _log_gbt(logger, gbt):
    """格式化输出国标指标。"""
    for ym, m in gbt["monthly"].items():
        logger.log_eval(
            f"  {ym}: Acc={m['accuracy']:.2f}%, "
            f"Qual={m['qualification_rate']:.2f}%, "
            f"R={m['r']:.4f}, "
            f"N_days={m['n_days']}"
        )
    logger.log_eval(
        f"  月均准确率: "
        f"{gbt['monthly_avg_accuracy']:.2f}%"
    )
    logger.log_eval(
        f"  月均合格率: "
        f"{gbt['monthly_avg_qualification_rate']:.2f}%"
    )
    logger.log_eval(
        f"  月均相关系数: "
        f"{gbt['monthly_avg_r']:.4f}"
    )


def _reconstruct_datetime(dates, minutes):
    """将日期编码 + 分钟数重构为可读时间戳。"""
    d = int(dates)
    m = int(minutes)
    return pd.Timestamp(
        year=d // 10000,
        month=(d // 100) % 100,
        day=d % 100,
        hour=m // 60,
        minute=m % 60,
    )


def _export_step_report(
    preds, trues, dates, minutes, step,
    capacity, result_dir
):
    """
    导出指定步长的逐点明细 CSV 和日报汇总 CSV。

    Parameters
    ----------
    preds : np.ndarray, shape [N, pred_len]
    trues : np.ndarray
    dates : np.ndarray, shape [N, pred_len]
    minutes : np.ndarray, shape [N, pred_len]
    step : int
        步长索引 (0-based)
    capacity : float
    result_dir : str
    """
    os.makedirs(result_dir, exist_ok=True)

    p = preds[:, step]
    t = trues[:, step]
    d = dates[:, step]
    m = minutes[:, step]
    tag = step + 1

    # ---- 逐点明细 ----
    threshold = 0.25 * capacity
    detail_rows = []
    for i in range(len(p)):
        dt = _reconstruct_datetime(d[i], m[i])
        abs_err = abs(float(p[i] - t[i]))
        detail_rows.append({
            "datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "date": dt.strftime("%Y-%m-%d"),
            "pred_power(kW)": round(float(p[i]), 2),
            "actual_power(kW)": round(float(t[i]), 2),
            "abs_error(kW)": round(abs_err, 2),
            "qualified": abs_err <= threshold,
        })

    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(
        result_dir, f"step{tag}_detail.csv"
    )
    detail_df.to_csv(
        detail_path, index=False, encoding="utf-8-sig"
    )

    # ---- 日报汇总 ----
    daily_groups = detail_df.groupby("date")
    daily_rows = []
    for date_str, group in daily_groups:
        p_day = group["pred_power(kW)"].values
        t_day = group["actual_power(kW)"].values
        error = p_day - t_day
        rmse = float(np.sqrt(np.mean(error ** 2)))
        accuracy = (1 - rmse / capacity) * 100
        qual_rate = group["qualified"].mean() * 100

        if np.std(p_day) < 1e-8 or np.std(t_day) < 1e-8:
            r = 0.0
        else:
            r = float(np.corrcoef(p_day, t_day)[0, 1])

        daily_rows.append({
            "date": date_str,
            "accuracy(%)": round(accuracy, 2),
            "qualification_rate(%)": round(qual_rate, 2),
            "r": round(r, 4),
            "n_points": len(group),
        })

    daily_df = pd.DataFrame(daily_rows)
    daily_path = os.path.join(
        result_dir, f"step{tag}_daily.csv"
    )
    daily_df.to_csv(
        daily_path, index=False, encoding="utf-8-sig"
    )

    return detail_path, daily_path


def main():
    """主测试流程。"""
    parser = argparse.ArgumentParser(description="WPF 测试入口")
    parser.add_argument(
        "--config", type=str, required=True, help="配置文件路径"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]

    set_seed(config.get("seed", 42))
    device = get_device(config)

    log_dir = f"output/log/{model_name}"
    logger = Logger(log_dir, model_name)

    # 构建数据
    dataset_type = config.get("dataset_type", "default")
    if dataset_type == "meteo":
        from dataset.meteo_power import get_meteo_dataloader
        test_loader = get_meteo_dataloader(config, "test")
    else:
        from dataset.wind_power import get_dataloader
        test_loader = get_dataloader(config, "test")
    logger.log_eval(f"测试集: {len(test_loader.dataset)} 样本")

    # 加载模型
    model = get_model(model_name, config)
    weight_path = f"output/checkpoint/{model_name}/best_model.pth"
    model.load_state_dict(
        torch.load(weight_path, map_location=device,
                   weights_only=True)
    )
    model = model.to(device)
    model.eval()
    logger.log_eval(f"已加载模型: {weight_path}")

    # 推理
    preds = []
    trues = []
    all_dates = []
    all_minutes = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 8:
                # MeteoPowerDataset: 8-tuple
                (power_h, gfs_h, tower_h,
                 gfs_fut, y_enc, _, dates,
                 mins) = batch
                power_h = power_h.to(device)
                gfs_h = gfs_h.to(device)
                tower_h = tower_h.to(device)
                gfs_fut = gfs_fut.to(device)
                y_enc = y_enc.to(device)
                output, _, _ = model(
                    power_h, gfs_h, tower_h,
                    gfs_fut, is_training=False,
                )
            else:
                # WindPowerDataset: 5-tuple
                x_enc, y_enc, _, dates, mins = batch
                x_enc = x_enc.to(device)
                y_enc = y_enc.to(device)
                output, _, _ = model(
                    x_enc, y_enc, is_training=False
                )

            preds.append(output[:, :, 0].cpu().numpy())
            trues.append(y_enc[:, :, 0].cpu().numpy())
            all_dates.append(dates.numpy())
            all_minutes.append(mins.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    all_dates = np.concatenate(all_dates, axis=0)
    all_minutes = np.concatenate(all_minutes, axis=0)
    logger.log_eval(f"预测结果: {preds.shape}")

    # 反归一化至 kW
    data_dir = config.get("data_dir", "data/wind_power/processed")
    with open(os.path.join(data_dir, "norm_params.json"),
              "r", encoding="utf-8") as f:
        norm_params = json.load(f)
    p_min = norm_params["actual_power"]["min"]
    p_max = norm_params["actual_power"]["max"]
    scale = p_max - p_min

    preds_kw = preds * scale + p_min
    trues_kw = trues * scale + p_min

    # 物理约束: 功率 ∈ [0, Cap]
    preds_kw = np.clip(preds_kw, 0, CAPACITY)

    # ---- 全局指标 (RMSE, MAE) ----
    global_m = calculate_global_metrics(preds_kw, trues_kw)
    logger.log_eval("--- 全局指标 ---")
    logger.log_eval(f"  RMSE: {global_m['rmse']:.2f} kW")
    logger.log_eval(f"  MAE: {global_m['mae']:.2f} kW")

    # ---- 国标指标: 月均第 4 小时 ----
    gbt_16 = calculate_gbt_metrics(
        preds_kw, trues_kw, all_dates, step=15
    )
    logger.log_eval(
        "--- 国标指标 (月均第 16 步 / 4h, 日→月) ---"
    )
    _log_gbt(logger, gbt_16)

    # ---- 国标指标: 月均第 1 步 ----
    gbt_01 = calculate_gbt_metrics(
        preds_kw, trues_kw, all_dates, step=0
    )
    logger.log_eval(
        "--- 国标指标 (月均第 1 步 / 15min, 日→月) ---"
    )
    _log_gbt(logger, gbt_01)

    # ---- 可视化 ----
    fig_dir = f"output/figure/{model_name}"
    plot_step_accuracy(
        preds_kw, trues_kw, CAPACITY,
        f"{fig_dir}/step_accuracy.png"
    )
    plot_step_series(
        preds_kw, trues_kw, step=0,
        save_path=f"{fig_dir}/step1_series.png",
        dates=all_dates, minutes=all_minutes,
        title="第 1 步预测 vs 真值 (15min ahead)",
    )
    plot_step_series(
        preds_kw, trues_kw, step=15,
        save_path=f"{fig_dir}/step16_series.png",
        dates=all_dates, minutes=all_minutes,
        title="第 16 步预测 vs 真值 (4h ahead)",
    )
    logger.log_eval(f"可视化图表保存至: {fig_dir}/")

    # ---- 导出 CSV ----
    result_dir = f"output/result/{model_name}"

    for step, label in [(0, "第1步"), (15, "第16步")]:
        d_path, day_path = _export_step_report(
            preds_kw, trues_kw, all_dates,
            all_minutes, step, CAPACITY, result_dir
        )
        logger.log_eval(
            f"  {label}明细: {d_path}"
        )
        logger.log_eval(
            f"  {label}日报: {day_path}"
        )
    logger.log_eval("\n\n")
    logger.close()
    print("测试完成。")


if __name__ == "__main__":
    main()
