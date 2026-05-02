"""
测试入口：加载模型 → 推理 → 评估 → 可视化。
"""

import argparse

import numpy as np
import pandas as pd
import torch

from tool.logger import Logger
from tool.metrics import (
    calculate_all_step_metrics,
    calculate_metrics,
    calculate_monthly_metrics,
    calculate_step_metrics,
)
from tool.utils import get_device, load_config, set_seed
from tool.visualize import plot_prediction, plot_step_accuracy


def get_model(model_name, config):
    """根据模型名实例化模型。"""
    from model.timealign import TimeAlign
    from model.itransformer import ITransformer
    from model.patchtst import PatchTST
    from model.dlinear import DLinear
    from model.lstm import LSTMForecaster

    models = {
        "timealign": TimeAlign,
        "itransformer": ITransformer,
        "patchtst": PatchTST,
        "dlinear": DLinear,
        "lstm": LSTMForecaster,
        "bilstm": LSTMForecaster,
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
    log_dir = f"output/log/{model_name}"
    logger = Logger(log_dir, model_name)

    # 4. 构建数据
    from dataset.wind_power import get_dataloader

    test_loader = get_dataloader(config, "test")
    logger.log_eval(f"测试集: {len(test_loader.dataset)} 样本")

    # 5. 加载模型
    model = get_model(model_name, config)
    weight_path = f"output/checkpoint/{model_name}/best_model.pth"
    model.load_state_dict(
        torch.load(weight_path, map_location=device,
                   weights_only=True)
    )
    model = model.to(device)
    model.eval()
    logger.log_eval(f"已加载模型: {weight_path}")

    # 6. 推理
    preds = []
    trues = []
    all_months = []
    with torch.no_grad():
        for batch in test_loader:
            x_enc, y_enc, months = batch
            x_enc = x_enc.to(device)
            y_enc = y_enc.to(device)

            output, _, _ = model(
                x_enc, y_enc, is_training=False
            )

            # 目标列（第0列）
            preds.append(output[:, :, 0].cpu().numpy())
            trues.append(y_enc[:, :, 0].cpu().numpy())
            all_months.append(months.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    all_months = np.concatenate(all_months, axis=0)
    logger.log_eval(f"预测结果: {preds.shape}")

    # 7. 反归一化至 kW（指标需在真实尺度计算）
    import json
    import os

    data_dir = config.get("data_dir", "data/wind_power/processed")
    with open(os.path.join(data_dir, "norm_params.json"),
              "r", encoding="utf-8") as f:
        norm_params = json.load(f)
    p_min = norm_params["actual_power"]["min"]
    p_max = norm_params["actual_power"]["max"]
    scale = p_max - p_min

    preds_kw = preds * scale + p_min
    trues_kw = trues * scale + p_min

    # 8. 计算指标（kW 尺度）
    overall = calculate_metrics(preds_kw, trues_kw)
    logger.log_eval("--- 整体指标 ---")
    logger.log_eval(f"  MAE: {overall['mae']:.2f} kW")
    logger.log_eval(f"  RMSE: {overall['rmse']:.2f} kW")
    logger.log_eval(f"  R: {overall['r']:.4f}")

    # 按月统计准确率（GBT+40607-2021）
    monthly_result = calculate_monthly_metrics(
        preds_kw, trues_kw, all_months
    )
    logger.log_eval("--- 月度指标 ---")
    for m, metrics in monthly_result["monthly"].items():
        logger.log_eval(
            f"  {m}月: Acc={metrics['accuracy']:.2f}%, "
            f"Qual={metrics['qualification_rate']:.2f}%, "
            f"RMSE={metrics['rmse']:.2f} kW, "
            f"R={metrics['r']:.4f}, "
            f"N={metrics['n_points']}"
        )
    logger.log_eval(
        f"  月均准确率: "
        f"{monthly_result['monthly_avg_accuracy']:.2f}%"
    )
    logger.log_eval(
        f"  月均合格率: "
        f"{monthly_result['monthly_avg_qualification_rate']:.2f}%"
    )

    step_16 = calculate_step_metrics(preds_kw, trues_kw, step=15)
    logger.log_eval("--- 第16步 (4h ahead) ---")
    logger.log_eval(f"  准确率: {step_16['accuracy']:.2f}%")
    logger.log_eval(f"  合格率: {step_16['qualification_rate']:.2f}%")

    # 9. 可视化（kW 尺度）
    fig_dir = f"output/figure/{model_name}"
    plot_prediction(
        preds_kw, trues_kw, f"{fig_dir}/prediction.png"
    )
    plot_step_accuracy(
        preds_kw, trues_kw, 70000,
        f"{fig_dir}/step_accuracy.png"
    )
    logger.log_eval(f"可视化图表保存至: {fig_dir}/")

    # 10. 导出预测结果 CSV
    pred_len = preds_kw.shape[1]
    result_rows = []
    for i in range(len(preds_kw)):
        for step in range(pred_len):
            result_rows.append({
                "sample": i,
                "step": step + 1,
                "actual_power(kW)": f"{trues_kw[i, step]:.2f}",
                "pred_power(kW)": f"{preds_kw[i, step]:.2f}",
            })

    result_df = pd.DataFrame(result_rows)
    result_dir = f"output/result/{model_name}"
    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/prediction.csv"
    result_df.to_csv(result_path, index=False,
                     encoding="utf-8-sig")
    logger.log_eval(f"预测结果保存至: {result_path}")

    # 10. 关闭日志
    logger.close()
    print("测试完成。")


if __name__ == "__main__":
    main()
