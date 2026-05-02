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
        包含 mae, mse, rmse, r, accuracy, qualification_rate
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

    # Pearson 相关系数 R
    preds_flat = preds.flatten()
    trues_flat = trues.flatten()
    if np.std(preds_flat) < 1e-8 or np.std(trues_flat) < 1e-8:
        r = 0.0
    else:
        r = float(np.corrcoef(preds_flat, trues_flat)[0, 1])

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
        "r": r,
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
    n_steps = preds.shape[1]

    results = []
    for s in range(n_steps):
        result = calculate_step_metrics(preds, trues, s, capacity)
        results.append(result)
    return results


def calculate_monthly_metrics(preds, trues, months, capacity=CAPACITY):
    """
    按月计算评估指标，返回各月指标及月均准确率。

    GBT+40607-2021 要求准确率按月统计后取平均。

    Parameters
    ----------
    preds : np.ndarray
        预测值，shape [N, pred_len]
    trues : np.ndarray
        真实值，shape [N, pred_len]
    months : np.ndarray
        各样本各步的月份，shape [N, pred_len]
    capacity : float
        装机容量

    Returns
    -------
    dict
        monthly: {月: 指标dict}, monthly_avg_accuracy,
        monthly_avg_qualification_rate
    """
    unique_months = sorted(np.unique(months))

    monthly = {}
    acc_list = []
    qual_list = []

    for m in unique_months:
        # 该月覆盖的 (样本, 步) 位置
        mask = months == m
        if mask.sum() == 0:
            continue

        m_preds = preds[mask]
        m_trues = trues[mask]

        error = m_preds - m_trues
        abs_error = np.abs(error)

        mae = float(np.mean(abs_error))
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))

        # 相关系数
        if (np.std(m_preds) < 1e-8
                or np.std(m_trues) < 1e-8):
            r = 0.0
        else:
            r = float(np.corrcoef(
                m_preds.flatten(), m_trues.flatten()
            )[0, 1])

        accuracy = (1 - rmse / capacity) * 100
        threshold = 0.25 * capacity
        qualification_rate = float(
            np.mean(abs_error <= threshold)
        ) * 100

        monthly[int(m)] = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r": r,
            "accuracy": accuracy,
            "qualification_rate": qualification_rate,
            "n_points": int(mask.sum()),
        }
        acc_list.append(accuracy)
        qual_list.append(qualification_rate)

    return {
        "monthly": monthly,
        "monthly_avg_accuracy": float(np.mean(acc_list)),
        "monthly_avg_qualification_rate": float(np.mean(qual_list)),
    }
