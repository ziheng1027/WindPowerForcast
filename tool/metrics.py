"""
评估指标：全局 RMSE/MAE + 国网 GBT+40607-2021 标准指标。

国网考核：月均第 4 小时（第 16 步）准确率、合格率、相关系数。
计算流程：按天求日均值 → 按月对日均值再取平均。
"""

import numpy as np


# 装机容量 (14台 × 5000kW, EN-171/5.0)
CAPACITY = 70000.0

# 考核步长 (0-based, 第 16 步 = 4h ahead)
EVAL_STEP = 15


def calculate_global_metrics(preds, trues, capacity=CAPACITY):
    """
    计算全局均值指标 RMSE 和 MAE。

    Parameters
    ----------
    preds : np.ndarray, shape [N] 或 [N, pred_len]
    trues : np.ndarray, 同 shape
    capacity : float

    Returns
    -------
    dict : rmse, mae
    """
    preds = np.asarray(preds).flatten()
    trues = np.asarray(trues).flatten()

    error = preds - trues
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    return {"rmse": rmse, "mae": mae}


def _step16_data(preds, trues, step=EVAL_STEP):
    """提取第 16 步 (4h ahead) 的预测与真值。"""
    if preds.ndim == 3:
        p = preds[:, step, 0]
        t = trues[:, step, 0]
    elif preds.ndim == 2:
        p = preds[:, step]
        t = trues[:, step]
    else:
        raise ValueError(
            f"preds 应为 2D 或 3D, 收到 {preds.ndim}D"
        )
    return p, t


def _daily_accuracy(preds_day, trues_day, capacity):
    """计算单日的准确率: (1 - RMSE/Cap) × 100%"""
    error = preds_day - trues_day
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return (1 - rmse / capacity) * 100


def _daily_qualification_rate(preds_day, trues_day, capacity):
    """计算单日的合格率: |误差| ≤ 25%Cap 的占比 × 100%"""
    abs_error = np.abs(preds_day - trues_day)
    threshold = 0.25 * capacity
    return float(np.mean(abs_error <= threshold)) * 100


def _daily_correlation(preds_day, trues_day):
    """计算单日的 Pearson 相关系数 R。"""
    if (np.std(preds_day) < 1e-8
            or np.std(trues_day) < 1e-8):
        return 0.0
    return float(np.corrcoef(preds_day, trues_day)[0, 1])


def calculate_gbt_metrics(
    preds, trues, dates, capacity=CAPACITY, step=EVAL_STEP
):
    """
    按国标 GBT+40607-2021 计算月均第 4 小时指标。

    流程:
    1. 提取第 16 步预测值
    2. 按目标日期分组, 计算每日准确率/合格率/相关系数
    3. 按月份分组, 对日均值取算术平均

    Parameters
    ----------
    preds : np.ndarray, shape [N, pred_len] 或 [N, pred_len, C]
    trues : np.ndarray, 同 shape
    dates : np.ndarray, shape [N, pred_len]
        每个预测步对应的日期编码 (YYYYMMDD)
    capacity : float
    step : int
        考核步长 (0-based)

    Returns
    -------
    dict
        monthly: {YYYYMM: {accuracy, qualification_rate, r, n_days}}
        monthly_avg_accuracy: 月均准确率
        monthly_avg_qualification_rate: 月均合格率
        monthly_avg_r: 月均相关系数
    """
    p16, t16 = _step16_data(preds, trues, step)
    d16 = dates[:, step]

    # ---- 按天分组, 计算日均值 ----
    unique_dates = np.unique(d16)
    daily_records = []

    for d in unique_dates:
        mask = d16 == d
        p_day = p16[mask]
        t_day = t16[mask]

        acc = _daily_accuracy(p_day, t_day, capacity)
        qual = _daily_qualification_rate(
            p_day, t_day, capacity
        )
        r = _daily_correlation(p_day, t_day)

        # 从日期编码提取月份 (YYYYMM)
        month_key = int(d // 100)

        daily_records.append({
            "date": int(d),
            "month": month_key,
            "accuracy": acc,
            "qualification_rate": qual,
            "r": r,
            "n_points": int(mask.sum()),
        })

    # ---- 按月分组, 对日均值取平均 ----
    month_groups = {}
    for rec in daily_records:
        m = rec["month"]
        month_groups.setdefault(m, []).append(rec)

    monthly = {}
    acc_list = []
    qual_list = []
    r_list = []

    for m in sorted(month_groups.keys()):
        records = month_groups[m]
        n_days = len(records)

        m_acc = float(np.mean(
            [r["accuracy"] for r in records]
        ))
        m_qual = float(np.mean(
            [r["qualification_rate"] for r in records]
        ))
        m_r = float(np.mean(
            [r["r"] for r in records]
        ))

        monthly[m] = {
            "accuracy": m_acc,
            "qualification_rate": m_qual,
            "r": m_r,
            "n_days": n_days,
        }
        acc_list.append(m_acc)
        qual_list.append(m_qual)
        r_list.append(m_r)

    return {
        "monthly": monthly,
        "monthly_avg_accuracy": float(np.mean(acc_list)),
        "monthly_avg_qualification_rate": float(
            np.mean(qual_list)
        ),
        "monthly_avg_r": float(np.mean(r_list)),
    }
