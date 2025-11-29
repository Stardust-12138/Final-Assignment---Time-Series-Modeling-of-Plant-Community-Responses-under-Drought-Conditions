# data_utils.py
# 统一的数据/滑窗/对齐评估工具（对所有模型共用）

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    roc_auc_score,
)


@dataclass
class DataConfig:
    data_path: str = "D:/12138/大数据系统原理与应用/期末作业/数据/GreatPlains_8day_merged.csv"
    date_col: str = "date"
    target_col: str = "ndvi"
    feature_cols: Optional[List[str]] = None

    # 时间划分
    train_end: str = "2014-12-31"
    val_end: str = "2018-12-31"

    # 滑动窗口
    window_size: int = 12
    horizon: int = 1

    # 二分类阈值：比如用 0.2 分位数来定义“低 NDVI”
    low_quantile: float = 0.2


def load_and_prepare_data(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    if cfg.feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cfg.feature_cols = numeric_cols.copy()

    # 时间特征
    df["doy"] = df[cfg.date_col].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.0)
    for col in ["doy_sin", "doy_cos"]:
        if col not in cfg.feature_cols:
            cfg.feature_cols.append(col)

    return df


def split_by_time(
    df: pd.DataFrame,
    cfg: DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = df[cfg.date_col] <= cfg.train_end
    val_mask = (df[cfg.date_col] > cfg.train_end) & (df[cfg.date_col] <= cfg.val_end)
    test_mask = df[cfg.date_col] > cfg.val_end

    df_train = df[train_mask].reset_index(drop=True)
    df_val = df[val_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    print("Train size:", len(df_train),
          "Val size:", len(df_val),
          "Test size:", len(df_test))
    return df_train, df_val, df_test


def make_sliding_windows(
    df: pd.DataFrame,
    cfg: DataConfig,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    构建滑动窗口：
    X: (N, L, F)
    y: (N,)
    target_idx: 目标在 df 中的行索引 (int)
    target_dates: 目标的日期（时间戳）
    """
    features = df[cfg.feature_cols].values
    target = df[cfg.target_col].values

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)

    X_list, y_list, idx_list, date_list = [], [], [], []

    total_len = len(df)
    max_start = total_len - cfg.window_size - cfg.horizon + 1

    for i in range(max_start):
        x_win = features_scaled[i : i + cfg.window_size]
        target_idx = i + cfg.window_size + cfg.horizon - 1
        y_val = target[target_idx]

        X_list.append(x_win)
        y_list.append(y_val)
        idx_list.append(target_idx)
        date_list.append(df.iloc[target_idx][cfg.date_col])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    target_idx_arr = np.array(idx_list)
    target_dates_arr = np.array(date_list)

    return X, y, target_idx_arr, target_dates_arr, scaler


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}


def compute_binary_metrics_from_continuous(
    y_true_cont: np.ndarray,
    y_pred_cont: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    定义： y < threshold -> 1 (低 NDVI / 疑似干旱)
    """
    y_true_bin = (y_true_cont < threshold).astype(int)
    y_pred_bin = (y_pred_cont < threshold).astype(int)

    f1 = f1_score(y_true_bin, y_pred_bin)
    try:
        # 把“越低越干旱”当成正类，所以用 -y_pred 表示“干旱分数”
        roc = roc_auc_score(y_true_bin, -y_pred_cont)
    except Exception as e:
        print("Warning: ROC-AUC calculation failed:", e)
        roc = float("nan")

    return {"F1-score": f1, "ROC-AUC": roc}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    reg = compute_regression_metrics(y_true, y_pred)
    cls = compute_binary_metrics_from_continuous(y_true, y_pred, threshold)
    out = {}
    out.update(reg)
    out.update(cls)
    return out


def export_metrics_to_csv(
    model_name: str,
    metrics: Dict[str, float],
    output_path: str,
    extra_info: Optional[Dict[str, str]] = None,
) -> None:
    """
    将 MAE / RMSE / F1 / ROC-AUC 等导出为 CSV（单行）。
    """
    record = {"model": model_name}
    record.update(metrics)
    if extra_info is not None:
        record.update(extra_info)

    df = pd.DataFrame([record])
    out_dir = os.path.dirname(output_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Metrics exported to: {output_path}")
