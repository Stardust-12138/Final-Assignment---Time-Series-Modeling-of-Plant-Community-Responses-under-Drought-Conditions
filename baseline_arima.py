## baseline_arima.py
# ARIMA / SARIMA 基线：只用 NDVI 序列（train+val 拟合，test 预测）
import os
import tempfile

# ====================== 1. 核心：自动设置临时路径 ======================
def set_custom_temp_dir():
    """自动设置固定的临时路径（无交互输入）"""
    custom_temp_path = "D:/temp1"
    try:
        if not os.path.exists(custom_temp_path):
            os.makedirs(custom_temp_path, exist_ok=True)
            print(f"已创建自定义临时文件夹: {custom_temp_path}")
        
        # 验证路径可写性
        test_file = os.path.join(custom_temp_path, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        
        # 修改环境变量（仅当前程序有效）
        os.environ["TMP"] = custom_temp_path
        os.environ["TEMP"] = custom_temp_path
        
        # 验证临时路径生效
        temp_file = tempfile.NamedTemporaryFile(dir=custom_temp_path, delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        os.remove(temp_file_path)
        
        print(f"临时路径已自动设置为: {custom_temp_path}")
        return True
    except PermissionError:
        print(f"错误：无权限访问 {custom_temp_path}，请检查路径权限")
        return False
    except Exception as e:
        print(f"设置临时路径失败: {str(e)}")
        return False

# 先设置临时路径（必须在导入scipy/sklearn等库之前执行）
set_custom_temp_dir()

# ====================== 2. WandB 可选导入（不强制） ======================
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
    print("[WandB] 未安装 wandb，自动关闭实验记录功能。")

# ====================== 3. 其他依赖 ======================
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_utils import (
    DataConfig,
    load_and_prepare_data,
    split_by_time,
    make_sliding_windows,
    compute_all_metrics,
    export_metrics_to_csv,
)

plt.switch_backend("Agg")  # 不弹窗，直接存图

# ====================== 4. 简单可视化函数 ======================
def plot_pred_vs_true(dates, y_true, y_pred, save_path, title="Pred vs True NDVI"):
    plt.figure()
    plt.plot(dates, y_true, label="True NDVI")
    plt.plot(dates, y_pred, label="Pred NDVI")
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[保存] 预测对比图: {save_path}")


def plot_residual_hist(y_true, y_pred, save_path, title="Residual Histogram"):
    residuals = y_pred - y_true
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel("Prediction - Truth")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[保存] 残差直方图: {save_path}")


# ====================== 5. ARIMA / SARIMA 网格搜索 ======================
def grid_search_arima(
    series_train: np.ndarray,
    p_values = (0, 1, 2, 3),
    d_values = (0, 1),
    q_values = (0, 1, 2, 3),
) -> Tuple[Optional[object], Optional[Tuple[int,int,int]], float]:
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    model = ARIMA(series_train, order=order)
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = order
                        best_model = fitted
                    print(f"ARIMA{order} AIC={fitted.aic:.2f}")
                except Exception as e:
                    print(f"ARIMA{order} failed: {e}")
                    continue

    print(f"Best ARIMA order: {best_order}, AIC={best_aic:.2f}")
    return best_model, best_order, best_aic


def grid_search_sarima(
    series_train: np.ndarray,
    s: int = 46,
    p_values = (0, 1, 2),
    d_values = (0, 1),
    q_values = (0, 1, 2),
    P_values = (0, 1),
    D_values = (0, 1),
    Q_values = (0, 1),
) -> Tuple[Optional[object], Optional[Tuple[int,int,int,int,int,int,int]], float]:
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, s)
                            try:
                                model = SARIMAX(
                                    series_train,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                                fitted = model.fit(disp=False)
                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_order = (*order, *seasonal_order)
                                    best_model = fitted
                                print(f"SARIMA{order}x{seasonal_order} AIC={fitted.aic:.2f}")
                            except Exception as e:
                                print(f"SARIMA{order}x{seasonal_order} failed: {e}")
                                continue

    print(f"Best SARIMA order: {best_order}, AIC={best_aic:.2f}")
    return best_model, best_order, best_aic


def align_predictions_to_targets(
    y_full_pred: np.ndarray,
    target_idx_in_df_test: np.ndarray
) -> np.ndarray:
    """
    y_full_pred: 针对 df_test 整个序列的预测值 (长度 = len(df_test))
    target_idx_in_df_test: 滑动窗口目标在 df_test 中的行索引
    返回：与 y_test_reg 对齐的预测序列 (同长度)
    """
    return y_full_pred[target_idx_in_df_test]


# ====================== 6. 结果保存 + wandb 记录 ======================
def save_and_log_results(
    model_name: str,
    df_dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    threshold: float,
    order_str: str,
    aic_val: float,
    metrics_filename: str,
    pred_filename: str,
    figs_subdir: str,
):
    """
    统一处理：
    - 输出 metrics_xxx.csv
    - 输出 pred_xxx.csv
    - 画两张图
    - 可选 wandb 记录
    """
    base_dir = "D:/12138/大数据系统原理与应用/期末作业/模型评估/arima"
    os.makedirs(base_dir, exist_ok=True)

    # 1) 指标 CSV
    metrics_path = os.path.join(base_dir, metrics_filename)
    extra_info = {
        "order": order_str,
        "AIC": f"{aic_val:.2f}",
        "threshold": threshold,
    }
    export_metrics_to_csv(
        model_name=model_name,
        metrics=metrics,
        output_path=metrics_path,
        extra_info=extra_info,
    )
    print(f"[保存] 指标: {metrics_path}")

    # 2) 预测结果 CSV（对齐后的）
    pred_path = os.path.join(base_dir, pred_filename)
    pred_df = pd.DataFrame({
        "date": df_dates,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    pred_df.to_csv(pred_path, index=False)
    print(f"[保存] 预测结果: {pred_path}")

    # 3) 可视化
    figs_dir = os.path.join(base_dir, figs_subdir)
    os.makedirs(figs_dir, exist_ok=True)

    pred_fig_path = os.path.join(figs_dir, f"{model_name}_pred_vs_true.png")
    plot_pred_vs_true(
        df_dates, y_true, y_pred,
        pred_fig_path,
        title=f"{model_name}: Test NDVI"
    )

    resid_fig_path = os.path.join(figs_dir, f"{model_name}_residuals.png")
    plot_residual_hist(
        y_true, y_pred,
        resid_fig_path,
        title=f"{model_name}: Residuals"
    )

    # 4) WandB 记录（可选，不影响主程序）
    WANDB_ENABLED = False
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="ndvi_drought_greatplains",
                name=f"{model_name}_Baseline",
                # mode="offline",  # 如果经常 401，可以打开离线模式
                config={
                    "model": model_name,
                    "order": order_str,
                    "AIC": aic_val,
                    "threshold": threshold,
                },
            )
            WANDB_ENABLED = True
            print(f"[WandB] 初始化成功，将记录本次 {model_name} 实验。")
        except Exception as e:
            print("[WandB] 初始化失败，将在不记录的情况下继续运行。错误信息：", e)

    if WANDB_ENABLED:
        wandb.log({
            "test_MAE": metrics["MAE"],
            "test_RMSE": metrics["RMSE"],
            "test_F1": metrics["F1-score"],
            "test_ROC_AUC": metrics["ROC-AUC"],
        })
        wandb.log({
            "pred_vs_true": wandb.Image(pred_fig_path),
            "residual_hist": wandb.Image(resid_fig_path),
        })
        wandb.finish()


# ====================== 7. 主流程 ======================
def main():
    cfg = DataConfig(
        data_path="D:/12138/大数据系统原理与应用/期末作业/数据/GreatPlains_8day_merged.csv",
        train_end="2014-12-31",
        val_end="2018-12-31",
        window_size=12,
        horizon=1,
        low_quantile=0.2,   # 定义“低 NDVI”阈值（20%分位）
    )

    # 1. 数据 & 划分
    df = load_and_prepare_data(cfg)
    df_train, df_val, df_test = split_by_time(df, cfg)

    # 2. 滑动窗口（只为了得到统一的 y_test 和 idx_test）
    X_train, y_train_reg, idx_train, dates_train, scaler = make_sliding_windows(
        df_train, cfg, scaler=None, fit_scaler=True
    )
    X_val, y_val_reg, idx_val, dates_val, _ = make_sliding_windows(
        df_val, cfg, scaler=scaler, fit_scaler=False
    )
    X_test, y_test_reg, idx_test, dates_test, _ = make_sliding_windows(
        df_test, cfg, scaler=scaler, fit_scaler=False
    )

    print("Train windows:", X_train.shape, "Test windows:", X_test.shape)

    # 3. 定义低 NDVI 阈值（基于训练集回归标签）
    threshold = float(np.quantile(y_train_reg, cfg.low_quantile))
    print(f"Low-NDVI threshold (quantile={cfg.low_quantile}): {threshold:.4f}")

    # 使用 train+val 序列做 ARIMA/SARIMA 网格搜索
    series_train_val = np.concatenate(
        [df_train[cfg.target_col].values, df_val[cfg.target_col].values]
    )
    series_test = df_test[cfg.target_col].values   # 整个 test 序列

    # ================= 4. 纯 ARIMA =================
    print("\n=== Grid search ARIMA ===")
    best_arima_model, best_arima_order, best_arima_aic = grid_search_arima(series_train_val)

    if best_arima_model is not None and best_arima_order is not None:
        # 在 test 段长度上做 forecast（对未来 len(series_test) 个点）
        y_test_full_pred_arima = best_arima_model.forecast(steps=len(series_test))
        y_test_full_pred_arima = np.array(y_test_full_pred_arima)

        # 对齐到滑窗目标点（与 y_test_reg 一一对应）
        y_pred_test_arima = align_predictions_to_targets(
            y_full_pred=y_test_full_pred_arima,
            target_idx_in_df_test=idx_test,
        )

        # 计算四个指标
        metrics_arima = compute_all_metrics(
            y_true=y_test_reg,
            y_pred=y_pred_test_arima,
            threshold=threshold,
        )
        print("Best ARIMA order:", best_arima_order)
        print("ARIMA metrics on aligned test:", metrics_arima)

        # 保存 + 可视化 + wandb
        save_and_log_results(
            model_name=f"ARIMA{best_arima_order}",
            df_dates=dates_test,
            y_true=y_test_reg,
            y_pred=y_pred_test_arima,
            metrics=metrics_arima,
            threshold=threshold,
            order_str=str(best_arima_order),
            aic_val=best_arima_aic,
            metrics_filename="metrics_arima_best.csv",
            pred_filename="pred_arima_best.csv",
            figs_subdir="figs_arima",
        )
    else:
        print("ARIMA 网格搜索未找到有效模型。")

    # ================= 5. 季节性 SARIMA =================
    print("\n=== Grid search SARIMA ===")
    best_sarima_model, best_sarima_order, best_sarima_aic = grid_search_sarima(
        series_train_val,
        s=46,   # 8 天 * 46 ≈ 368 天，近似 1 年季节性
    )

    if best_sarima_model is not None and best_sarima_order is not None:
        y_test_full_pred_sarima = best_sarima_model.forecast(steps=len(series_test))
        y_test_full_pred_sarima = np.array(y_test_full_pred_sarima)

        y_pred_test_sarima = align_predictions_to_targets(
            y_full_pred=y_test_full_pred_sarima,
            target_idx_in_df_test=idx_test,
        )

        metrics_sarima = compute_all_metrics(
            y_true=y_test_reg,
            y_pred=y_pred_test_sarima,
            threshold=threshold,
        )
        print("Best SARIMA order:", best_sarima_order)
        print("SARIMA metrics on aligned test:", metrics_sarima)

        save_and_log_results(
            model_name=f"SARIMA{best_sarima_order}",
            df_dates=dates_test,
            y_true=y_test_reg,
            y_pred=y_pred_test_sarima,
            metrics=metrics_sarima,
            threshold=threshold,
            order_str=str(best_sarima_order),
            aic_val=best_sarima_aic,
            metrics_filename="metrics_sarima_best.csv",
            pred_filename="pred_sarima_best.csv",
            figs_subdir="figs_sarima",
        )
    else:
        print("SARIMA 网格搜索未找到有效模型。")


if __name__ == "__main__":
    main()
