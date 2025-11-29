# baseline_linear_regression.py
# 线性回归 / 岭回归基线（基于滑动窗口，多变量输入）
import os
import tempfile

# ====================== 核心：自动设置临时路径 ======================
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

# ====================== WandB 可选导入（不强制） ======================
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
    print("[WandB] 未安装 wandb，自动关闭实验记录功能。")

# ====================== 其他依赖 ======================
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

from data_utils import (
    DataConfig,
    load_and_prepare_data,
    split_by_time,
    make_sliding_windows,
    compute_all_metrics,
    export_metrics_to_csv,
)

plt.switch_backend("Agg")  # 不弹窗，直接存图


# ====================== 简单可视化函数（只有预测相关） ======================
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


# ====================== 线性 / 岭回归基线 ======================
class LinearRegressionBaseline:
    """
    多变量滑动窗口线性 / 岭回归：
    X: (N, L, F) -> 展平为 (N, L*F)，做回归
    """
    def __init__(self, use_ridge: bool = False, alpha: float = 1.0):
        self.use_ridge = use_ridge
        self.alpha = alpha
        self.model = Ridge(alpha=alpha) if use_ridge else LinearRegression()

    @staticmethod
    def _flatten_windows(X: np.ndarray) -> np.ndarray:
        n, w, f = X.shape
        return X.reshape(n, w * f)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_flat = self._flatten_windows(X)
        self.model.fit(X_flat, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten_windows(X)
        return self.model.predict(X_flat)


def run_linear_baseline(use_ridge: bool = False, alpha: float = 1.0) -> Dict[str, float]:
    """
    线性 / 岭回归基线：
    - 数据与 LSTM / TCN / TFT 等保持同样划分
    - 多变量滑窗输入
    - 单步 NDVI 回归
    - 输出 metrics_*.csv + pred_*.csv + 简单可视化 + wandb（可选）
    """
    cfg = DataConfig(
        data_path="D:/12138/大数据系统原理与应用/期末作业/数据/GreatPlains_8day_merged.csv",
        train_end="2014-12-31",
        val_end="2018-12-31",
        window_size=12,
        horizon=1,
        low_quantile=0.2,
    )

    # 1. 数据 & 划分
    df = load_and_prepare_data(cfg)
    df_train, df_val, df_test = split_by_time(df, cfg)

    # 2. 滑动窗口（train / val / test）
    X_train, y_train_reg, idx_train, dates_train, scaler = make_sliding_windows(
        df_train, cfg, scaler=None, fit_scaler=True
    )
    X_val, y_val_reg, idx_val, dates_val, _ = make_sliding_windows(
        df_val, cfg, scaler=scaler, fit_scaler=False
    )
    X_test, y_test_reg, idx_test, dates_test, _ = make_sliding_windows(
        df_test, cfg, scaler=scaler, fit_scaler=False
    )

    print("Train windows:", X_train.shape, 
          "Val windows:", X_val.shape, 
          "Test windows:", X_test.shape)

    # 3. 低 NDVI 阈值（基于训练集标签）
    threshold = float(np.quantile(y_train_reg, cfg.low_quantile))
    print(f"Low-NDVI threshold (quantile={cfg.low_quantile}): {threshold:.4f}")

    # 4. 拟合模型
    model = LinearRegressionBaseline(use_ridge=use_ridge, alpha=alpha)
    model.fit(X_train, y_train_reg)

    # 5. 在 test 上预测（与 y_test_reg 天然对齐）
    y_pred_test = model.predict(X_test)

    # 6. 统一指标
    metrics = compute_all_metrics(
        y_true=y_test_reg,
        y_pred=y_pred_test,
        threshold=threshold,
    )

    model_name = "Ridge" if use_ridge else "Linear"
    print(model_name, "metrics on aligned test:", metrics)

    # 7. 导出指标 CSV
    out_dir = "D:/12138/大数据系统原理与应用/期末作业/模型评估/linear_ridge"
    os.makedirs(out_dir, exist_ok=True)

    if use_ridge:
        metrics_path = os.path.join(out_dir, "metrics_ridge.csv")
    else:
        metrics_path = os.path.join(out_dir, "metrics_linear.csv")

    export_metrics_to_csv(
        model_name=model_name,
        metrics=metrics,
        output_path=metrics_path,
    )
    print(f"[保存] 指标: {metrics_path}")

    # 8. 保存预测结果（方便画图和对比）
    if use_ridge:
        pred_path = os.path.join(out_dir, "pred_ridge.csv")
    else:
        pred_path = os.path.join(out_dir, "pred_linear.csv")

    pred_df = pd.DataFrame({
        "date": dates_test,
        "y_true": y_test_reg,
        "y_pred": y_pred_test,
    })
    pred_df.to_csv(pred_path, index=False)
    print(f"[保存] 预测结果: {pred_path}")

    # 9. 可视化（没有训练过程，就只画预测相关的两张图）
    figs_dir = os.path.join(out_dir, "figs_ridge" if use_ridge else "figs_linear")
    os.makedirs(figs_dir, exist_ok=True)

    pred_fig_path = os.path.join(
        figs_dir,
        "ridge_pred_vs_true.png" if use_ridge else "linear_pred_vs_true.png"
    )
    plot_pred_vs_true(
        dates_test, y_test_reg, y_pred_test,
        pred_fig_path,
        title=f"{model_name} Regression: Test NDVI"
    )

    resid_fig_path = os.path.join(
        figs_dir,
        "ridge_residuals.png" if use_ridge else "linear_residuals.png"
    )
    plot_residual_hist(
        y_test_reg, y_pred_test,
        resid_fig_path,
        title=f"{model_name} Regression: Residuals"
    )

    # 10. WandB 记录（可选，失败不影响程序）
    WANDB_ENABLED = False
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="ndvi_drought_greatplains",
                name=f"{model_name}_Regression_Baseline",
                # 如果经常遇到 401，可以打开离线模式：
                # mode="offline",
                config={
                    "model": model_name,
                    "use_ridge": use_ridge,
                    "alpha": alpha if use_ridge else None,
                    "window_size": cfg.window_size,
                    "horizon": cfg.horizon,
                    "train_end": cfg.train_end,
                    "val_end": cfg.val_end,
                    "low_quantile": cfg.low_quantile,
                    "features": "multi-variate with seasonal encodings",
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

    print(f"{model_name} 回归基线 评估 / 可视化 / WandB 记录 完成。")

    return metrics


def main():
    print("=== Linear Regression Baseline (aligned) ===")
    run_linear_baseline(use_ridge=False)

    print("\n=== Ridge Regression Baseline (aligned) ===")
    run_linear_baseline(use_ridge=True, alpha=1.0)


if __name__ == "__main__":
    main()
