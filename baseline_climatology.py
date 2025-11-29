# baseline_climatology.py
# 气候平均基线：预测值 = 同一 DOY 的多年平均 NDVI
# 加入：对齐评估 + 可视化（预测 vs 真实 + 残差）+ wandb（可选）

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import (
    DataConfig,
    load_and_prepare_data,
    split_by_time,
    make_sliding_windows,
    compute_all_metrics,
    export_metrics_to_csv,
)

plt.switch_backend("Agg")  # 不弹窗，直接存图


# ====================== 可视化函数（只有预测相关的） ======================
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


# ====================== 气候平均基线模型 ======================
class ClimatologyBaseline:
    """
    气候平均模型：训练集按 DOY 求多年平均 NDVI，
    预测时根据 DOY 返回相应平均值。
    """
    def __init__(self):
        self.doy_mean: pd.Series | None = None

    def fit(self, df_train: pd.DataFrame, cfg: DataConfig):
        # 注意：doy 列在 data_utils.load_and_prepare_data 中已经构建好了
        self.doy_mean = df_train.groupby("doy")[cfg.target_col].mean()

    def predict_full(self, df_any: pd.DataFrame, cfg: DataConfig) -> np.ndarray:
        """
        对 df_any（如 df_test）的每一行按 DOY 预测一个 NDVI 气候平均值。
        返回长度 = len(df_any) 的预测序列。
        """
        assert self.doy_mean is not None, "请先调用 fit()"
        doy_array = df_any["doy"].values
        preds = np.array([
            self.doy_mean.get(d, self.doy_mean.mean()) for d in doy_array
        ])
        return preds


def main():
    # 1. 数据配置（保持和其他模型一致）
    cfg = DataConfig(
        data_path="D:/12138/大数据系统原理与应用/期末作业/数据/GreatPlains_8day_merged.csv",
        train_end="2014-12-31",
        val_end="2018-12-31",
        window_size=12,
        horizon=1,
        low_quantile=0.2,
    )

    # 2. 数据 & 划分
    df = load_and_prepare_data(cfg)
    df_train, df_val, df_test = split_by_time(df, cfg)

    # 3. 滑动窗口（只为了得到对齐的 y_test_reg 和 idx_test）
    X_train, y_train_reg, idx_train, dates_train, scaler = make_sliding_windows(
        df_train, cfg, scaler=None, fit_scaler=True
    )
    X_test, y_test_reg, idx_test, dates_test, _ = make_sliding_windows(
        df_test, cfg, scaler=scaler, fit_scaler=False
    )

    print("Train windows:", X_train.shape, "Test windows:", X_test.shape)

    # 4. 低 NDVI 阈值（统一用训练集 target 的 0.2 分位）
    threshold = float(np.quantile(y_train_reg, cfg.low_quantile))
    print(f"Low-NDVI threshold (quantile={cfg.low_quantile}): {threshold:.4f}")

    # 5. 拟合气候平均模型
    model = ClimatologyBaseline()
    model.fit(df_train, cfg)

    # 6. 在整个 df_test 上按 DOY 预测
    y_full_pred = model.predict_full(df_test, cfg)   # 长度 = len(df_test)

    # 7. 用 idx_test 对齐到 y_test_reg（与滑窗目标完全一致）
    y_pred_test = y_full_pred[idx_test]              # 与 y_test_reg 同长度

    # 8. 统一指标计算
    metrics = compute_all_metrics(
        y_true=y_test_reg,
        y_pred=y_pred_test,
        threshold=threshold,
    )
    print("Climatology metrics on aligned test:", metrics)

    # 9. 导出指标 CSV
    out_dir = "D:/12138/大数据系统原理与应用/期末作业/模型评估/climatology"
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics_climatology.csv")

    export_metrics_to_csv(
        model_name="Climatology",
        metrics=metrics,
        output_path=metrics_path,
    )
    print(f"[保存] 指标: {metrics_path}")

    # 10. 保存预测结果（方便对比和画图）
    pred_path = os.path.join(out_dir, "pred_climatology.csv")
    pred_df = pd.DataFrame({
        "date": dates_test,
        "y_true": y_test_reg,
        "y_pred": y_pred_test,
    })
    pred_df.to_csv(pred_path, index=False)
    print(f"[保存] 预测结果: {pred_path}")

    # 11. 可视化：预测 vs 真实 + 残差直方图（没有训练过程就不画 loss）
    figs_dir = os.path.join(out_dir, "figs_climatology")
    os.makedirs(figs_dir, exist_ok=True)

    pred_fig_path = os.path.join(figs_dir, "climatology_pred_vs_true.png")
    plot_pred_vs_true(
        dates_test, y_test_reg, y_pred_test,
        pred_fig_path,
        title="Climatology Baseline: Test NDVI"
    )

    resid_fig_path = os.path.join(figs_dir, "climatology_residuals.png")
    plot_residual_hist(
        y_test_reg, y_pred_test,
        resid_fig_path,
        title="Climatology Baseline: Residuals"
    )

    # 12. WandB 记录（可选，失败不会影响程序）
    WANDB_ENABLED = False
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="ndvi_drought_greatplains",
                name="Climatology_Baseline",
                # 如果你经常遇到 401，可以加上 mode="offline"
                # mode="offline",
                config={
                    "model": "Climatology",
                    "window_size": cfg.window_size,
                    "horizon": cfg.horizon,
                    "train_end": cfg.train_end,
                    "val_end": cfg.val_end,
                    "low_quantile": cfg.low_quantile,
                },
            )
            WANDB_ENABLED = True
            print("[WandB] 初始化成功，将记录本次 Climatology 实验。")
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

    print("气候平均基线 评估 / 可视化 / WandB 记录 完成。")


if __name__ == "__main__":
    main()
