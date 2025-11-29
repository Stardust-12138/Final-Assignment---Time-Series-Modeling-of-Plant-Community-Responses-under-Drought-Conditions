# baseline_persistence.py
# 持久性基线：预测值 = 上一时刻真实 NDVI

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
import matplotlib.pyplot as plt
import pandas as pd

from data_utils import (
    DataConfig,
    load_and_prepare_data,
    split_by_time,
    make_sliding_windows,
    compute_all_metrics,
    export_metrics_to_csv,
)

plt.switch_backend("Agg")  # 不弹窗，直接存图


# ====================== 简单可视化函数（没有训练，所以没有 loss 曲线） ======================
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


# ====================== 持久性预测核心函数 ======================
def persistence_full(series_test: np.ndarray) -> np.ndarray:
    """
    对 test 整个序列做持久性预测：
    y_pred[t] = y_true[t-1]，第一个样本用自身填补。
    """
    y_pred = np.empty_like(series_test)
    y_pred[0] = series_test[0]
    y_pred[1:] = series_test[:-1]
    return y_pred


def main():
    # 1. 配置（注意 low_quantile 控制“低 NDVI”阈值）
    cfg = DataConfig(
        data_path="D:/12138/大数据系统原理与应用/期末作业/数据/GreatPlains_8day_merged.csv",
        train_end="2014-12-31",
        val_end="2018-12-31",
        window_size=12,
        horizon=1,
        low_quantile=0.2,  # 20% 分位当“低 NDVI”
    )

    # 2. 数据与时间划分
    df = load_and_prepare_data(cfg)
    df_train, df_val, df_test = split_by_time(df, cfg)

    # 3. 滑动窗口（train & test），得到统一的 y_test_reg 和 idx_test
    X_train, y_train_reg, idx_train, dates_train, scaler = make_sliding_windows(
        df_train, cfg, scaler=None, fit_scaler=True
    )
    X_test, y_test_reg, idx_test, dates_test, _ = make_sliding_windows(
        df_test, cfg, scaler=scaler, fit_scaler=False
    )

    print("Train windows:", X_train.shape, "Test windows:", X_test.shape)

    # 4. 定义“低 NDVI”阈值（基于训练集回归标签的低分位数）
    threshold = float(np.quantile(y_train_reg, cfg.low_quantile))
    print(f"Low-NDVI threshold (quantile={cfg.low_quantile}): {threshold:.4f}")

    # 5. 在 df_test 上生成完整的 persistence 预测序列
    series_test = df_test[cfg.target_col].values       # 长度 = len(df_test)
    y_full_pred = persistence_full(series_test)        # 同长度的预测值

    # 6. 用 idx_test 对齐到滑窗目标点（与 y_test_reg 对齐）
    #    idx_test 是滑窗目标在 df_test 中的行索引
    y_pred_test = y_full_pred[idx_test]                # 长度与 y_test_reg 一致

    # 7. 计算 MAE / RMSE / F1 / ROC-AUC
    metrics = compute_all_metrics(
        y_true=y_test_reg,
        y_pred=y_pred_test,
        threshold=threshold,
    )
    print("Persistence metrics on aligned test:", metrics)

    # 8. 导出指标 CSV（和之前保持一致）
    out_dir = "D:/12138/大数据系统原理与应用/期末作业/模型评估/persistence"
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics_persistence.csv")
    export_metrics_to_csv(
        model_name="Persistence",
        metrics=metrics,
        output_path=metrics_path,
    )
    print(f"[保存] 指标: {metrics_path}")

    # 9. 保存预测结果（方便后面对比和画图）
    pred_path = os.path.join(out_dir, "pred_persistence.csv")
    pred_df = pd.DataFrame({
        "date": dates_test,
        "y_true": y_test_reg,
        "y_pred": y_pred_test,
    })
    pred_df.to_csv(pred_path, index=False)
    print(f"[保存] 预测结果: {pred_path}")

    # 10. 可视化（没有训练过程，就只画预测相关的图）
    figs_dir = os.path.join(out_dir, "figs_persistence")
    os.makedirs(figs_dir, exist_ok=True)

    pred_fig_path = os.path.join(figs_dir, "persistence_pred_vs_true.png")
    plot_pred_vs_true(
        dates_test, y_test_reg, y_pred_test,
        pred_fig_path,
        title="Persistence Baseline: Test NDVI"
    )

    resid_fig_path = os.path.join(figs_dir, "persistence_residuals.png")
    plot_residual_hist(
        y_test_reg, y_pred_test,
        resid_fig_path,
        title="Persistence Baseline: Residuals"
    )

    # 11. WandB 记录（可选，如登录失败不会影响程序）
    WANDB_ENABLED = False
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="ndvi_drought_greatplains",
                name="Persistence_Baseline",
                # 如果你经常 401，可以加上 mode="offline"
                # mode="offline",
                config={
                    "model": "Persistence",
                    "window_size": cfg.window_size,
                    "horizon": cfg.horizon,
                    "train_end": cfg.train_end,
                    "val_end": cfg.val_end,
                    "low_quantile": cfg.low_quantile,
                },
            )
            WANDB_ENABLED = True
            print("[WandB] 初始化成功，将记录本次 Persistence 实验。")
        except Exception as e:
            print("[WandB] 初始化失败，将在不记录的情况下继续运行。错误信息：", e)

    if WANDB_ENABLED:
        # 记录指标
        wandb.log({
            "test_MAE": metrics["MAE"],
            "test_RMSE": metrics["RMSE"],
            "test_F1": metrics["F1-score"],
            "test_ROC_AUC": metrics["ROC-AUC"],
        })
        # 记录图像
        wandb.log({
            "pred_vs_true": wandb.Image(pred_fig_path),
            "residual_hist": wandb.Image(resid_fig_path),
        })
        wandb.finish()

    print("持久性基线 评估 / 可视化 / WandB 记录 完成。")


if __name__ == "__main__":
    main()
