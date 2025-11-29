# tcn_model.py
# Temporal Convolutional Network（TCN，多变量，单步预测），评估与基线对齐 + WandB + 可视化

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

# ====================== WandB 可选导入 ======================
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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ====================== 可视化函数 ======================
def plot_loss_curves(train_losses, val_losses, save_path, title="Training Curve"):
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[保存] Loss 曲线: {save_path}")


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


# ====================== TCN 模型定义 ======================
class Chomp1d(nn.Module):
    """裁掉右侧 padding，保证因果卷积"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNRegressor(nn.Module):
    """
    多变量 TCN 回归：
    输入: (B, L, F) -> 内部转成 (B, F, L)
    输出: (B, )
    """
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        in_ch = input_size
        for i in range(num_levels):
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x):
        # x: (B, L, F) -> (B, F, L)
        x = x.permute(0, 2, 1)
        y = self.network(x)        # (B, C, L)
        y_last = y[:, :, -1]       # (B, C)
        out = self.fc(y_last)      # (B, 1)
        return out.squeeze(-1)     # (B,)


# ====================== 训练 & 评估通用函数 ======================
def build_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))
    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        yp = model(Xb)
        loss = criterion(yp, yb)
        loss.backward()
        optimizer.step()

        bs = Xb.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    ys = []
    ps = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            yp = model(Xb)
            loss = criterion(yp, yb)

            bs = Xb.size(0)
            total += loss.item() * bs
            n += bs

            ys.append(yb.cpu().numpy())
            ps.append(yp.cpu().numpy())

    if n == 0:
        return 0.0, np.array([]), np.array([])

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return total / n, y_true, y_pred


# ====================== 主流程：TCN 单步预测（多变量输入） ======================
def run_tcn(
    hidden_channels=(64, 64, 64),
    kernel_size=3,
    dropout=0.2,
    batch_size=32,
    lr=1e-3,
    num_epochs=80,
    patience=10,
) -> Dict[str, float]:
    model_name = "TCN"

    # 和其他模型统一的数据配置
    cfg = DataConfig(
        data_path="D:/12138/大数据系统原理与应用/期末作业/数据/GreatPlains_8day_merged.csv",
        train_end="2014-12-31",
        val_end="2018-12-31",
        window_size=12,
        horizon=1,
        low_quantile=0.2,
    )

    df = load_and_prepare_data(cfg)
    print("TCN 使用的多变量特征列:", cfg.feature_cols)
    df_train, df_val, df_test = split_by_time(df, cfg)

    # 滑窗：X (N, L, F 多变量), y (N,)
    X_train, y_train, idx_train, dates_train, scaler = make_sliding_windows(
        df_train, cfg, scaler=None, fit_scaler=True
    )
    X_val, y_val, idx_val, dates_val, _ = make_sliding_windows(
        df_val, cfg, scaler=scaler, fit_scaler=False
    )
    X_test, y_test, idx_test, dates_test, _ = make_sliding_windows(
        df_test, cfg, scaler=scaler, fit_scaler=False
    )

    print("TCN sliding windows:")
    print("  Train:", X_train.shape, " Val:", X_val.shape, " Test:", X_test.shape)

    # 二分类阈值（基于训练集）
    threshold = float(np.quantile(y_train, cfg.low_quantile))
    print(f"Low-NDVI threshold (q={cfg.low_quantile}): {threshold:.4f}")

    train_loader = build_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = build_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    device = DEVICE
    input_size = X_train.shape[-1]  # 多变量数 F
    print(f"TCN 输入维度 input_size = {input_size}")

    model = TCNRegressor(
        input_size=input_size,
        num_channels=list(hidden_channels),
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------- WandB 初始化（可选 & 容错） ----------
    WANDB_ENABLED = False
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="ndvi_drought_greatplains",
                name=f"{model_name}_channels{hidden_channels}_k{kernel_size}",
                # 如果你经常登录失败，可以加 mode="offline"
                # mode="offline",
                config={
                    "model": model_name,
                    "hidden_channels": hidden_channels,
                    "kernel_size": kernel_size,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "lr": lr,
                    "num_epochs": num_epochs,
                    "window_size": cfg.window_size,
                    "horizon": cfg.horizon,
                    "train_end": cfg.train_end,
                    "val_end": cfg.val_end,
                    "features": cfg.feature_cols,
                },
            )
            WANDB_ENABLED = True
            print("[WandB] 初始化成功，将记录本次 TCN 实验。")
        except Exception as e:
            print("[WandB] 初始化失败，将在不记录的情况下继续运行。错误信息：", e)
    else:
        print("[WandB] 当前环境不可用，跳过 WandB 记录。")

    # ---------- 训练循环（带早停 + loss 记录） ----------
    best_val = float("inf")
    best_state = None
    wait = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train={tr_loss:.6f} | val={val_loss:.6f}")

        if WANDB_ENABLED:
            wandb.log({
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
            })

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            print("  -> New best model (saved).")
        else:
            wait += 1
            print(f"  -> No improvement. Patience {wait}/{patience}")
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- 测试集评估 ----------
    test_loss, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"Test MSE: {test_loss:.6f}")

    metrics = compute_all_metrics(y_true=y_true, y_pred=y_pred, threshold=threshold)
    print("TCN metrics:", metrics)

    if WANDB_ENABLED:
        wandb.log({
            "test_loss": test_loss,
            "test_MAE": metrics["MAE"],
            "test_RMSE": metrics["RMSE"],
            "test_F1": metrics["F1-score"],
            "test_ROC_AUC": metrics["ROC-AUC"],
        })

    # ---------- 保存指标 & 预测结果 ----------
    out_dir = "D:/12138/大数据系统原理与应用/期末作业/模型评估/tcn"
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics_tcn.csv")
    extra = {
        "hidden_channels": str(hidden_channels),
        "kernel_size": str(kernel_size),
        "dropout": str(dropout),
        "batch_size": str(batch_size),
        "lr": str(lr),
        "best_val_mse": f"{best_val:.6f}",
        "threshold": threshold,
    }
    export_metrics_to_csv(model_name, metrics, metrics_path, extra_info=extra)

    # 预测结果（方便画图与对比）
    pred_path = os.path.join(out_dir, "pred_tcn.csv")
    pred_df = pd.DataFrame({
        "date": dates_test,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    pred_df.to_csv(pred_path, index=False)
    print(f"[保存] 预测结果: {pred_path}")

    # ---------- 可视化 ----------
    figs_dir = os.path.join(out_dir, "figs_tcn")
    os.makedirs(figs_dir, exist_ok=True)

    loss_fig_path = os.path.join(figs_dir, "tcn_loss.png")
    plot_loss_curves(train_losses, val_losses, loss_fig_path, title="TCN Loss")

    pred_fig_path = os.path.join(figs_dir, "tcn_pred_vs_true.png")
    plot_pred_vs_true(dates_test, y_true, y_pred, pred_fig_path,
                      title="TCN Test NDVI")

    resid_fig_path = os.path.join(figs_dir, "tcn_residuals.png")
    plot_residual_hist(y_true, y_pred, resid_fig_path,
                       title="TCN Residuals")

    # 把图上传到 WandB（如果可用）
    if WANDB_ENABLED:
        wandb.log({
            "loss_curve": wandb.Image(loss_fig_path),
            "pred_vs_true": wandb.Image(pred_fig_path),
            "residual_hist": wandb.Image(resid_fig_path),
        })
        wandb.finish()

    print("TCN 训练 / 评估 / 可视化 完成。")
    return metrics


def main():
    print("=== Temporal Convolutional Network (TCN) ===")
    run_tcn()


if __name__ == "__main__":
    main()
