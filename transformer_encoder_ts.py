# transformer_encoder_ts.py
# Transformer Encoder（Time-Series Transformer，多变量，单步预测）

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

# ====================== 2. WandB 可选导入（失败不影响训练） ======================
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
    print("[WandB] 未安装 wandb，自动关闭实验记录功能。")

# ====================== 3. 其他依赖 ======================
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

plt.switch_backend("Agg")  # 不弹窗口，直接存图
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ====================== 4. 位置编码 ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, B, d_model)
        L = x.size(0)
        return x + self.pe[:L]


# ====================== 5. Time-Series Transformer Encoder ======================
class TimeSeriesTransformerEncoder(nn.Module):
    """
    多变量序列 -> 线性投影到 d_model -> Transformer Encoder -> 取最后时刻 -> 回归
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        B, L, F = x.shape
        x = self.input_proj(x)        # (B, L, d_model)
        x = x.permute(1, 0, 2)        # (L, B, d_model)
        x = self.pos_encoder(x)       # (L, B, d_model)
        enc_out = self.encoder(x)     # (L, B, d_model)
        last = enc_out[-1]            # (B, d_model)
        out = self.fc(last)           # (B, 1)
        return out.squeeze(-1)


# ====================== 6. 通用训练 & 评估 ======================
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


# ====================== 7. 可视化函数 ======================
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


# ====================== 8. 主流程：Transformer Encoder ======================
def run_transformer(
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    batch_size=32,
    lr=1e-3,
    num_epochs=80,
    patience=10,
) -> Dict[str, float]:
    model_name = "TransformerEncoder_TS"

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
    print("Transformer 使用的多变量特征列:", cfg.feature_cols)
    df_train, df_val, df_test = split_by_time(df, cfg)

    # 2. 滑动窗口（统一与其他模型）
    X_train, y_train, idx_train, dates_train, scaler = make_sliding_windows(
        df_train, cfg, scaler=None, fit_scaler=True
    )
    X_val, y_val, idx_val, dates_val, _ = make_sliding_windows(
        df_val, cfg, scaler=scaler, fit_scaler=False
    )
    X_test, y_test, idx_test, dates_test, _ = make_sliding_windows(
        df_test, cfg, scaler=scaler, fit_scaler=False
    )

    print("Transformer sliding windows:")
    print("  Train:", X_train.shape, " Val:", X_val.shape, " Test:", X_test.shape)

    # 3. 二分类低 NDVI 阈值
    threshold = float(np.quantile(y_train, cfg.low_quantile))
    print(f"Low-NDVI threshold (q={cfg.low_quantile}): {threshold:.4f}")

    # 4. DataLoader
    train_loader = build_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = build_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    # 5. 模型、损失、优化器
    device = DEVICE
    input_size = X_train.shape[-1]
    print(f"Transformer 输入维度 input_size = {input_size}")

    model = TimeSeriesTransformerEncoder(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6. WandB 初始化（可选 & 容错）
    WANDB_ENABLED = False
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="ndvi_drought_greatplains",
                name=f"{model_name}_dm{d_model}_h{nhead}_L{num_layers}",
                # 如果你的网络经常 401，可以加 mode="offline"
                # mode="offline",
                config={
                    "model": model_name,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "dim_feedforward": dim_feedforward,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "lr": lr,
                    "num_epochs": num_epochs,
                    "patience": patience,
                    "window_size": cfg.window_size,
                    "horizon": cfg.horizon,
                    "train_end": cfg.train_end,
                    "val_end": cfg.val_end,
                    "features": cfg.feature_cols,
                },
            )
            WANDB_ENABLED = True
            print("[WandB] 初始化成功，将记录本次 Transformer 实验。")
        except Exception as e:
            print("[WandB] 初始化失败，将在不记录的情况下继续运行。错误信息：", e)
    else:
        print("[WandB] 当前环境不可用，跳过 WandB 记录。")

    # 7. 训练循环（带早停 + loss 记录）
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
            print("  -> New best Transformer model (saved).")
        else:
            wait += 1
            print(f"  -> No improvement. Patience {wait}/{patience}")
            if wait >= patience:
                print("Early stopping.")
                break

    # 8. 加载最佳模型，测试集评估
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"Test MSE: {test_loss:.6f}")

    metrics = compute_all_metrics(y_true=y_true, y_pred=y_pred, threshold=threshold)
    print("Transformer metrics:", metrics)

    if WANDB_ENABLED:
        wandb.log({
            "test_loss": test_loss,
            "test_MAE": metrics["MAE"],
            "test_RMSE": metrics["RMSE"],
            "test_F1": metrics["F1-score"],
            "test_ROC_AUC": metrics["ROC-AUC"],
        })

    # 9. 保存指标 & 预测结果
    out_dir = "D:/12138/大数据系统原理与应用/期末作业/模型评估/transformer_encoder"
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics_transformer_encoder.csv")
    extra = {
        "d_model": str(d_model),
        "nhead": str(nhead),
        "num_layers": str(num_layers),
        "dim_feedforward": str(dim_feedforward),
        "dropout": str(dropout),
        "batch_size": str(batch_size),
        "lr": str(lr),
        "best_val_mse": f"{best_val:.6f}",
        "threshold": threshold,
    }
    export_metrics_to_csv(model_name, metrics, metrics_path, extra_info=extra)
    print(f"[保存] 指标: {metrics_path}")

    # 预测结果保存（方便画图和对比）
    pred_path = os.path.join(out_dir, "pred_transformer_encoder.csv")
    pred_df = pd.DataFrame({
        "date": dates_test,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    pred_df.to_csv(pred_path, index=False)
    print(f"[保存] 预测结果: {pred_path}")

    # 10. 可视化：Loss 曲线 + 预测 vs 真实 + 残差
    figs_dir = os.path.join(out_dir, "figs_transformer_encoder")
    os.makedirs(figs_dir, exist_ok=True)

    loss_fig_path = os.path.join(figs_dir, "transformer_loss.png")
    plot_loss_curves(train_losses, val_losses, loss_fig_path, title="Transformer Encoder Loss")

    pred_fig_path = os.path.join(figs_dir, "transformer_pred_vs_true.png")
    plot_pred_vs_true(dates_test, y_true, y_pred, pred_fig_path,
                      title="Transformer Encoder Test NDVI")

    resid_fig_path = os.path.join(figs_dir, "transformer_residuals.png")
    plot_residual_hist(y_true, y_pred, resid_fig_path,
                       title="Transformer Encoder Residuals")

    # 11. 把图传到 WandB（如果启用）
    if WANDB_ENABLED:
        wandb.log({
            "loss_curve": wandb.Image(loss_fig_path),
            "pred_vs_true": wandb.Image(pred_fig_path),
            "residual_hist": wandb.Image(resid_fig_path),
        })
        wandb.finish()

    print("Transformer Encoder 训练 / 评估 / 可视化 完成。")
    return metrics


def main():
    print("=== Transformer Encoder for Time Series ===")
    run_transformer()


if __name__ == "__main__":
    main()
