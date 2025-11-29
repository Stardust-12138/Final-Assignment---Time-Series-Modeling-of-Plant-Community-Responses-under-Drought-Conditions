# model_lstm_seq2seq.py
# Encoder–Decoder LSTM 多步预测（Seq2Seq）
# 与 data_utils.py 对齐数据 & 指标逻辑，同时加入 WandB（可选）和可视化
import os
import tempfile
import shutil


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
import os
import tempfile
import shutil
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ====================== 2. 可选导入 WandB ======================

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
    print("[WandB] 未安装 wandb，自动关闭实验记录功能。")

# ====================== 3. 其他依赖 & data_utils ======================

import matplotlib.pyplot as plt

from data_utils import (
    DataConfig,
    load_and_prepare_data,
    split_by_time,
    make_sliding_windows,
    compute_all_metrics,
    export_metrics_to_csv,
    compute_regression_metrics,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ====================== 4. 可视化函数 ======================

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


def plot_pred_vs_true(dates, y_true, y_pred, save_path, title="Pred vs True NDVI (step 1)"):
    plt.figure()
    plt.plot(dates, y_true, label="True NDVI (step 1)")
    plt.plot(dates, y_pred, label="Pred NDVI (step 1)")
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[保存] 预测对比图: {save_path}")


def plot_residual_hist(y_true, y_pred, save_path, title="Residual Histogram (step 1)"):
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


# ====================== 5. 构造多步标签（从单步滑窗扩展） ======================

def build_multi_step_targets(
    target_series: np.ndarray,
    idx_arr: np.ndarray,
    horizon_multi: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据单步滑窗得到的 target 索引 idx_arr，构建多步 NDVI 标签:
    - 单步时 y[i] = target_series[idx_arr[i]] = NDVI_{t+1}
    - 多步时 Y[i, :] = [NDVI_{t+1}, NDVI_{t+2}, ..., NDVI_{t+H}]
    最后几条因未来不足 H 步会被丢弃。
    返回:
        Y_multi: (N_valid, H)
        idx_new: 对应的 idx_arr（长度 N_valid）
        valid_mask: 原 idx_arr 上的布尔 mask
    """
    max_valid_idx = len(target_series) - horizon_multi
    valid_mask = idx_arr <= max_valid_idx

    idx_new = idx_arr[valid_mask]
    Y_list = []
    for idx in idx_new:
        seq = target_series[idx: idx + horizon_multi]  # 长度 H
        Y_list.append(seq)

    Y_multi = np.stack(Y_list, axis=0)  # (N_valid, H)
    return Y_multi, idx_new, valid_mask


# ====================== 6. Encoder / Decoder / Seq2Seq 定义 ======================

class EncoderLSTM(nn.Module):
    """
    Encoder 输入：多变量序列 X (B, L, F)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src: torch.Tensor):
        outputs, (hidden, cell) = self.lstm(src)
        return outputs, (hidden, cell)


class DecoderLSTM(nn.Module):
    """
    Decoder 每一步只输入 1 维 NDVI 标量（上一时刻真实或预测值），逐步生成未来 NDVI。
    """
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,         # 解码器输入：前一时刻 NDVI（标量）
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)  # 输出一个 NDVI 标量

    def forward(self, input_step: torch.Tensor, hidden, cell):
        """
        input_step: (B, 1, 1)
        hidden, cell: (num_layers, B, H)
        """
        output, (hidden, cell) = self.lstm(input_step, (hidden, cell))
        pred = self.fc(output[:, -1, :])    # (B, 1)
        return pred, hidden, cell


class Seq2SeqLSTM(nn.Module):
    """
    标准 Encoder–Decoder LSTM：
    - Encoder 读入过去 L 步多变量特征 (B, L, F)
    - Decoder 自回归生成未来 H 步 NDVI (B, H)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        horizon_multi: int = 3,
        teacher_forcing_ratio: float = 0.5,
    ):
        super().__init__()
        self.encoder = EncoderLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = DecoderLSTM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.horizon_multi = horizon_multi
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        src: (B, L, F)   —— 编码器输入（多变量，已标准化）
        targets: (B, H)  —— 未来 NDVI 序列（只在训练/验证时用于 teacher forcing）
        返回:
            outputs: (B, H) —— 预测的未来 NDVI 序列
        """
        batch_size = src.size(0)
        device = src.device

        _, (hidden, cell) = self.encoder(src)

        # 解码起始输入：用 0 作为起始 token
        input_step = torch.zeros(batch_size, 1, 1, device=device)

        outputs = torch.zeros(batch_size, self.horizon_multi, device=device)

        for t in range(self.horizon_multi):
            pred_step, hidden, cell = self.decoder(input_step, hidden, cell)  # pred_step: (B, 1)
            pred_step = pred_step.squeeze(-1)  # (B,)

            outputs[:, t] = pred_step

            # 决定下一步输入（teacher forcing）
            if self.training and targets is not None and t < self.horizon_multi - 1:
                use_teacher = torch.rand(1).item() < self.teacher_forcing_ratio
                if use_teacher:
                    next_input = targets[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                else:
                    next_input = pred_step.unsqueeze(-1).unsqueeze(-1)      # (B, 1, 1)
            else:
                # 推理或最后一步：自回归
                next_input = pred_step.unsqueeze(-1).unsqueeze(-1)          # (B, 1, 1)

            input_step = next_input

        return outputs  # (B, H)


# ====================== 7. DataLoader / 训练 / 评估 ======================

def build_dataloader_multi(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    X_tensor = torch.from_numpy(X.astype(np.float32))
    Y_tensor = torch.from_numpy(Y.astype(np.float32))
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_one_epoch_seq2seq(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)   # (B, H)

        optimizer.zero_grad()
        Y_pred = model(X_batch, Y_batch)  # (B, H)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)


def evaluate_seq2seq(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    teacher_forcing: bool = False,
):
    """
    如果 teacher_forcing=False，则在验证/测试时完全自回归推理。
    """
    model.eval()
    running_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            if teacher_forcing:
                Y_pred = model(X_batch, Y_batch)
            else:
                Y_pred = model(X_batch, None)

            loss = criterion(Y_pred, Y_batch)

            running_loss += loss.item() * X_batch.size(0)
            all_true.append(Y_batch.cpu().numpy())
            all_pred.append(Y_pred.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    Y_true = np.concatenate(all_true, axis=0)   # (N, H)
    Y_pred = np.concatenate(all_pred, axis=0)   # (N, H)
    return avg_loss, Y_true, Y_pred


# ====================== 8. 主流程：多步预测 + 对齐基线 + WandB + 可视化 ======================

def run_encoder_decoder_lstm(
    horizon_multi: int = 3,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_epochs: int = 80,
    patience: int = 10,
    teacher_forcing_ratio: float = 0.5,
) -> Dict[str, float]:
    """
    Encoder–Decoder LSTM 多步预测：
    - 训练时预测 H 步 NDVI
    - 评估时，用第 1 步 (step-1) 与基线 y_test_reg 对齐
    加入 WandB（可选）和可视化。
    """
    model_name = "LSTM_EncoderDecoder_step1"

    cfg = DataConfig(
        data_path="D:/12138/大数据系统原理与应用/期末作业/数据/GreatPlains_8day_merged.csv",
        train_end="2014-12-31",
        val_end="2018-12-31",
        window_size=12,
        horizon=1,           # 滑窗仍用单步 (保证 idx 与基线统一)
        low_quantile=0.2,
    )

    # 1. 数据与划分（这里也是多变量：data_utils 里定义的所有数值列 + 时间特征）
    df = load_and_prepare_data(cfg)
    print("Encoder-Decoder 使用的多变量特征列 cfg.feature_cols：")
    print(cfg.feature_cols)

    df_train, df_val, df_test = split_by_time(df, cfg)

    series_train = df_train[cfg.target_col].values
    series_val   = df_val[cfg.target_col].values
    series_test  = df_test[cfg.target_col].values

    # 2. 单步滑窗（为了得到统一的 idx 与标准化特征）
    X_train, y_train_reg, idx_train, dates_train, scaler = make_sliding_windows(
        df_train, cfg, scaler=None, fit_scaler=True
    )
    X_val, y_val_reg, idx_val, dates_val, _ = make_sliding_windows(
        df_val, cfg, scaler=scaler, fit_scaler=False
    )
    X_test, y_test_reg, idx_test, dates_test, _ = make_sliding_windows(
        df_test, cfg, scaler=scaler, fit_scaler=False
    )

    print("Single-step sliding windows:")
    print("  Train:", X_train.shape)  # (N, L, F>1)
    print("  Val:  ", X_val.shape)
    print("  Test: ", X_test.shape)

    # 3. 基于单步 idx 构造多步标签
    Y_train, idx_train_new, mask_train = build_multi_step_targets(
        target_series=series_train,
        idx_arr=idx_train,
        horizon_multi=horizon_multi,
    )
    Y_val, idx_val_new, mask_val = build_multi_step_targets(
        target_series=series_val,
        idx_arr=idx_val,
        horizon_multi=horizon_multi,
    )
    Y_test, idx_test_new, mask_test = build_multi_step_targets(
        target_series=series_test,
        idx_arr=idx_test,
        horizon_multi=horizon_multi,
    )

    # 同步裁剪 X / 单步标签 / 时间
    X_train = X_train[mask_train]
    X_val   = X_val[mask_val]
    X_test  = X_test[mask_test]

    y_train_reg_step1 = y_train_reg[mask_train]  # 等于 Y_train[:, 0]
    y_val_reg_step1   = y_val_reg[mask_val]
    y_test_reg_step1  = y_test_reg[mask_test]

    dates_test_step1  = dates_test[mask_test]

    print("Multi-step dataset shapes:")
    print("  Train X:", X_train.shape, " Y:", Y_train.shape)
    print("  Val   X:", X_val.shape,   " Y:", Y_val.shape)
    print("  Test  X:", X_test.shape,  " Y:", Y_test.shape)

    # 4. 低 NDVI 阈值（仍然基于训练集单步标签）
    threshold = float(np.quantile(y_train_reg_step1, cfg.low_quantile))
    print(f"Low-NDVI threshold (quantile={cfg.low_quantile}): {threshold:.4f}")

    # 5. DataLoader
    train_loader = build_dataloader_multi(X_train, Y_train, batch_size=batch_size, shuffle=True)
    val_loader   = build_dataloader_multi(X_val,   Y_val,   batch_size=batch_size, shuffle=False)
    test_loader  = build_dataloader_multi(X_test,  Y_test,  batch_size=batch_size, shuffle=False)

    # 6. 模型、损失、优化器
    input_size = X_train.shape[-1]   # 多变量数 F
    print(f"Encoder-Decoder LSTM 输入维度 input_size = {input_size} (多变量数)")

    model = Seq2SeqLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon_multi=horizon_multi,
        teacher_forcing_ratio=teacher_forcing_ratio,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 7. 初始化 WandB（可选）
    WANDB_ENABLED = False
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="ndvi_drought_greatplains",
                name=f"{model_name}_H{horizon_multi}_hid{hidden_size}_L{num_layers}",
                # 如果你经常登录失败，也可以加 mode="offline"
                # mode="offline",
                config={
                    "model": model_name,
                    "horizon_multi": horizon_multi,
                    "window_size": cfg.window_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "lr": lr,
                    "num_epochs": num_epochs,
                    "teacher_forcing_ratio": teacher_forcing_ratio,
                    "train_end": cfg.train_end,
                    "val_end": cfg.val_end,
                    "features": cfg.feature_cols,
                },
            )
            WANDB_ENABLED = True
            print("[WandB] 初始化成功，将记录本次 Seq2Seq 实验。")
        except Exception as e:
            print("[WandB] 初始化失败，将在不记录的情况下继续运行。错误信息：", e)
    else:
        print("[WandB] 当前环境不可用，跳过 WandB 记录。")

    # 8. 训练（对整段 H 步序列做 MSE），记录 loss 曲线
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch_seq2seq(model, train_loader, optimizer, criterion, DEVICE)
        # 验证阶段用 teacher forcing 以稳定验证损失
        val_loss, _, _ = evaluate_seq2seq(
            model, val_loader, criterion, DEVICE, teacher_forcing=True
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if WANDB_ENABLED:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print("  -> New best seq2seq model (saved).")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 9. 加载最佳权重，在测试集上自回归推理
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, Y_true_test, Y_pred_test = evaluate_seq2seq(
        model, test_loader, criterion, DEVICE, teacher_forcing=False
    )
    print(f"Test MSE loss (multi-step sequence): {test_loss:.6f}")

    # 10. 与基线对齐：只取第 1 步预测进行统一评估
    y_true_step1 = Y_true_test[:, 0]   # NDVI_{t+1}
    y_pred_step1 = Y_pred_test[:, 0]   # 预测的 NDVI_{t+1}

    metrics_step1 = compute_all_metrics(
        y_true=y_true_step1,
        y_pred=y_pred_step1,
        threshold=threshold,
    )
    print("Encoder–Decoder LSTM (step-1) metrics on aligned test:", metrics_step1)

    if WANDB_ENABLED:
        wandb.log({
            "test_loss_multi_seq": test_loss,
            "test_MAE_step1": metrics_step1["MAE"],
            "test_RMSE_step1": metrics_step1["RMSE"],
            "test_F1_step1": metrics_step1["F1-score"],
            "test_ROC_AUC_step1": metrics_step1["ROC-AUC"],
        })

    # 11. 可选：输出每个 step 的 MAE / RMSE（多步预测分析）
    for h in range(Y_true_test.shape[1]):
        step_metrics = compute_regression_metrics(Y_true_test[:, h], Y_pred_test[:, h])
        print(f"  Step {h+1} MAE={step_metrics['MAE']:.4f}, RMSE={step_metrics['RMSE']:.4f}")

    # 12. 导出 CSV（主对比：只导出 step-1 指标，保证与基线可比）
    metrics_out_path = "D:/12138/大数据系统原理与应用/期末作业/模型评估/lstm_encdec/metrics_lstm_encdec_step1.csv"
    base_dir = os.path.dirname(metrics_out_path)
    os.makedirs(base_dir, exist_ok=True)

    extra_info = {
        "horizon_multi": str(horizon_multi),
        "hidden_size": str(hidden_size),
        "num_layers": str(num_layers),
        "dropout": str(dropout),
        "batch_size": str(batch_size),
        "lr": str(lr),
        "best_val_mse_seq": f"{best_val_loss:.6f}",
        "threshold": threshold,
    }
    export_metrics_to_csv(
        model_name=model_name,
        metrics=metrics_step1,
        output_path=metrics_out_path,
        extra_info=extra_info,
    )

    # 13. 保存第 1 步预测结果（方便画图 & 报告）
    pred_out_path = os.path.join(base_dir, "pred_lstm_encdec_step1.csv")
    pred_df = {
        "date": dates_test_step1,
        "y_true_step1": y_true_step1,
        "y_pred_step1": y_pred_step1,
    }
    import pandas as pd
    pred_df = pd.DataFrame(pred_df)
    pred_df.to_csv(pred_out_path, index=False)
    print(f"[保存] 第 1 步预测结果: {pred_out_path}")

    # 14. 画图：loss 曲线 + 第 1 步预测 vs 真实 + 残差直方图
    figs_dir = os.path.join(base_dir, "figs_lstm_encdec")
    os.makedirs(figs_dir, exist_ok=True)

    loss_fig_path = os.path.join(figs_dir, "lstm_encdec_loss.png")
    plot_loss_curves(train_losses, val_losses, loss_fig_path,
                     title=f"{model_name} Loss (Seq2Seq)")

    pred_fig_path = os.path.join(figs_dir, "lstm_encdec_pred_vs_true_step1.png")
    plot_pred_vs_true(
        dates_test_step1, y_true_step1, y_pred_step1,
        pred_fig_path,
        title=f"{model_name} Test NDVI (step 1)"
    )

    resid_fig_path = os.path.join(figs_dir, "lstm_encdec_residuals_step1.png")
    plot_residual_hist(
        y_true_step1, y_pred_step1,
        resid_fig_path,
        title=f"{model_name} Residuals (step 1)"
    )

    # 15. 把图传到 WandB
    if WANDB_ENABLED:
        wandb.log({
            "loss_curve": wandb.Image(loss_fig_path),
            "pred_vs_true_step1": wandb.Image(pred_fig_path),
            "residual_hist_step1": wandb.Image(resid_fig_path),
        })
        wandb.finish()

    print("全部完成 Encoder–Decoder LSTM 训练 / 评估 / 可视化。")

    return metrics_step1


def main():
    print("=== Encoder–Decoder LSTM (multi-step, multi-variate) ===")
    # 你可以把 horizon_multi 改成 3 或 6，分别对应 24 天 & 48 天
    run_encoder_decoder_lstm(
        horizon_multi=3,          # H = 3 (未来 3 个 8 日时间步)
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        batch_size=32,
        lr=1e-3,
        num_epochs=80,
        patience=10,
        teacher_forcing_ratio=0.5,
    )


if __name__ == "__main__":
    main()
