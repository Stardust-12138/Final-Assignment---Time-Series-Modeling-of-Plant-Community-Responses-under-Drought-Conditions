# model_lstm_many_to_one.py
# 多变量 LSTM 单步预测（Many-to-One）
# 与 data_utils.py 中基线模型使用完全一致的数据划分和指标定义
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
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    roc_auc_score,
)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import wandb


# ====================== 一些全局设置 ======================
plt.switch_backend("Agg")  # 不弹出窗口，直接存图

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ====================== 数据配置 ======================
@dataclass
class DataConfig:
    data_path: str = "D:/12138/大数据系统原理与应用/期末作业/数据/GreatPlains_8day_merged.csv"  # 你的 8 日聚合 CSV
    date_col: str = "date"
    target_col: str = "ndvi"

    # 时间划分
    train_end: str = "2014-12-31"
    val_end: str = "2018-12-31"

    # 时序设置
    window_size: int = 12     # 12*8 天历史
    horizon: int = 1          # 预测 1 个 8 日步 NDVI

    # 低 NDVI 阈值的分位数
    low_quantile: float = 0.2


# ====================== 数据加载 & 预处理 ======================
def load_and_prepare_data(cfg: DataConfig) -> pd.DataFrame:
    """读取 CSV，按时间排序，构建多变量特征（含季节特征）"""

    df = pd.read_csv(cfg.data_path)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    # 数值列全部作为基础特征（包括 ndvi 和几个气象量）
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 添加季节性特征（正余弦）
    df["doy"] = df[cfg.date_col].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.0)

    feature_cols = num_cols + ["doy_sin", "doy_cos"]

    # 确保 target_col 在列中
    assert cfg.target_col in df.columns, f"目标列 {cfg.target_col} 不在 CSV 中！"

    print("全部特征列（包含目标）：", feature_cols)
    return df, feature_cols


def split_by_time(
    df: pd.DataFrame, cfg: DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间切分 train / val / test"""

    train_end = pd.to_datetime(cfg.train_end)
    val_end = pd.to_datetime(cfg.val_end)

    df_train = df[df[cfg.date_col] <= train_end].copy()
    df_val = df[(df[cfg.date_col] > train_end) & (df[cfg.date_col] <= val_end)].copy()
    df_test = df[df[cfg.date_col] > val_end].copy()

    print(f"Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    return df_train, df_val, df_test


def make_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: DataConfig,
    scaler: StandardScaler = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    输入一个子 DataFrame（train / val / test），
    输出：
        X: (N, L, F)
        y: (N,)
        idx: 每个样本对应的行索引（在 df 内部的索引）
        dates: 每个样本预测目标对应的时间
    """
    values = df[feature_cols].values.astype(np.float32)
    target = df[cfg.target_col].values.astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        scaler.fit(values)

    values_scaled = scaler.transform(values)

    L = cfg.window_size
    H = cfg.horizon
    N = len(df)

    X_list = []
    y_list = []
    idx_list = []
    date_list = []

    for end_idx in range(L, N - H + 1):
        # 历史窗口 [end_idx-L, end_idx)
        start_idx = end_idx - L
        X_window = values_scaled[start_idx:end_idx, :]
        # 预测未来 H 步的第一个 NDVI
        y_target = target[end_idx + H - 1]

        X_list.append(X_window)
        y_list.append(y_target)
        idx_list.append(end_idx + H - 1)
        date_list.append(df.iloc[end_idx + H - 1][cfg.date_col])

    X = np.stack(X_list, axis=0)           # (N_samples, L, F)
    y = np.array(y_list, dtype=np.float32) # (N_samples,)
    idx = np.array(idx_list, dtype=np.int64)
    dates = np.array(date_list)

    return X, y, idx, dates, scaler


# ====================== 指标计算 & 导出 ======================
def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    输出： MAE / RMSE / F1 / ROC-AUC
    定义： y < threshold -> 1 （干旱 / 低 NDVI）
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    y_true_bin = (y_true < threshold).astype(int)
    y_pred_bin = (y_pred < threshold).astype(int)

    f1 = f1_score(y_true_bin, y_pred_bin)

    # 越低越干旱，所以用 -y_pred 做“干旱分数”
    try:
        auc = roc_auc_score(y_true_bin, -y_pred)
    except ValueError:
        auc = float("nan")

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "F1-score": f1,
        "ROC-AUC": auc,
    }
    return metrics


def export_metrics_to_csv(
    model_name: str,
    metrics: Dict[str, float],
    output_path: str,
    extra_info: Dict[str, float] = None,
):
    record = {"model": model_name}
    record.update(metrics)
    if extra_info is not None:
        record.update(extra_info)

    df = pd.DataFrame([record])
    out_dir = os.path.dirname(output_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[保存] 指标写入: {output_path}")


# ====================== Dataset & LSTM 模型 ======================
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, L, F)
        self.y = torch.from_numpy(y).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMManyToOne(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, L, F)
        out, (h_n, c_n) = self.lstm(x)  # h_n: (num_layers, B, H)
        h_last = h_n[-1]                # (B, H)
        y = self.fc(h_last)             # (B, 1)
        return y


# ====================== 训练 & 验证 ======================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            all_true.append(y_batch.cpu().numpy())
            all_pred.append(y_pred.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    y_true = np.concatenate(all_true, axis=0).squeeze(-1)
    y_pred = np.concatenate(all_pred, axis=0).squeeze(-1)
    return avg_loss, y_true, y_pred


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


# ====================== 主流程：训练 + 可视化 + WandB ======================
def main():
    # ---------- 1. 基本配置 ----------
    cfg = DataConfig(
        data_path="D:/12138/大数据系统原理与应用/期末作业/数据/GreatPlains_8day_merged.csv",
        train_end="2014-12-31",
        val_end="2018-12-31",
        window_size=12,
        horizon=1,
        low_quantile=0.2,
    )

    hidden_size = 64
    num_layers = 1
    dropout = 0.2
    batch_size = 32
    lr = 1e-3
    num_epochs = 80
    patience = 10

    model_name = "LSTM_ManyToOne"

    # ---------- 2. 读取数据 & 划分 ----------
    df, feature_cols = load_and_prepare_data(cfg)
    df_train, df_val, df_test = split_by_time(df, cfg)

    # ---------- 3. 构建滑动窗口 ----------
    X_train, y_train, idx_train, dates_train, scaler = make_sliding_windows(
        df_train, feature_cols, cfg, scaler=None, fit_scaler=True
    )
    X_val, y_val, idx_val, dates_val, _ = make_sliding_windows(
        df_val, feature_cols, cfg, scaler=scaler, fit_scaler=False
    )
    X_test, y_test, idx_test, dates_test, _ = make_sliding_windows(
        df_test, feature_cols, cfg, scaler=scaler, fit_scaler=False
    )

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # 阈值按训练集 NDVI 分位数
    threshold = np.quantile(y_train, cfg.low_quantile)
    print(f"低 NDVI 阈值 (train {cfg.low_quantile} 分位): {threshold:.4f}")

    # ---------- 4. 构建 Dataset / DataLoader ----------
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ---------- 5. 定义模型 / 损失 / 优化器 ----------
    input_size = X_train.shape[-1]
    model = LSTMManyToOne(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------- 6. 初始化 WandB ----------
    wandb.init(
        project="ndvi_drought_greatplains",
        name=f"{model_name}_hid{hidden_size}_L{num_layers}",
        config={
            "model": model_name,
            "window_size": cfg.window_size,
            "horizon": cfg.horizon,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "train_end": cfg.train_end,
            "val_end": cfg.val_end,
            "features": feature_cols,
        },
    )

    # ---------- 7. 训练循环（含早停 & WandB 记录） ----------
    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_epochs = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )

        # WandB 记录
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # 早停
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Val loss {patience} 个 epoch 没提升，提前停止。")
                break

    # 使用最优模型
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ---------- 8. 在测试集上评估 ----------
    test_loss, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, DEVICE)
    metrics = compute_all_metrics(y_test_true, y_test_pred, threshold)

    print("测试集 Loss:", test_loss)
    print("测试集指标:", metrics)

    # WandB 记录测试指标
    wandb.log({
        "test_loss": test_loss,
        "test_MAE": metrics["MAE"],
        "test_RMSE": metrics["RMSE"],
        "test_F1": metrics["F1-score"],
        "test_ROC_AUC": metrics["ROC-AUC"],
    })

    # ---------- 9. 保存指标 & 预测结果 ----------
    os.makedirs("D:/12138/大数据系统原理与应用/期末作业/模型评估", exist_ok=True)

    export_metrics_to_csv(
        model_name=model_name,
        metrics=metrics,
        output_path=f"D:/12138/大数据系统原理与应用/期末作业/模型评估/metrics_{model_name}.csv",
        extra_info={"test_loss": test_loss, "threshold": threshold},
    )

    # 保存预测 vs 真实，用于以后画图或别的模型对比
    pred_df = pd.DataFrame({
        "date": dates_test,
        "y_true": y_test_true,
        "y_pred": y_test_pred,
    })
    pred_path = f"D:/12138/大数据系统原理与应用/期末作业/模型评估/pred_{model_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[保存] 预测结果: {pred_path}")

    # ---------- 10. 可视化 ----------
    figs_dir = "D:/12138/大数据系统原理与应用/期末作业/模型评估/figs"
    os.makedirs(figs_dir, exist_ok=True)

    # Loss 曲线
    loss_fig_path = os.path.join(figs_dir, f"{model_name}_loss.png")
    plot_loss_curves(train_losses, val_losses, loss_fig_path, title=f"{model_name} Loss")

    # 预测 vs 真实
    pred_fig_path = os.path.join(figs_dir, f"{model_name}_pred_vs_true.png")
    plot_pred_vs_true(dates_test, y_test_true, y_test_pred, pred_fig_path,
                      title=f"{model_name} Test NDVI")

    # 残差直方图
    resid_fig_path = os.path.join(figs_dir, f"{model_name}_residuals.png")
    plot_residual_hist(y_test_true, y_test_pred, resid_fig_path,
                       title=f"{model_name} Residuals")

    # 把图也传到 WandB
    wandb.log({
        "loss_curve": wandb.Image(loss_fig_path),
        "pred_vs_true": wandb.Image(pred_fig_path),
        "residual_hist": wandb.Image(resid_fig_path),
    })

    wandb.finish()
    print("全部完成。")


if __name__ == "__main__":
    main()
