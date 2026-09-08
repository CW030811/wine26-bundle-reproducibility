"""
多层数 EdgeScoringGCN 模型训练脚本（采用 Train_edge_GCN_single_update_edge 的无向+逐层边更新结构）

目标：
- 使用“无向 message passing + 每层边特征更新”的新网络结构
- 训练 3 层和 4 层模型，每层数跑多个 seed
- 训练产物统一存放到新的输出目录，避免覆盖旧结果
"""

from __future__ import annotations
import csv
import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GENConv
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import glob
import time
from datetime import datetime
import msgpack
import msgpack_numpy as mnp

class EdgeScoringGCN(nn.Module):
    """Layer-wise edge updates with undirected message passing（与 Train_edge_GCN_single_update_edge 保持一致）."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 128,
        num_layers: int = 2,
        edge_dim: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.edge_updates = nn.ModuleList()

        current_node_dim = in_channels
        current_edge_dim = edge_dim

        for _ in range(num_layers):
            # node update takes current edge feature dim
            self.convs.append(GENConv(current_node_dim, hidden_channels, edge_dim=current_edge_dim))
            # edge update consumes updated nodes + previous edge feature
            self.edge_updates.append(nn.Sequential(
                nn.Linear(hidden_channels * 2 + current_edge_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            ))
            current_node_dim = hidden_channels
            current_edge_dim = hidden_channels

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.edge_head = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dst = edge_index

        h = x
        current_edge_attr = edge_attr

        for layer_idx in range(self.num_layers):
            # undirected message passing for stability; rebuild because edge_attr changes per layer
            undirected_edge_index, undirected_edge_attr = to_undirected(
                edge_index, edge_attr=current_edge_attr, num_nodes=x.size(0)
            )

            h = self.act(self.convs[layer_idx](h, undirected_edge_index, undirected_edge_attr))
            h = self.dropout(h)

            cat_input = torch.cat([h[src], h[dst], current_edge_attr], dim=-1)
            current_edge_attr = self.edge_updates[layer_idx](cat_input)

        logits = self.edge_head(self.dropout(current_edge_attr)).squeeze(-1)
        out = { 'edge_logits': logits }

        if hasattr(data, 'product_num') and hasattr(data, 'segment_num'):
            try:
                n = int(data.product_num)
                m = int(data.segment_num)
                if logits.numel() == n * m:
                    out['logit_matrix'] = logits.view(n, m)
            except Exception:
                pass

        return out


# 修复PyTorch 2.6的weights_only问题 - 添加安全全局变量
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([EdgeScoringGCN])

# Matplotlib 简单中文支持（可选）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_data(file_path):
    """读取数据文件并转换为图数据格式（从 bundle_utils.py 复制）"""
    with open(file_path, 'rb') as f:
        loaded_data = msgpack.load(f, object_hook=mnp.decode, strict_map_key=False)
    
    # 提取数据
    product_num = loaded_data['product_num']
    segment_num = loaded_data['segment_num']
    unit_cs = loaded_data['unit_cs']
    ship_cs = loaded_data['ship_cs']
    unit_us = loaded_data['unit_us']
    Ns = loaded_data['Ns']
    opt_bundles = loaded_data['opt_bundles']
    opt_prices = loaded_data['opt_prices']
    opt_rev = loaded_data['opt_rev']
    running_time = loaded_data['running_time']
    gap = loaded_data['gap']
    
    # 构建图数据
    node_num = product_num + segment_num
    feature_mat = np.zeros((node_num, 4))
    feature_mat[:product_num, 0] = unit_cs[0, :]
    feature_mat[:product_num, 1] = np.average(unit_us, axis=0)
    feature_mat[product_num:, 2] = Ns[:, 0]
    feature_mat[product_num:, 3] = ship_cs[:, 0]
    
    x = torch.tensor(feature_mat, dtype=torch.float)
    
    # 构建边
    left_nodes = []
    right_nodes = []
    weights = []
    for i in range(product_num):
        for j in range(segment_num):
            left_nodes.append(i)
            right_nodes.append(j + product_num)
            weights.append([unit_us[j, i]])
    
    edge_index = torch.tensor([left_nodes, right_nodes], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    # 标签
    label = torch.tensor(np.append(np.array(opt_bundles).T, -np.ones((segment_num, segment_num), dtype=int), axis=0), dtype=torch.long)
    side_ind = torch.tensor(np.array([1]*product_num + [0]*segment_num)[:, np.newaxis], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=label, side_ind=side_ind)
    meta = (product_num, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap)
    
    return data, meta


def _split_indices(n: int, val_ratio: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    """手动划分训练/验证索引，避免额外依赖 sklearn
    返回 (train_indices, val_indices)
    """
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_size = int(round(n * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    # 防止极端情况下为空
    if len(train_indices) == 0 and n > 0:
        train_indices, val_indices = indices[:-1], indices[-1:]
    return train_indices, val_indices


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _clean_previous_run_artifacts(model_dir: str, keep_best: bool = True, model_pattern: str = "*") -> None:
    """清理上一次训练在 dataset 目录下生成的工件，避免累积失效文件。
    - 默认保留 best_model*.pt
    - 清理 model*.pt、model-*.pt、train_loss*.csv、val_loss*.csv、training_curves*.png
    """
    patterns = [
        os.path.join(model_dir, f"model-{model_pattern}-*.pt"),
        os.path.join(model_dir, f"train_loss{model_pattern}*.csv"),
        os.path.join(model_dir, f"val_loss{model_pattern}*.csv"),
        os.path.join(model_dir, f"training_curves{model_pattern}*.png"),
    ]
    target_model_pt = os.path.join(model_dir, f"model_edge{model_pattern}.pt")
    if os.path.exists(target_model_pt):
        try:
            os.remove(target_model_pt)
        except Exception:
            pass

    for pat in patterns:
        for f in glob.glob(pat):
            try:
                os.remove(f)
            except Exception:
                pass


def _upsert_seed_training_summary(summary_path: str, row: dict) -> None:
    """Write one summary row per (num_layers, seed), replacing older runs of the same seed."""
    fieldnames = [
        "num_layers",
        "seed",
        "epochs_completed",
        "batch_size",
        "learning_rate",
        "dropout",
        "weight_decay",
        "grad_clip",
        "final_train_loss",
        "best_val_loss",
        "best_model_path",
    ]

    rows = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for existing in reader:
                    if (
                        str(existing.get("num_layers")) == str(row["num_layers"])
                        and str(existing.get("seed")) == str(row["seed"])
                    ):
                        continue
                    rows.append(existing)
        except Exception:
            rows = []

    rows.append({key: row.get(key, "") for key in fieldnames})

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _attach_edge_labels_to_data(data, meta):
    """根据 meta 内的 opt_bundles 为每条边生成监督标签。
    - 边的构造顺序在 read_data 中是：for i in products, for j in segments
    - 对应标签应为：label[i,j] = opt_bundles[j, i]
    """
    product_num = int(meta[0])
    segment_num = int(meta[1])
    opt_bundles = np.asarray(meta[6])  # 形状 (segment_num, product_num)
    # 向量化：转置后按行展平，顺序与 read_data 中边的构造相同（先产品 i，再 segment j）
    edge_label = torch.tensor(opt_bundles.T.reshape(-1), dtype=torch.float)
    data.edge_label = edge_label
    # 该训练脚本不使用 data.y。为避免不同 m 导致 DataLoader 在拼接 y 时维度不匹配，将其置为空张量。
    try:
        data.y = torch.empty(0, dtype=torch.long)
    except Exception:
        pass
    data.product_num = product_num
    data.segment_num = segment_num
    return data


def train(
    data_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    train_subdir: str = "data/deterministic/train_m10n10_correct_1e_3",
    model_subdir: str = "results/base_training/models",
    charts_subdir: str = "results/base_training/charts",
    log_subdir: str = "results/base_training/logs",
    epochs: int = 500,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    hidden_channels: int = 128,
    num_layers: int = 3,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    segment_num: int = 10,           # 仅用于日志与兼容入参；模型不再依赖它
    save_interval: int = 100,
    val_ratio: float = 0.2,          # 8:2 划分
    split_seed: int = 2026,
    early_stopping_patience: int = 50,
    save_best_as_model_pt: bool = True,
    seed: int = 42,
    cleanup_before_train: bool = True,
) -> Tuple[torch.nn.Module, np.ndarray, Optional[np.ndarray]]:
    """训练 EdgeScoringGCN 模型
    参数说明同原 Training.py；segment_num 保留为兼容入参（模型无需 out_channels）。
    返回：model, train_loss_hist, val_loss_hist
    """
    seed_train_start_time = time.perf_counter()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
        try:
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception:
            pass
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA不可用，使用CPU训练")

    train_path = os.path.join(data_dir, train_subdir)
    # 将训练产物写入新的目录，避免覆盖旧结果
    model_dir = os.path.join(data_dir, model_subdir)
    charts_root = os.path.join(data_dir, charts_subdir)
    log_dir = os.path.join(data_dir, log_subdir)
    if os.path.exists(log_dir) and not os.path.isdir(log_dir):
        alt_log_dir = os.path.join(data_dir, "tensorboard_logs")
        print(f"⚠️ 检测到 {log_dir} 已存在且不是目录，改用 {alt_log_dir} 作为日志目录")
        log_dir = alt_log_dir
    _ensure_dirs(model_dir, charts_root, log_dir)
    print(f"📁 模型输出目录: {model_dir}")
    print(f"🖼️ 曲线输出目录: {charts_root}")
    print(f"📝 日志目录: {log_dir}")

    # 清理历史工件
    if cleanup_before_train:
        _clean_previous_run_artifacts(model_dir, keep_best=True)

    # TensorBoard 日志
    try:
        writer = SummaryWriter(log_dir=log_dir)
    except Exception as e:
        print(f"⚠️ TensorBoard 初始化失败，尝试回退。原因为: {e}")
        alt_log_dir = os.path.join(model_dir, "tb_logs")
        _ensure_dirs(alt_log_dir)
        try:
            writer = SummaryWriter(log_dir=alt_log_dir)
            print(f"ℹ️ 已回退到 {alt_log_dir} 作为日志目录")
        except Exception as e2:
            print(f"⚠️ 回退仍失败，将禁用TensorBoard日志。原因为: {e2}")
            class _DummyWriter:
                def add_scalar(self, *args, **kwargs):
                    pass
                def flush(self):
                    pass
                def close(self):
                    pass
            writer = _DummyWriter()

    # 加载数据
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据路径不存在: {train_path}")

    file_list = [f for f in os.listdir(train_path) if f != ".DS_Store"]
    file_list.sort()
    dataset = []

    print("Start reading the dataset…")
    for i, fname in enumerate(file_list, 1):
        dat, meta = read_data(os.path.join(train_path, fname))
        dat = _attach_edge_labels_to_data(dat, meta)
        dataset.append(dat)
        if i % 100 == 0:
            print(f"Loaded data: {i}/{len(file_list)}")
    print(f"📦 总计加载 {len(dataset)} 个样本")

    # 固定划分 8:2 训练/验证，避免不同训练 seed 使用不同 validation set。
    train_idx, val_idx = _split_indices(len(dataset), val_ratio, split_seed)
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]

    print("📊 数据划分结果：")
    print(f"  Split seed: {split_seed}")
    print(f"  训练集: {len(train_data)} ({(1-val_ratio)*100:.1f}%)")
    print(f"  验证集: {len(val_data)} ({val_ratio*100:.1f}%)")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data else None

    # 类别不平衡：基于训练集一次性统计 pos_weight，避免每个 batch 波动
    total_pos = 0.0
    total_cnt = 0
    for dat in train_data:
        el = dat.edge_label
        total_pos += float(el.sum().item())
        total_cnt += int(el.numel())
    pos_rate = (total_pos / max(1, total_cnt)) if total_cnt > 0 else 0.0
    pos_rate = float(min(max(pos_rate, 1e-6), 1 - 1e-6))
    pos_weight_value = (1.0 - pos_rate) / pos_rate
    pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float, device=device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(f"⚖️ 全局正例比例 pos_rate={pos_rate:.6f}, 使用 pos_weight={pos_weight_value:.3f}")

    # 初始化模型（无向逐层边更新结构）
    print(
        f"🏗️ 初始化 EdgeScoringGCN（无向逐层边更新）"
        f"(num_layers={num_layers}, hidden_channels={hidden_channels}, dropout={dropout})"
    )
    model = EdgeScoringGCN(
        in_channels=4,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        edge_dim=1,
        dropout=dropout,
    )
    model = model.to(device)
    print(f"📱 模型已移动到设备: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )
    print(
        f"🔧 Optimizer: AdamW(lr={learning_rate}, weight_decay={weight_decay}), "
        f"grad_clip={grad_clip}, scheduler=ReduceLROnPlateau"
    )

    def _eval(model_: torch.nn.Module, loader: Optional[DataLoader]) -> float:
        if loader is None or len(loader) == 0:
            return float("inf")
        model_.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model_(batch)
                logits = out['edge_logits'].view(-1)
                labels = batch.edge_label.to(logits.dtype).view(-1)
                loss = criterion_bce(logits, labels)
                total += float(loss.item())
        return total / max(1, len(loader))

    best_val = float("inf")
    patience = 0
    best_model_path = os.path.join(model_dir, f"best_model_edge_{num_layers}layer_seed{seed}.pt")

    train_hist: List[Tuple[int, float]] = []
    val_hist: List[Tuple[int, float]] = []

    print(f"\n开始训练 {epochs} 轮…")
    for epoch in range(epochs):
        # 训练
        model.train()
        total_train = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            logits = out['edge_logits'].view(-1)
            labels = batch.edge_label.to(logits.dtype).view(-1)
            optimizer.zero_grad()
            loss = criterion_bce(logits, labels)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_train += float(loss.item())

        avg_train = total_train / max(1, len(train_loader))
        writer.add_scalar("Loss/train", avg_train, epoch)
        train_hist.append((epoch, avg_train))

        # 验证
        avg_val = _eval(model, val_loader)
        if val_loader is not None and np.isfinite(avg_val):
            writer.add_scalar("Loss/validation", avg_val, epoch)
            val_hist.append((epoch, avg_val))
            scheduler.step(avg_val)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            # 最优模型保存 + 早停
            if avg_val < best_val - 1e-12:
                best_val = avg_val
                patience = 0
                model_cpu = model.cpu()
                torch.save(model_cpu, best_model_path)
                model = model_cpu.to(device)  # 移回继续训练
                print(f"  🎯 新最佳模型：val_loss={avg_val:.6f} (epoch={epoch})")
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print(f"  ⏹️ 触发早停（连续 {early_stopping_patience} 轮无改进）")
                    break
        else:
            scheduler.step(avg_train)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # 日志
        if epoch % 100 == 0:
            if val_loader is not None and np.isfinite(avg_val):
                print(
                    f"Epoch {epoch:3d} | train={avg_train:.6f} | "
                    f"val={avg_val:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}"
                )
            else:
                print(f"Epoch {epoch:3d} | train={avg_train:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # 中间 checkpoint
        if epoch % save_interval == 0 and epoch > 0:
            ckpt_path = os.path.join(model_dir, f"model-{num_layers}layer_seed{seed}-{epoch}.pt")
            model_cpu = model.cpu()
            torch.save(model_cpu, ckpt_path)
            model = model_cpu.to(device)
            # 可选：保存历史
            np.savetxt(os.path.join(model_dir, f"train_loss_{num_layers}layer_seed{seed}-{epoch}.csv"), np.array(train_hist), delimiter=",")
            if val_hist:
                np.savetxt(os.path.join(model_dir, f"val_loss_{num_layers}layer_seed{seed}-{epoch}.csv"), np.array(val_hist), delimiter=",")
            print(f"  💾 保存检查点: {ckpt_path}")

    # 结束后保存
    final_model_path = os.path.join(model_dir, f"model_edge_{num_layers}layer_seed{seed}.pt")
    if save_best_as_model_pt and os.path.exists(best_model_path):
        best_model = torch.load(best_model_path, weights_only=False)
        torch.save(best_model, final_model_path)
    else:
        model_cpu = model.cpu()
        torch.save(model_cpu, final_model_path)

    # 保存完整损失历史
    train_hist_np = np.array(train_hist, dtype=float)
    train_loss_csv = os.path.join(model_dir, f"train_loss_edge_{num_layers}layer_seed{seed}.csv")
    val_loss_csv = os.path.join(model_dir, f"val_loss_edge_{num_layers}layer_seed{seed}.csv")
    np.savetxt(train_loss_csv, train_hist_np, delimiter=",")
    if val_hist:
        val_hist_np = np.array(val_hist, dtype=float)
        np.savetxt(val_loss_csv, val_hist_np, delimiter=",")
    else:
        val_hist_np = None
        val_loss_csv = ""

    # 绘制曲线
    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    chart_dir = os.path.join(charts_root, f"run_{run_tag}_edge_{num_layers}layer_seed{seed}")
    _ensure_dirs(chart_dir)

    plt.figure(figsize=(10, 7))
    if val_hist:
        plt.plot(train_hist_np[:, 0], train_hist_np[:, 1], label="Training Loss", color="C0")
        plt.plot(val_hist_np[:, 0], val_hist_np[:, 1], label="Validation Loss", color="C3")
        title = (
            f"Training & Validation Loss ({num_layers}layers, seed={seed}, split={split_seed}, "
            f"epochs={epochs}, batch={batch_size}, lr={learning_rate}, dropout={dropout}, wd={weight_decay})"
        )
        plt.title(title)
    else:
        plt.plot(train_hist_np[:, 0], train_hist_np[:, 1], label="Training Loss", color="C0")
        title = (
            f"Training Loss ({num_layers}layers, seed={seed}, split={split_seed}, "
            f"epochs={epochs}, batch={batch_size}, lr={learning_rate}, dropout={dropout}, wd={weight_decay})"
        )
        plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Details")
    plt.tight_layout()
    fig_name = (
        f"loss_curves_{num_layers}layer_seed{seed}_split{split_seed}_e{epochs}_"
        f"b{batch_size}_h{hidden_channels}_lr{learning_rate}_drop{dropout}_wd{weight_decay}.png"
    )
    fig_path = os.path.join(chart_dir, fig_name)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    writer.flush()
    writer.close()

    seed_training_time = time.perf_counter() - seed_train_start_time
    final_train_loss = float(train_hist_np[-1, 1]) if train_hist_np.size else float("nan")
    summary_csv = os.path.join(model_dir, "seed_training_summary.csv")
    _upsert_seed_training_summary(
        summary_csv,
        {
            "num_layers": num_layers,
            "seed": seed,
            "epochs_completed": len(train_hist),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "final_train_loss": final_train_loss,
            "best_val_loss": best_val if np.isfinite(best_val) else "",
            "best_model_path": best_model_path if os.path.exists(best_model_path) else "",
        },
    )

    print("\n✅ 训练完成！")
    print(f"  层数: {num_layers}")
    print(f"  Seed: {seed}")
    print(f"  最终模型: {final_model_path}")
    if os.path.exists(best_model_path):
        print(f"  最佳模型: {best_model_path} (val={best_val:.6f})")
    print(f"  训练曲线: {fig_path}")
    print(f"  图表目录: {chart_dir}")
    print(f"  Seed训练时间: {seed_training_time:.2f} 秒 ({seed_training_time / 60:.2f} 分钟)")
    print(f"  Seed训练汇总CSV: {summary_csv}")

    return model, train_hist_np, val_hist_np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train multi-layer EdgeScoringGCN models with different seeds")
    parser.add_argument("--data_dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    parser.add_argument("--train_subdir", type=str, default="data/deterministic/train_m10n10_correct_1e_3")
    parser.add_argument("--model_subdir", type=str, default="results/base_training/models", help="模型保存目录")
    parser.add_argument("--charts_subdir", type=str, default="results/base_training/charts", help="训练曲线保存目录")
    parser.add_argument("--log_subdir", type=str, default="results/base_training/logs", help="TensorBoard日志目录")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--no_cleanup", action="store_true", help="禁用训练前清理旧工件")
    args = parser.parse_args()

    # 训练配置
    layers = [4]  # 训练 3 层和 4 层
    seeds = [1,2,3,4,5,6,7,8,9,10] 

    print("=" * 80)
    print("🚀 开始训练多层 EdgeScoringGCN 模型")
    print("=" * 80)
    total_train_start_time = time.perf_counter()

    for num_layers in layers:
        print(f"\n{'='*80}")
        print(f"📊 训练 {num_layers} 层模型")
        print(f"{'='*80}")
        
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"🌱 使用 seed={seed} 训练 {num_layers} 层模型")
            print(f"{'='*60}")
            
            train(
                data_dir=args.data_dir,
                train_subdir=args.train_subdir,
                model_subdir=args.model_subdir,
                charts_subdir=args.charts_subdir,
                log_subdir=args.log_subdir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                hidden_channels=args.hidden,
                num_layers=num_layers,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                segment_num=args.segments,
                save_interval=args.save_interval,
                val_ratio=args.val_ratio,
                split_seed=args.split_seed,
                early_stopping_patience=args.patience,
                cleanup_before_train=(
                    not args.no_cleanup
                    and num_layers == layers[0]
                    and seed == seeds[0]
                ),
                seed=seed,
            )
            
            print(f"\n✅ {num_layers}层 seed={seed} 训练完成\n")

    print("\n" + "=" * 80)
    print("🎉 所有模型训练完成！")
    print("=" * 80)
    print("\n生成的模型文件：")
    for num_layers in layers:
        for seed in seeds:
            print(f"  - best_model_edge_{num_layers}layer_seed{seed}.pt")
    total_train_time = time.perf_counter() - total_train_start_time
    print(f"\n总训练时间: {total_train_time:.2f} 秒 ({total_train_time / 60:.2f} 分钟)")
    print("=" * 80)
