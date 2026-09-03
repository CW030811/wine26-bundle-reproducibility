"""
CPBSD edge-level GCN training script with MB-derived x_kn supervision.

This keeps the graph structure close to the original Training_multi_layer.py:
- Product node features: [c_n, mean_k(v_kn), 0, 0]
- Customer node features: [0, 0, K, rho_k], rho_k = (1/N) * sum_n 1(v_kn - c_n > 0)
- Edge features: [v_kn]
- Edge supervision: x_kn in {0,1}, where x_kn = 1 iff product n is in the MB-optimal bundle chosen by customer k
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GENConv
from torch_geometric.utils import to_undirected

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TABLE5_DATA = PACKAGE_ROOT / "data" / "random_valuation_n5"
TABLE5_LABELS = PACKAGE_ROOT / "artifacts" / "random_valuation" / "labels"
TABLE5_TRAIN_OUT = PACKAGE_ROOT / "results" / "table5_training"


class EdgeScoringGCN(nn.Module):
    """Layer-wise edge updates with undirected message passing."""

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
            self.convs.append(GENConv(current_node_dim, hidden_channels, edge_dim=current_edge_dim))
            self.edge_updates.append(
                nn.Sequential(
                    nn.Linear(hidden_channels * 2 + current_edge_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                )
            )
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
            undirected_edge_index, undirected_edge_attr = to_undirected(
                edge_index,
                edge_attr=current_edge_attr,
                num_nodes=x.size(0),
            )
            h = self.act(self.convs[layer_idx](h, undirected_edge_index, undirected_edge_attr))
            h = self.dropout(h)
            edge_input = torch.cat([h[src], h[dst], current_edge_attr], dim=-1)
            current_edge_attr = self.edge_updates[layer_idx](edge_input)

        logits = self.edge_head(self.dropout(current_edge_attr)).squeeze(-1)
        out = {"edge_logits": logits}
        if hasattr(data, "product_num") and hasattr(data, "segment_num"):
            try:
                n = int(data.product_num)
                m = int(data.segment_num)
                if logits.numel() == n * m:
                    out["logit_matrix"] = logits.view(n, m)
            except Exception:
                pass
        return out


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([EdgeScoringGCN])

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _resolve_device(device_name: str) -> torch.device:
    requested = device_name.lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {device_name}")

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif mps_available:
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda, but CUDA is unavailable.")
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    if requested == "mps":
        if not mps_available:
            raise RuntimeError("Requested device=mps, but MPS is unavailable.")
        print("✅ Using MPS")
        return torch.device("mps")

    print("⚠️ Using CPU")
    return torch.device("cpu")


def _split_indices(n: int, val_ratio: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_size = int(round(n * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if len(train_indices) == 0 and n > 0:
        train_indices, val_indices = indices[:-1], indices[-1:]
    return train_indices, val_indices


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _clean_previous_run_artifacts(model_dir: str, suffix: str = "") -> None:
    patterns = [
        os.path.join(model_dir, f"model_edge_cpbsd_mb_x{suffix}.pt"),
        os.path.join(model_dir, f"best_model_edge_cpbsd_mb_x{suffix}.pt"),
        os.path.join(model_dir, f"train_loss_edge_cpbsd_mb_x{suffix}.csv"),
        os.path.join(model_dir, f"val_loss_edge_cpbsd_mb_x{suffix}.csv"),
        os.path.join(model_dir, f"metrics_edge_cpbsd_mb_x{suffix}.json"),
    ]
    for f in patterns:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


def _match_instance_result_pairs(
    instance_dir: str,
    result_dir: str,
    result_suffix: str = "__mb.json",
    max_files: int = 0,
) -> List[Tuple[Path, Path]]:
    instance_dir_path = Path(instance_dir)
    result_dir_path = Path(result_dir)
    pairs = []
    instance_map = {path.stem: path for path in instance_dir_path.rglob("*.msgpack")}
    result_files = sorted(result_dir_path.glob(f"*{result_suffix}"))
    for result_path in result_files:
        stem = result_path.name[: -len(result_suffix)]
        instance_path = instance_map.get(stem)
        if instance_path is None:
            continue
        pairs.append((instance_path, result_path))
        if max_files > 0 and len(pairs) >= max_files:
            break
    return pairs


def _read_result_manifest(path: str) -> List[Tuple[Path, Path]]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("has_solution", "").lower() not in {"true", "1"}:
                continue
            instance_path = Path(row["instance_path"])
            result_path = Path(row["result_path"])
            if instance_path.exists() and result_path.exists():
                rows.append((instance_path, result_path))
    return rows


def read_cpbsd_mb_data(instance_path: str, result_path: str) -> Tuple[Data, Dict]:
    with open(instance_path, "rb") as f:
        instance = msgpack.load(f, object_hook=mnp.decode, strict_map_key=False)
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    chosen_product_matrix = result.get("chosen_product_matrix")
    if chosen_product_matrix is None:
        raise ValueError(f"Result file has no chosen_product_matrix: {result_path}")

    v_kn = np.asarray(instance["valuation_samples_V"], dtype=float)
    c_n = np.asarray(instance["production_cost_c"], dtype=float)
    x_kn = np.asarray(chosen_product_matrix, dtype=float)

    if x_kn.ndim != 2:
        raise ValueError(f"Expected chosen_product_matrix to be 2D (K,N), got shape {x_kn.shape}")

    k_count, n_products = v_kn.shape
    if x_kn.shape != (k_count, n_products):
        raise ValueError(f"x_kn shape mismatch: expected {(k_count, n_products)}, got {x_kn.shape}")

    feature_mat = np.zeros((n_products + k_count, 4), dtype=float)
    feature_mat[:n_products, 0] = c_n
    feature_mat[:n_products, 1] = np.mean(v_kn, axis=0)

    rho_k = np.mean((v_kn - c_n[None, :]) > 0, axis=1)
    feature_mat[n_products:, 2] = float(k_count)
    feature_mat[n_products:, 3] = rho_k

    x = torch.tensor(feature_mat, dtype=torch.float)

    left_nodes = []
    right_nodes = []
    edge_attr = []
    edge_label = []
    for product_idx in range(n_products):
        for customer_idx in range(k_count):
            left_nodes.append(product_idx)
            right_nodes.append(customer_idx + n_products)
            edge_attr.append([v_kn[customer_idx, product_idx]])
            edge_label.append(x_kn[customer_idx, product_idx])

    edge_index = torch.tensor([left_nodes, right_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_label = torch.tensor(edge_label, dtype=torch.float)
    side_ind = torch.tensor(np.array([1] * n_products + [0] * k_count)[:, None], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        side_ind=side_ind,
        y=torch.empty(0, dtype=torch.long),
    )
    data.product_num = n_products
    data.segment_num = k_count

    meta = {
        "instance_path": str(instance_path),
        "result_path": str(result_path),
        "setup": instance.get("setup", {}),
        "product_num": n_products,
        "segment_num": k_count,
        "edge_label_mean": float(np.mean(edge_label.numpy())),
        "positive_edge_rate": float(np.mean(edge_label.numpy() > 0.5)),
        "solver_status": result.get("solver_status"),
        "solver_runtime": result.get("runtime"),
    }
    return data, meta


def train(
    instance_dir: str = str(TABLE5_DATA),
    result_dir: str = str(TABLE5_LABELS),
    model_dir: str = str(TABLE5_TRAIN_OUT / "models"),
    charts_dir: str = str(TABLE5_TRAIN_OUT / "charts"),
    log_dir: str = str(TABLE5_TRAIN_OUT / "logs"),
    result_suffix: str = "__mb.json",
    train_manifest: str = "",
    eval_manifest: str = "",
    test_manifest: str = "",
    max_files: int = 0,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_channels: int = 128,
    num_layers: int = 2,
    save_interval: int = 100,
    val_ratio: float = 0.2,
    early_stopping_patience: int = 30,
    seed: int = 42,
    cleanup_before_train: bool = True,
    device_name: str = "auto",
) -> Tuple[torch.nn.Module, np.ndarray, Optional[np.ndarray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = _resolve_device(device_name)

    _ensure_dirs(model_dir, charts_dir, log_dir)
    suffix = f"_{num_layers}layer_seed{seed}"
    if cleanup_before_train:
        _clean_previous_run_artifacts(model_dir, suffix=suffix)

    class _DummyWriter:
        def add_scalar(self, *args, **kwargs):
            pass
        def flush(self):
            pass
        def close(self):
            pass

    if SummaryWriter is None:
        writer = _DummyWriter()
    else:
        try:
            writer = SummaryWriter(log_dir=log_dir)
        except Exception:
            writer = _DummyWriter()

    explicit_split = bool(train_manifest or eval_manifest or test_manifest)
    if explicit_split:
        if not train_manifest or not eval_manifest:
            raise ValueError("When using explicit manifests, both train_manifest and eval_manifest are required.")
        split_pairs = {
            "train": _read_result_manifest(train_manifest),
            "eval": _read_result_manifest(eval_manifest),
            "test": _read_result_manifest(test_manifest) if test_manifest else [],
        }
    else:
        pairs = _match_instance_result_pairs(instance_dir, result_dir, result_suffix=result_suffix, max_files=max_files)
        if not pairs:
            raise FileNotFoundError(
                f"No matched instance/result pairs found under instance_dir={instance_dir}, result_dir={result_dir}, suffix={result_suffix}"
            )
        split_pairs = {"all": pairs}

    split_datasets: Dict[str, List[Data]] = {}
    split_metas: Dict[str, List[Dict]] = {}
    for split_name, pairs in split_pairs.items():
        split_datasets[split_name] = []
        split_metas[split_name] = []
        if not pairs:
            continue
        print(f"Start reading the CPBSD MB dataset for split={split_name}...")
        for idx, (instance_path, result_path) in enumerate(pairs, 1):
            try:
                dat, meta = read_cpbsd_mb_data(str(instance_path), str(result_path))
                split_datasets[split_name].append(dat)
                split_metas[split_name].append(meta)
            except Exception as exc:
                print(f"⚠️ Skip pair due to read failure: {instance_path.name} / {result_path.name}: {exc}")
            if idx % 50 == 0:
                print(f"Loaded data ({split_name}): {idx}/{len(pairs)}")

    if explicit_split:
        train_data = split_datasets["train"]
        val_data = split_datasets["eval"]
        test_data = split_datasets["test"]
        dataset = train_data + val_data + test_data
        metas = split_metas["train"] + split_metas["eval"] + split_metas["test"]
    else:
        dataset = split_datasets["all"]
        metas = split_metas["all"]
        if not dataset:
            raise RuntimeError("No CPBSD MB training samples were loaded successfully.")
        train_idx, val_idx = _split_indices(len(dataset), val_ratio, seed)
        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]
        test_data = []

    if not train_data or not val_data:
        raise RuntimeError("Training or validation split is empty after loading CPBSD MB labeled data.")

    print(f"📦 总计加载 {len(dataset)} 个样本")
    pos_rate = float(np.mean(np.concatenate([(d.edge_label.numpy() > 0.5).astype(float) for d in dataset])))
    print(f"📈 全局正边比例(x_kn=1): {pos_rate:.6f}")
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    if test_data:
        print(f"  测试集: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data else None
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) if test_data else None

    model = EdgeScoringGCN(
        in_channels=4,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        edge_dim=1,
        dropout=0.5,
    ).to(device)

    raw_pos_rate = float(np.mean(np.concatenate([(d.edge_label.numpy() > 0.5).astype(float) for d in train_data])))
    raw_pos_rate = min(max(raw_pos_rate, 1e-6), 1 - 1e-6)
    pos_weight = (1.0 - raw_pos_rate) / raw_pos_rate
    print(f"⚖️ 训练集正边比例(x_kn=1): {raw_pos_rate:.6f}, pos_weight={pos_weight:.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _compute_classification_stats(logits: torch.Tensor, label: torch.Tensor):
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        tp = float(((pred == 1) & (label == 1)).sum().item())
        tn = float(((pred == 0) & (label == 0)).sum().item())
        fp = float(((pred == 1) & (label == 0)).sum().item())
        fn = float(((pred == 0) & (label == 1)).sum().item())
        total = max(1.0, tp + tn + fp + fn)
        acc = (tp + tn) / total
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        return acc, precision, recall, f1, float(prob.mean().item()), float(label.mean().item())

    def _eval(loader: Optional[DataLoader]) -> Tuple[float, float, float, float, float, float]:
        if loader is None or len(loader) == 0:
            return float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf")
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_prob_mean = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)["edge_logits"].view(-1)
                label = batch.edge_label.to(logits.dtype).view(-1)
                loss = criterion(logits, label)
                acc, precision, recall, f1, prob_mean, _ = _compute_classification_stats(logits, label)
                total_loss += float(loss.item())
                total_acc += acc
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_prob_mean += prob_mean
                count += 1
        return (
            total_loss / count,
            total_acc / count,
            total_precision / count,
            total_recall / count,
            total_f1 / count,
            total_prob_mean / count,
        )

    best_val = float("inf")
    patience = 0
    best_model_path = os.path.join(model_dir, f"best_model_edge_cpbsd_mb_x{suffix}.pt")
    final_model_path = os.path.join(model_dir, f"model_edge_cpbsd_mb_x{suffix}.pt")
    metrics_path = os.path.join(model_dir, f"metrics_edge_cpbsd_mb_x{suffix}.json")

    train_hist: List[Tuple[int, float]] = []
    val_hist: List[Tuple[int, float]] = []
    metrics = []

    print(f"\n开始训练 {epochs} 轮...")
    for epoch in range(epochs):
        model.train()
        total_train = 0.0
        total_train_acc = 0.0
        total_train_f1 = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)["edge_logits"].view(-1)
            label = batch.edge_label.to(logits.dtype).view(-1)
            optimizer.zero_grad()
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            acc, _, _, f1, _, _ = _compute_classification_stats(logits.detach(), label)
            total_train += float(loss.item())
            total_train_acc += acc
            total_train_f1 += f1

        avg_train = total_train / max(1, len(train_loader))
        avg_train_acc = total_train_acc / max(1, len(train_loader))
        avg_train_f1 = total_train_f1 / max(1, len(train_loader))
        train_hist.append((epoch, avg_train))
        writer.add_scalar("Loss/train", avg_train, epoch)
        writer.add_scalar("Accuracy/train", avg_train_acc, epoch)
        writer.add_scalar("F1/train", avg_train_f1, epoch)

        avg_val, avg_val_acc, avg_val_precision, avg_val_recall, avg_val_f1, avg_val_prob_mean = _eval(val_loader)
        if np.isfinite(avg_val):
            val_hist.append((epoch, avg_val))
            writer.add_scalar("Loss/validation", avg_val, epoch)
            writer.add_scalar("Accuracy/validation", avg_val_acc, epoch)
            writer.add_scalar("Precision/validation", avg_val_precision, epoch)
            writer.add_scalar("Recall/validation", avg_val_recall, epoch)
            writer.add_scalar("F1/validation", avg_val_f1, epoch)

            if avg_val < best_val - 1e-12:
                best_val = avg_val
                patience = 0
                model_cpu = model.cpu()
                torch.save(model_cpu, best_model_path)
                model = model_cpu.to(device)
                print(f"  🎯 新最佳模型：val_loss={avg_val:.6f}, val_f1={avg_val_f1:.6f} (epoch={epoch})")
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print(f"  ⏹️ 触发早停（连续 {early_stopping_patience} 轮无改进）")
                    break

        metrics.append(
            {
                "epoch": epoch,
                "train_loss": avg_train,
                "train_accuracy": avg_train_acc,
                "train_f1": avg_train_f1,
                "val_loss": avg_val,
                "val_accuracy": avg_val_acc,
                "val_precision": avg_val_precision,
                "val_recall": avg_val_recall,
                "val_f1": avg_val_f1,
                "val_prob_mean": avg_val_prob_mean,
            }
        )

        if epoch % 10 == 0:
            if np.isfinite(avg_val):
                print(
                    f"Epoch {epoch:3d} | train_loss={avg_train:.6f} | train_acc={avg_train_acc:.4f} | train_f1={avg_train_f1:.4f} | "
                    f"val_loss={avg_val:.6f} | val_acc={avg_val_acc:.4f} | val_f1={avg_val_f1:.4f}"
                )
            else:
                print(f"Epoch {epoch:3d} | train_loss={avg_train:.6f} | train_acc={avg_train_acc:.4f} | train_f1={avg_train_f1:.4f}")

        if epoch % save_interval == 0 and epoch > 0:
            ckpt_path = os.path.join(model_dir, f"model_edge_cpbsd_mb_x{suffix}_epoch{epoch}.pt")
            model_cpu = model.cpu()
            torch.save(model_cpu, ckpt_path)
            model = model_cpu.to(device)

    if os.path.exists(best_model_path):
        best_model = torch.load(best_model_path, weights_only=False)
        torch.save(best_model, final_model_path)
        model = best_model.to(device)
    else:
        model_cpu = model.cpu()
        torch.save(model_cpu, final_model_path)
        model = model_cpu.to(device)

    train_hist_np = np.array(train_hist, dtype=float)
    np.savetxt(os.path.join(model_dir, f"train_loss_edge_cpbsd_mb_x{suffix}.csv"), train_hist_np, delimiter=",")
    if val_hist:
        val_hist_np = np.array(val_hist, dtype=float)
        np.savetxt(os.path.join(model_dir, f"val_loss_edge_cpbsd_mb_x{suffix}.csv"), val_hist_np, delimiter=",")
    else:
        val_hist_np = None

    train_eval = _eval(train_loader)
    val_eval = _eval(val_loader)
    test_eval = _eval(test_loader) if test_loader is not None else None

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "instance_dir": instance_dir,
                    "result_dir": result_dir,
                    "result_suffix": result_suffix,
                    "train_manifest": train_manifest,
                    "eval_manifest": eval_manifest,
                    "test_manifest": test_manifest,
                    "max_files": max_files,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "hidden_channels": hidden_channels,
                    "num_layers": num_layers,
                    "val_ratio": val_ratio,
                    "seed": seed,
                    "pos_weight": pos_weight,
                    "device": str(device),
                    "requested_device": device_name,
                },
                "dataset_summary": {
                    "num_samples": len(dataset),
                    "num_train": len(train_data),
                    "num_val": len(val_data),
                    "num_test": len(test_data),
                    "global_positive_edge_rate": pos_rate,
                    "global_edge_label_mean": float(np.mean([m["edge_label_mean"] for m in metas])),
                },
                "split_metrics": {
                    "train": {
                        "loss": train_eval[0],
                        "accuracy": train_eval[1],
                        "precision": train_eval[2],
                        "recall": train_eval[3],
                        "f1": train_eval[4],
                        "prob_mean": train_eval[5],
                    },
                    "eval": {
                        "loss": val_eval[0],
                        "accuracy": val_eval[1],
                        "precision": val_eval[2],
                        "recall": val_eval[3],
                        "f1": val_eval[4],
                        "prob_mean": val_eval[5],
                    },
                    "test": None
                    if test_eval is None
                    else {
                        "loss": test_eval[0],
                        "accuracy": test_eval[1],
                        "precision": test_eval[2],
                        "recall": test_eval[3],
                        "f1": test_eval[4],
                        "prob_mean": test_eval[5],
                    },
                },
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    chart_dir = os.path.join(charts_dir, f"run_{run_tag}_cpbsd_mb_x_{num_layers}layer_seed{seed}")
    _ensure_dirs(chart_dir)

    plt.figure(figsize=(10, 7))
    plt.plot(train_hist_np[:, 0], train_hist_np[:, 1], label="Training Loss", color="C0")
    if val_hist_np is not None:
        plt.plot(val_hist_np[:, 0], val_hist_np[:, 1], label="Validation Loss", color="C3")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogits Loss")
    plt.title(f"CPBSD MB x_kn Training Loss ({num_layers} layers, seed={seed})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(chart_dir, f"loss_curves_cpbsd_mb_x_{num_layers}layer_seed{seed}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    writer.flush()
    writer.close()

    print("\n✅ 训练完成！")
    print(f"  最终模型: {final_model_path}")
    print(f"  指标文件: {metrics_path}")
    print(f"  曲线图: {fig_path}")

    return model, train_hist_np, val_hist_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CPBSD edge-scoring GCN with MB-derived x_kn supervision")
    parser.add_argument("--instance_dir", type=str, default=str(TABLE5_DATA))
    parser.add_argument("--result_dir", type=str, default=str(TABLE5_LABELS))
    parser.add_argument("--model_dir", type=str, default=str(TABLE5_TRAIN_OUT / "models"))
    parser.add_argument("--charts_dir", type=str, default=str(TABLE5_TRAIN_OUT / "charts"))
    parser.add_argument("--log_dir", type=str, default=str(TABLE5_TRAIN_OUT / "logs"))
    parser.add_argument("--result_suffix", type=str, default="__mb.json")
    parser.add_argument("--train_manifest", type=str, default=str(TABLE5_LABELS / "manifest_train__portable.csv"))
    parser.add_argument("--eval_manifest", type=str, default=str(TABLE5_LABELS / "manifest_eval__portable.csv"))
    parser.add_argument("--test_manifest", type=str, default=str(TABLE5_LABELS / "manifest_test__portable.csv"))
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no_cleanup", action="store_true")
    args = parser.parse_args()

    train(
        instance_dir=args.instance_dir,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        charts_dir=args.charts_dir,
        log_dir=args.log_dir,
        result_suffix=args.result_suffix,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        test_manifest=args.test_manifest,
        max_files=args.max_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        save_interval=args.save_interval,
        val_ratio=args.val_ratio,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        cleanup_before_train=not args.no_cleanup,
        device_name=args.device,
    )
