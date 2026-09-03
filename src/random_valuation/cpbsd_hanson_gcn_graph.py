"""Hanson-style CPBSD graph construction for MB x_kn edge supervision.

This is the label-free counterpart to ``cpbsd_invariant_gcn_graph.py``. It
reproduces the original Hanson (paper) edge-scoring GCN graph used by
``Training_multi_layer_cpbsd_mb_x.py`` so the same features can be built for
inference (no labels) as well as training (with labels).

Features (kept byte-for-byte identical to
``Training_multi_layer_cpbsd_mb_x.read_cpbsd_mb_data``):
- Product node features: [c_n, mean_k(v_kn), 0, 0]
- Customer node features: [0, 0, K, rho_k],  rho_k = mean_n 1(v_kn - c_n > 0)
- Edge features: [v_kn]  (single utility/value, edge_dim=1)
- Edge supervision (optional): x_kn in {0,1}

Unlike the invariant builder this DOES encode the raw customer count K on every
customer node, matching Hanson's CPBSD model. See the project plan for the
raw-K trade-off (harmless under fixed K; only matters under K-transfer).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
from torch_geometric.data import Data


def read_result_manifest(path: str) -> List[Tuple[Path, Path]]:
    rows: List[Tuple[Path, Path]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("has_solution", "").lower() not in {"true", "1"}:
                continue
            instance_path = Path(row["instance_path"])
            result_path = Path(row["result_path"])
            if instance_path.exists() and result_path.exists():
                rows.append((instance_path, result_path))
    return rows


def load_instance(path: str | Path) -> Dict:
    with open(path, "rb") as f:
        return msgpack.load(f, object_hook=mnp.decode, strict_map_key=False)


def load_mb_result(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_hanson_cpbsd_graph(
    v_kn: np.ndarray,
    c_n: np.ndarray,
    x_kn: Optional[np.ndarray] = None,
    instance_path: str = "",
    result_path: str = "",
    setup: Optional[Dict] = None,
) -> Tuple[Data, Dict]:
    """Build the original Hanson-style product-customer graph.

    Product node features:  [cost, mean value, 0, 0]
    Customer node features: [0, 0, K (raw customer count), positive-margin rate]
    Edge features:          [value]  (edge_dim=1)

    Mirrors Training_multi_layer_cpbsd_mb_x.read_cpbsd_mb_data exactly so a model
    trained there can be deployed against this builder.
    """
    v_kn = np.asarray(v_kn, dtype=float)
    c_n = np.asarray(c_n, dtype=float)
    if v_kn.ndim != 2:
        raise ValueError(f"Expected v_kn to be 2D, got shape {v_kn.shape}")
    k_count, n_products = v_kn.shape
    if c_n.shape != (n_products,):
        raise ValueError(f"Expected c_n shape {(n_products,)}, got {c_n.shape}")

    if x_kn is not None:
        x_kn = np.asarray(x_kn, dtype=float)
        if x_kn.shape != (k_count, n_products):
            raise ValueError(f"Expected x_kn shape {(k_count, n_products)}, got {x_kn.shape}")

    feature_mat = np.zeros((n_products + k_count, 4), dtype=float)
    feature_mat[:n_products, 0] = c_n
    feature_mat[:n_products, 1] = np.mean(v_kn, axis=0)

    rho_k = np.mean((v_kn - c_n[None, :]) > 0, axis=1)
    feature_mat[n_products:, 2] = float(k_count)
    feature_mat[n_products:, 3] = rho_k

    left_nodes: List[int] = []
    right_nodes: List[int] = []
    edge_attr: List[List[float]] = []
    edge_label: List[float] = []
    for product_idx in range(n_products):
        for customer_idx in range(k_count):
            left_nodes.append(product_idx)
            right_nodes.append(customer_idx + n_products)
            edge_attr.append([v_kn[customer_idx, product_idx]])
            if x_kn is not None:
                edge_label.append(float(x_kn[customer_idx, product_idx]))

    data = Data(
        x=torch.tensor(feature_mat, dtype=torch.float),
        edge_index=torch.tensor([left_nodes, right_nodes], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.empty(0, dtype=torch.long),
    )
    if x_kn is not None:
        data.edge_label = torch.tensor(edge_label, dtype=torch.float)
    data.side_ind = torch.tensor(np.array([1] * n_products + [0] * k_count)[:, None], dtype=torch.long)
    data.product_num = n_products
    data.segment_num = k_count

    meta = {
        "instance_path": str(instance_path),
        "result_path": str(result_path),
        "setup": setup or {},
        "product_num": n_products,
        "segment_num": k_count,
    }
    if x_kn is not None:
        edge_label_arr = np.asarray(edge_label, dtype=float)
        meta["edge_label_mean"] = float(edge_label_arr.mean())
        meta["positive_edge_rate"] = float((edge_label_arr > 0.5).mean())
    return data, meta


def read_cpbsd_mb_data(instance_path: str, result_path: str) -> Tuple[Data, Dict]:
    instance = load_instance(instance_path)
    result = load_mb_result(result_path)
    chosen_product_matrix = result.get("chosen_product_matrix")
    if chosen_product_matrix is None:
        raise ValueError(f"Result file has no chosen_product_matrix: {result_path}")

    data, meta = build_hanson_cpbsd_graph(
        v_kn=np.asarray(instance["valuation_samples_V"], dtype=float),
        c_n=np.asarray(instance["production_cost_c"], dtype=float),
        x_kn=np.asarray(chosen_product_matrix, dtype=float),
        instance_path=str(instance_path),
        result_path=str(result_path),
        setup=instance.get("setup", {}),
    )
    meta["solver_status"] = result.get("solver_status")
    meta["solver_runtime"] = result.get("runtime")
    return data, meta
