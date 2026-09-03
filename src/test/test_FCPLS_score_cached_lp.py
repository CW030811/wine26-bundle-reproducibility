"""
Local Search Path Test - Global Top-K Strategy with Mixed Add/Drop Ordering

Cached LP-skeleton variant.

This script tests the impact of different search path strategies on Local Search efficiency.
Uses a global Top-K approach instead of segment-based (2*m) neighbor generation.

Strategy:
1. Use threshold method to generate initial bundle prediction
2. Use MILP solver to obtain initial optimal bundle assignment
3. Use LP solver for fast local search neighborhood evaluation
4. Use global Top-K neighbor generation: K = ceil(sqrt(m))
   - Generate K add neighbors with score = P[k,j] (probability)
   - Generate K drop neighbors with score = 1-P[k,j] (1-probability)
   - Mix and sort all 2K neighbors by score (descending)
5. Use greedy strategy: accept improvement when found
6. Convert final assignment back to MILP for global optimization

Implementation notes:
- Local search caches the LP skeleton by active bundle set F and only updates
  objective coefficients plus assignment-specific upper constraints.
- Compact Rs/cs matrices are cached by dataset arrays and bundle set.
- Exact subadditivity cover enumeration is cached by (n, bundle set).
""" 

import os
import gc
import numpy as np
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from math import ceil, sqrt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import msgpack
import msgpack_numpy as mnp
from torch_geometric.nn import GENConv
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import gurobipy as gp
from gurobipy import GRB
import argparse
from typing import List, Tuple
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ============================================================================
# Helper Functions
# ============================================================================

def parse_list_arg(arg: str) -> List[int]:
    """Parse comma-separated string to list of integers"""
    return [int(x) for x in arg.split(",") if x.strip()]


def parse_paths_arg(arg: str) -> List[str]:
    """Parse semicolon-separated paths string to list"""
    return [p.strip() for p in arg.split(";") if p.strip()]


def _append_objective_event(history: List[dict], elapsed: float, value: float, source: str) -> None:
    """Append one objective event, keeping times nonnegative and values finite."""
    if history is None:
        return
    if value is None or not np.isfinite(value):
        return
    event = {
        "time": max(0.0, float(elapsed)),
        "value": float(value),
        "source": str(source),
    }
    if history:
        prev = history[-1]
        if (
            abs(prev["time"] - event["time"]) < 1e-9
            and abs(prev["value"] - event["value"]) < 1e-12
            and prev["source"] == event["source"]
        ):
            return
    history.append(event)


def _extend_with_gurobi_objective_history(
    history: List[dict],
    gurobi_history: List[dict],
    time_offset: float,
    opt_rev: float,
    source_prefix: str,
) -> None:
    """Convert Gurobi absolute objective history to competitive-ratio events."""
    if history is None or opt_rev is None or abs(opt_rev) <= 1e-12:
        return
    for event in gurobi_history:
        objective = event.get("objective")
        runtime = event.get("runtime")
        if objective is None or runtime is None or not np.isfinite(objective):
            continue
        _append_objective_event(
            history,
            time_offset + float(runtime),
            float(objective) / float(opt_rev),
            f"{source_prefix}_{event.get('source', 'gurobi')}",
        )


def _history_to_grid_values(history: List[dict], time_grid: np.ndarray) -> np.ndarray:
    """Convert objective events to best-known values at each time limit."""
    if len(time_grid) == 0:
        return np.array([], dtype=float)
    if not history:
        return np.zeros(len(time_grid), dtype=float)

    ordered = sorted(history, key=lambda x: x["time"])
    event_times = np.array([event["time"] for event in ordered], dtype=float)
    event_values = np.array([event["value"] for event in ordered], dtype=float)
    values = np.zeros(len(time_grid), dtype=float)

    for t_idx, t in enumerate(time_grid):
        idx = int(np.searchsorted(event_times, t, side="right")) - 1
        values[t_idx] = event_values[idx] if idx >= 0 else 0.0
    return values


def _build_time_grid(histories: List[List[dict]], interval: float) -> np.ndarray:
    """Build a common time grid from 0 to the maximum observed runtime."""
    max_time = 0.0
    for history in histories:
        if history:
            max_time = max(max_time, max(event["time"] for event in history))
    if max_time <= 0:
        return np.array([0.0], dtype=float)
    num_steps = int(np.ceil(max_time / interval))
    return np.arange(num_steps + 1, dtype=float) * interval


def _aggregate_instance_plot_curves(instance_histories: dict, interval: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match the FCP/PCP reporting convention:
    for each time point, average the seed curves within each instance first,
    then compute mean/std across instance-level seed-averaged curves.
    """
    all_histories = []
    for seed_to_history in instance_histories.values():
        all_histories.extend(seed_to_history.values())
    if not all_histories:
        return np.array([]), np.array([]), np.array([])

    time_grid = _build_time_grid(all_histories, interval)
    instance_curves = []
    for instance_id in sorted(instance_histories.keys()):
        seed_to_history = instance_histories[instance_id]
        if not seed_to_history:
            continue
        seed_curves = [
            _history_to_grid_values(history, time_grid)
            for _, history in sorted(seed_to_history.items())
        ]
        if seed_curves:
            instance_curves.append(np.mean(np.vstack(seed_curves), axis=0))

    if not instance_curves:
        return time_grid, np.array([]), np.array([])

    instance_curve_matrix = np.vstack(instance_curves)
    mean_curve = np.mean(instance_curve_matrix, axis=0)
    std_curve = np.std(instance_curve_matrix, axis=0)
    return time_grid, mean_curve, std_curve


def _save_objective_curve_plot(
    time_grid: np.ndarray,
    mean_curve: np.ndarray,
    std_curve: np.ndarray,
    out_png: str,
    title: str,
    x_label: str = "Time Limit",
) -> None:
    """Save one mean competitive-ratio curve with std shading."""
    fig, ax = plt.subplots(figsize=(8, 6))
    lower = np.clip(mean_curve - std_curve, 0.0, None)
    upper = mean_curve + std_curve

    ax.fill_between(time_grid, lower, upper, color="#4aa8ff", alpha=0.25, linewidth=0)
    ax.plot(
        time_grid,
        mean_curve,
        color="#1296db",
        linewidth=2.5,
        marker="x",
        markersize=8,
        markeredgewidth=2,
        markevery=max(1, len(time_grid) // 12),
    )

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel("Competitive Ratio", fontsize=16)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.25)
    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_objective_curve_csv(
    time_grid: np.ndarray,
    mean_curve: np.ndarray,
    std_curve: np.ndarray,
    out_csv: str,
) -> None:
    df = pd.DataFrame(
        {
            "time_sec": time_grid,
            "mean_competitive_ratio": mean_curve,
            "std_competitive_ratio": std_curve,
        }
    )
    df.to_csv(out_csv, index=False)


def _save_plot_for_key(
    nl: int,
    m: int,
    n: int,
    instance_histories: dict,
    result_dir: str,
    test_folder_name: str,
    time_kind: str,
    time_label: str,
    silent: bool = False,
    interval: float = 5.0,
) -> None:
    time_grid, mean_curve, std_curve = _aggregate_instance_plot_curves(instance_histories, interval=interval)
    if len(time_grid) == 0 or len(mean_curve) == 0:
        return

    base_name = f"test_obj_curve_FCPLS_{time_kind}_{nl}layer_{test_folder_name}_m{m}n{n}"
    png_path = os.path.join(result_dir, base_name + ".png")
    csv_path = os.path.join(result_dir, base_name + ".csv")
    title = f"FCPLS {nl}-Layer | {test_folder_name} | m={m}, n={n} | {time_label}"

    _save_objective_curve_plot(time_grid, mean_curve, std_curve, png_path, title, x_label=time_label)
    _save_objective_curve_csv(time_grid, mean_curve, std_curve, csv_path)

    if not silent:
        print(f"📈 Saved objective curve plot: {png_path}")
        print(f"📄 Saved objective curve data: {csv_path}")


def load_models(
    model_root: str,
    layers: List[int],
    seeds: List[int],
    device: torch.device,
    silent: bool = False,
) -> dict:
    """Load models for specified layers and seeds. Returns {layer: [(seed, model, path), ...]}"""
    loaded = {nl: [] for nl in layers}
    for nl in layers:
        for sd in seeds:
            cand_paths = [
                os.path.join(model_root, f"best_model_edge_{nl}layer_seed{sd}.pt"),
                os.path.join(model_root, f"model_edge_{nl}layer_seed{sd}.pt"),
                os.path.join(model_root, f"model-{nl}layer_seed{sd}.pt"),
            ]
            path = next((p for p in cand_paths if os.path.exists(p)), None)
            if path is None:
                if not silent:
                    print(f"⚠️ Model not found: layer={nl}, seed={sd}, searched={cand_paths}")
                continue
            try:
                mdl = torch.load(path, map_location=device)
                mdl.to(device)
                mdl.eval()
                loaded[nl].append((sd, mdl, path))
                if not silent:
                    print(f"✅ Loaded model: layer={nl}, seed={sd}, path={path}")
            except Exception as e:
                if not silent:
                    print(f"❌ Failed to load layer={nl}, seed={sd}, path={path}: {e}")
    return loaded


# ============================================================================
# Model Definition
# ============================================================================

class EdgeScoringGCN(nn.Module):
    """
    Undirected message passing with layer-wise edge updates.
    Two-layer, hidden=128 by default; outputs edge_logits for BCEWithLogitsLoss.
    """

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

        for i in range(self.num_layers):
            undirected_edge_index, undirected_edge_attr = to_undirected(
                edge_index, edge_attr=current_edge_attr, num_nodes=x.size(0)
            )

            h = self.act(self.convs[i](h, undirected_edge_index, undirected_edge_attr))
            h = self.dropout(h)

            cat_input = torch.cat([h[src], h[dst], current_edge_attr], dim=-1)
            current_edge_attr = self.edge_updates[i](cat_input)

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

# Allow torch.load to resolve EdgeScoringGCN safely (PyTorch 2.6+)
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([EdgeScoringGCN])


# ============================================================================
# Data Processing Functions
# ============================================================================

def process_data(file_path):
    """
    Process data file and return graph data and related parameters
    Supports both old and new data formats, dynamically calculates cs and Rs matrices
    """
    with open(file_path, 'rb') as f:
        data = msgpack.load(f, object_hook=mnp.decode, strict_map_key=False)
    
    product_num = int(data['product_num'])
    segment_num = int(data['segment_num'])
    unit_cs = data['unit_cs']
    ship_cs = data['ship_cs']
    unit_us = data['unit_us']
    Ns = data['Ns']
    opt_bundles = data['opt_bundles']
    opt_prices = data['opt_prices']
    opt_rev = data['opt_rev']
    running_time = data['running_time']
    gap = data['gap']
    
    # Check if cs and Rs are stored, if not we'll calculate them on-demand
    has_stored_cs = 'cs' in data
    has_stored_Rs = 'Rs' in data
    
    if has_stored_cs and has_stored_Rs:
        # Use stored matrices (old format)
        cs = data['cs']
        Rs = data['Rs']
    else:
        # New format - will be calculated on-demand
        cs = None
        Rs = None
    
    node_num = product_num + segment_num
    
    # Build node features robustly against shape mismatches
    feature = np.zeros((node_num, 4), dtype=float)

    # unit_cs: expect shape (1, n) or (n,)
    if isinstance(unit_cs, np.ndarray):
        if unit_cs.ndim == 2:
            uc = unit_cs[0]
        else:
            uc = unit_cs
        uc = np.asarray(uc).reshape(-1)[:product_num]
    else:
        uc = np.asarray(unit_cs).reshape(-1)[:product_num]
    feature[:product_num, 0] = uc

    # unit_us: shape (m, n), average across segments for product feature
    uu_avg = np.average(unit_us, axis=0)
    feature[:product_num, 1] = np.asarray(uu_avg).reshape(-1)[:product_num]

    # Ns: shape (m, 1) or (m,)
    if isinstance(Ns, np.ndarray) and Ns.ndim == 2:
        ns_vec = Ns[:, 0]
    else:
        ns_vec = np.asarray(Ns).reshape(-1)
    ns_vec = ns_vec[:segment_num]
    feature[product_num:, 2] = ns_vec

    # ship_cs: may have more rows than m; take the first m strictly
    if isinstance(ship_cs, np.ndarray) and ship_cs.ndim == 2:
        sc_vec = ship_cs[:segment_num, 0]
    else:
        sc_vec = np.asarray(ship_cs).reshape(-1)[:segment_num]
    feature[product_num:, 3] = sc_vec

    x = torch.tensor(feature, dtype=torch.float)
    
    prods = []
    custs = []
    edge_weights = []
    for i in range(product_num):
        for j in range(segment_num):
            prods.append(i)
            custs.append(j+product_num)
            edge_weights.append([float(unit_us[j, i])])
            
    edge_index = torch.tensor([prods, custs], dtype=torch.long)        
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    side_ind = torch.tensor([1]*product_num + [0]*segment_num, dtype=torch.long).view(-1, 1)

    prod_labels = np.array(opt_bundles).T  
    seg_labels = -np.ones((segment_num, segment_num), dtype=int)
    y = np.append(prod_labels, seg_labels, axis=0)
    y = torch.tensor(y, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, side_ind=side_ind, y=y)
    miscellaneous = (product_num, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, cs, Rs)
    return data, miscellaneous


def convert_pred_assort_to_assignment(pred_assort):
    """
    Convert pred_assort matrix to segment_bundle_assignment dictionary
    """
    assignment = {}
    m, n = pred_assort.shape
    
    for k in range(m):
        bundle_idx = binary_vector_to_bundle_id(pred_assort[k, :])
        assignment[k] = bundle_idx
    
    return assignment


def binary_vector_to_bundle_id(bundle_binary):
    """Convert a binary assortment vector to its integer bitmask."""
    bundle_idx = 0
    for bit in np.asarray(bundle_binary).reshape(-1):
        bundle_idx = (bundle_idx << 1) | int(bit)
    return bundle_idx


def bundle_id_to_binary_vector(bundle_id, n):
    """Convert an integer bitmask to a length-n binary assortment vector."""
    bits = np.zeros(n, dtype=int)
    current = int(bundle_id)
    bit_pos = 0
    while current and bit_pos < n:
        bits[n - 1 - bit_pos] = current & 1
        current >>= 1
        bit_pos += 1
    return bits


def bundle_ids_to_assortment_matrix(bundle_ids, n):
    """Convert bundle ids to an assortment matrix without string formatting."""
    predicted_assortments = np.zeros((len(bundle_ids), n), dtype=int)
    for row_idx, bundle_id in enumerate(bundle_ids):
        current = int(bundle_id)
        bit_pos = 0
        while current and bit_pos < n:
            predicted_assortments[row_idx, n - 1 - bit_pos] = current & 1
            current >>= 1
            bit_pos += 1
    return predicted_assortments


def assignment_to_pred_assort(assignment, n, m):
    """
    Convert one-to-one assignment back to pred_assort matrix
    """
    pred_assort = np.zeros((m, n), dtype=int)

    for k in range(m):
        bundle_idx = assignment[k]
        pred_assort[k, :] = bundle_id_to_binary_vector(bundle_idx, n)

    return pred_assort


_RS_CS_MATRIX_CACHE = {}
_SUBADDITIVITY_COVER_CACHE = {}


def clear_internal_caches():
    """Clear process-local construction caches to avoid long-run memory growth."""
    _RS_CS_MATRIX_CACHE.clear()
    _SUBADDITIVITY_COVER_CACHE.clear()


def get_predicted_Rs_cs(n, m, unit_cs, ship_cs, unit_us, bundle_ids, stored_cs=None, stored_Rs=None):
    """
    Return compact Rs/cs matrices for a bundle set, with process-local caching.

    The cache key uses object identity for the current dataset arrays plus the
    ordered bundle tuple. Arrays are treated as read-only by the solvers below.
    """
    bundle_ids = tuple(int(b) for b in bundle_ids)
    cache_key = (
        id(unit_cs),
        id(ship_cs),
        id(unit_us),
        id(stored_cs),
        id(stored_Rs),
        int(n),
        int(m),
        bundle_ids,
    )
    cached = _RS_CS_MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bundle_to_idx = {bundle_id: idx for idx, bundle_id in enumerate(bundle_ids)}

    if stored_cs is not None and stored_Rs is not None:
        Rs_predicted = np.zeros((m, len(bundle_ids)))
        cs_predicted = np.zeros((m, len(bundle_ids)))
        for bundle_id, idx in bundle_to_idx.items():
            Rs_predicted[:, idx] = stored_Rs[:, bundle_id]
            cs_predicted[:, idx] = stored_cs[:, bundle_id]
    else:
        predicted_assortments = bundle_ids_to_assortment_matrix(bundle_ids, n)
        Rs_predicted = np.sqrt(unit_us.dot(predicted_assortments.T))
        cs_base = np.sum(predicted_assortments * unit_cs, axis=1)
        cs_predicted = (cs_base + ship_cs) * 0.2

    Rbar = float(np.max(Rs_predicted)) if Rs_predicted.size else 0.0
    result = (Rs_predicted, cs_predicted, bundle_to_idx, Rbar)
    _RS_CS_MATRIX_CACHE[cache_key] = result
    return result


def get_exact_price_subadditivity_covers(bundle_ids, n):
    """
    Enumerate exact free-disposal price-subadditivity covers once per bundle set.

    Returns a tuple of (target_bundle, ordered_cover_tuple) pairs.
    """
    bundle_key = tuple(sorted(int(b) for b in bundle_ids))
    cache_key = (int(n), bundle_key)
    cached = _SUBADDITIVITY_COVER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    predicted_nonempty = [int(b) for b in bundle_key if b != 0]
    if not predicted_nonempty:
        _SUBADDITIVITY_COVER_CACHE[cache_key] = ()
        return ()

    covers = []
    added_constraints = set()
    item_bits = [1 << bit_idx for bit_idx in range(n)]

    def add_cover(target_bundle, cover):
        ordered_cover = tuple(sorted(int(b) for b in cover))
        key = (int(target_bundle), ordered_cover)
        if key in added_constraints:
            return
        added_constraints.add(key)
        covers.append(key)

    for k in predicted_nonempty:
        target_mask = int(k)
        singleton_covers = []
        multi_cover_candidates = []

        for b in predicted_nonempty:
            if b == k:
                continue
            contribution = int(b) & target_mask
            if contribution == 0:
                continue
            if contribution == target_mask:
                singleton_covers.append(int(b))
            else:
                multi_cover_candidates.append((int(b), contribution))

        for b in singleton_covers:
            add_cover(target_mask, (b,))

        if not multi_cover_candidates:
            continue

        item_to_candidate_indices = {bit: [] for bit in item_bits if target_mask & bit}
        for idx, (_, contribution) in enumerate(multi_cover_candidates):
            remaining = contribution
            while remaining:
                lowbit = remaining & -remaining
                item_to_candidate_indices[lowbit].append(idx)
                remaining ^= lowbit

        if any(len(item_to_candidate_indices[bit]) == 0 for bit in item_to_candidate_indices):
            continue

        chosen = []
        item_cover_count = {bit: 0 for bit in item_to_candidate_indices}

        def dfs(available_indices, covered_mask):
            if covered_mask == target_mask:
                for bundle_id, contribution in chosen:
                    if all(
                        item_cover_count[bit] > 1
                        for bit in item_cover_count
                        if contribution & bit
                    ):
                        return
                add_cover(target_mask, tuple(bundle_id for bundle_id, _ in chosen))
                return

            remaining_union = covered_mask
            for idx in available_indices:
                remaining_union |= multi_cover_candidates[idx][1]
            if remaining_union != target_mask:
                return

            uncovered_mask = target_mask & ~covered_mask
            branch_options = None
            remaining = uncovered_mask
            while remaining:
                lowbit = remaining & -remaining
                options = [idx for idx in item_to_candidate_indices[lowbit] if idx in available_indices]
                if not options:
                    return
                if branch_options is None or len(options) < len(branch_options):
                    branch_options = options
                    if len(branch_options) == 1:
                        break
                remaining ^= lowbit

            for option_pos, idx in enumerate(branch_options):
                bundle_id, contribution = multi_cover_candidates[idx]
                chosen.append((bundle_id, contribution))
                updated_items = []
                remaining_contribution = contribution
                while remaining_contribution:
                    lowbit = remaining_contribution & -remaining_contribution
                    item_cover_count[lowbit] += 1
                    updated_items.append(lowbit)
                    remaining_contribution ^= lowbit
                blocked = set(branch_options[:option_pos + 1])
                next_available = tuple(j for j in available_indices if j not in blocked)
                dfs(next_available, covered_mask | contribution)
                for bit in updated_items:
                    item_cover_count[bit] -= 1
                chosen.pop()

        dfs(tuple(range(len(multi_cover_candidates))), 0)

    result = tuple(covers)
    _SUBADDITIVITY_COVER_CACHE[cache_key] = result
    return result


def add_exact_price_subadditivity_constraints(model, p, bundle_ids, n, name_prefix="sa"):
    """
    Add the exact free-disposal price-subadditivity constraints used in test_FCP.

    The construction is exact and works in two stages:
    1. Add singleton covers p[k] <= p[b] whenever k ⊆ b and b != k.
    2. Enumerate only minimal multi-bundle covers via bitmask DFS.
    """
    covers = get_exact_price_subadditivity_covers(bundle_ids, n)
    for subadd_ctr, (target_bundle, ordered_cover) in enumerate(covers):
        model.addConstr(
            p[int(target_bundle)] <= gp.quicksum(p[int(b)] for b in ordered_cover),
            name=f"{name_prefix}{subadd_ctr}",
        )
    return len(covers)


# ============================================================================
# Optimization Functions
# ============================================================================

def competitive_ratio_with_optimal_bundle(
    n,
    m,
    unit_cs,
    ship_cs,
    unit_us,
    Ns,
    opt_rev,
    pred_assort,
    stored_cs=None,
    stored_Rs=None,
    obj_history=None,
    history_time_offset=0.0,
    history_source_prefix="milp",
    obj_record_interval=5.0,
):
    milp_start_time = time.time()
    
    segment_ind = np.array([i for i in range(m)])
    
    # Get unique predicted bundles
    bundle_dic = {}
    for i in range(m):
        bundle_idx = binary_vector_to_bundle_id(pred_assort[i, :])
        try:
            bundle_dic[bundle_idx].append(i)
        except:
            bundle_dic[bundle_idx] = [i]
    
    predicted_bundles = list(bundle_dic.keys())
    if 0 not in predicted_bundles:
        predicted_bundles.append(0)
    
    # Calculate compact Rs/cs matrices only for predicted bundles.
    Rs_predicted, cs_predicted, bundle_to_idx, Rbar = get_predicted_Rs_cs(
        n,
        m,
        unit_cs,
        ship_cs,
        unit_us,
        predicted_bundles,
        stored_cs,
        stored_Rs,
    )

    model = gp.Model("Bundle MILP")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-3
    model.Params.TimeLimit = 600

    # Create variables ONLY for predicted bundles
    p = model.addVars(predicted_bundles, vtype=GRB.CONTINUOUS, lb=0, name="p")
    theta = model.addVars(m, predicted_bundles, vtype=GRB.BINARY, name="theta")
    s = model.addVars(m, vtype=GRB.CONTINUOUS, name="s")
    S = model.addVars(m, predicted_bundles, vtype=GRB.CONTINUOUS, lb=0, name='S')
    Z = model.addVars(m, predicted_bundles, vtype=GRB.CONTINUOUS, name='Z')
    P = model.addVars(m, predicted_bundles, vtype=GRB.CONTINUOUS, lb=0, name='P')

    # Objective: only consider predicted bundles
    model.setObjective(gp.quicksum(Ns[k, 0]*Z[k, i] for k in segment_ind for i in predicted_bundles), GRB.MAXIMIZE)

    # Standard constraints (only for predicted bundles)
    model.addConstrs((s[k] >= Rs_predicted[k, bundle_to_idx[i]] - p[i] for i in predicted_bundles for k in segment_ind))
    
    # Exact price-subadditivity constraints (same logic as the corrected FCP).
    add_exact_price_subadditivity_constraints(
        model, p, predicted_bundles, n, name_prefix="milp_sa"
    )

    # Remaining constraints (only for predicted bundles)
    model.addConstrs((P[k, i] >= p[i] - Rbar*(1-theta[k, i]) for i in predicted_bundles for k in segment_ind))
    model.addConstrs((P[k, i] <= p[i] for i in predicted_bundles for k in segment_ind))
    
    # Modified constraint: only sum over predicted bundles
    model.addConstrs((s[k] >= gp.quicksum(Rs_predicted[k, bundle_to_idx[i]]*theta[j, i] - P[j, i] for i in predicted_bundles) for k in segment_ind for j in segment_ind))
    
    model.addConstrs((Z[k, i] == P[k, i] - cs_predicted[k, bundle_to_idx[i]] * theta[k, i] for i in predicted_bundles for k in segment_ind))
    model.addConstrs((S[k, i] == Rs_predicted[k, bundle_to_idx[i]]*theta[k, i] - P[k, i] for i in predicted_bundles for k in segment_ind))
    model.addConstrs((s[k] == gp.quicksum(S[k, i] for i in predicted_bundles) for k in segment_ind))
    model.addConstrs((gp.quicksum(theta[k, i] for i in predicted_bundles) == 1 for k in segment_ind))

    if 0 in predicted_bundles:
        model.addConstr(p[0] == 0, name="Empty_bundle_price")
        model.addConstrs((S[k, 0] == 0 for k in segment_ind))

    gurobi_obj_history = []
    last_recorded_runtime = {"value": -float(obj_record_interval)}

    def _record_gurobi_objective(runtime, objective, source, force=False):
        if objective is None or not np.isfinite(objective):
            return
        if not force and runtime - last_recorded_runtime["value"] < obj_record_interval:
            return
        gurobi_obj_history.append(
            {
                "runtime": float(runtime),
                "objective": float(objective),
                "source": source,
            }
        )
        last_recorded_runtime["value"] = float(runtime)

    def _milp_callback(cb_model, where):
        if where == GRB.Callback.MIP:
            runtime = cb_model.cbGet(GRB.Callback.RUNTIME)
            incumbent_obj = cb_model.cbGet(GRB.Callback.MIP_OBJBST)
            if -GRB.INFINITY < incumbent_obj < GRB.INFINITY:
                _record_gurobi_objective(runtime, incumbent_obj, "mip_best")
        elif where == GRB.Callback.MIPSOL:
            runtime = cb_model.cbGet(GRB.Callback.RUNTIME)
            incumbent_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            _record_gurobi_objective(runtime, incumbent_obj, "mipsol", force=True)

    try:
        if obj_history is not None:
            model.optimize(_milp_callback)
        else:
            model.optimize()

        milp_time = float(model.Runtime)

        optimal_bundle_assignment = {}
        if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
            obj_val = float(model.ObjVal)
            _record_gurobi_objective(model.Runtime, obj_val, "final", force=True)
            _extend_with_gurobi_objective_history(
                obj_history,
                gurobi_obj_history,
                history_time_offset,
                opt_rev,
                history_source_prefix,
            )
            for k in segment_ind:
                for i in predicted_bundles:
                    if theta[k, i].X > 0.5:  # Binary variable is 1
                        optimal_bundle_assignment[k] = i
                        break
            return obj_val / opt_rev, milp_time, optimal_bundle_assignment

        if model.Status == GRB.INFEASIBLE:
            raise ValueError(f"Initial/Final FCP MILP is infeasible (status={model.Status})")

        raise ValueError(f"Initial/Final FCP MILP failed with status {model.Status}")
    finally:
        model.dispose()


def competitive_ratio_LP(
    n,
    m,
    unit_cs,
    ship_cs,
    unit_us,
    Ns,
    opt_rev,
    segment_bundle_assignment,
    stored_cs=None,
    stored_Rs=None,
    obj_history=None,
    history_time_offset=0.0,
    history_source="lp",
):
    lp_start_time = time.time()

    # Extract involved bundle set from assignment
    assigned_bundles = set(segment_bundle_assignment.values())
    predicted_bundles = list(assigned_bundles)

    # Ensure empty bundle (index 0) is in the set
    if 0 not in predicted_bundles:
        predicted_bundles.append(0)

    segment_ind = np.array([i for i in range(m)])
    Rs_predicted, cs_predicted, bundle_to_idx, _ = get_predicted_Rs_cs(
        n,
        m,
        unit_cs,
        ship_cs,
        unit_us,
        predicted_bundles,
        stored_cs,
        stored_Rs,
    )

    # Create LP model
    model = gp.Model("Bundle LP-IC")

    # DECISION VARIABLES: p_i (i ∈ F), s_k (k = 1,...,M)
    # Price variables: only create for bundles involved in the assignment
    p = model.addVars(predicted_bundles, vtype=GRB.CONTINUOUS, lb=0, name="p")

    # Consumer surplus variables: one for each customer segment
    s = model.addVars(m, vtype=GRB.CONTINUOUS, name="s")

    # OBJECTIVE FUNCTION (same as original MILP when θ is fixed)
    # max Σ_{k=1}^M N_k (p_{b_k} - c_{b_k})
    # Since θ is fixed, profit = Σ_k N_k * (p_{b_k} - c_{k,b_k})
    objective_expr = gp.LinExpr()
    for k in segment_ind:
        b_k = segment_bundle_assignment[k]  # bundle chosen by segment k
        if b_k in predicted_bundles:
            # Profit = price - cost for the assigned bundle
            objective_expr += Ns[k, 0] * (p[b_k] - cs_predicted[k, bundle_to_idx[b_k]])

    model.setObjective(objective_expr, GRB.MAXIMIZE)

    # CONSTRAINT 1: IC (Incentive Compatibility) - Individual rationality lower bounds
    # Mathematical formulation: s_k ≥ R_{ki} - p_i, ∀k, ∀i ∈ F
    # Ensures each segment's chosen bundle is optimal relative to all available bundles
    for k in segment_ind:
        for i in predicted_bundles:
            model.addConstr(s[k] >= Rs_predicted[k, bundle_to_idx[i]] - p[i],
                          name=f"IC_k{k}_i{i}")

    # CONSTRAINT 2: Upper bound constraint - Bind "assigned bundle" with surplus upper bound
    # Mathematical formulation: s_k ≤ R_{k,b_k} - p_{b_k}, ∀k
    # Derived from Hanson's "tightening + single price schedule" when θ is fixed
    # Together with Constraint 1, this ensures s_k = max_i∈F{R_{ki} - p_i} = R_{k,b_k} - p_{b_k}
    # This "locks in" the θ choice and eliminates binary variables
    for k in segment_ind:
        b_k = segment_bundle_assignment[k]  # bundle assigned to segment k
        if b_k in predicted_bundles:
            model.addConstr(s[k] <= Rs_predicted[k, bundle_to_idx[b_k]] - p[b_k],
                          name=f"Upper_bound_k{k}")

    # CONSTRAINT 3: exact price subadditivity, matching the corrected FCP model.
    add_exact_price_subadditivity_constraints(
        model, p, predicted_bundles, n, name_prefix="lp_sa"
    )

    # CONSTRAINT 4: Non-negativity and normalization
    # Mathematical formulation: p_i ≥ 0 (i ∈ F), p_0 = 0
    # Empty bundle price is zero (already enforced by variable bounds, but explicit for clarity)
    if 0 in predicted_bundles:
        model.addConstr(p[0] == 0, name="Empty_bundle_price")

    # Solver parameter settings optimized for small-scale LP problems
    model.setParam("OutputFlag", 0)  # Disable output
    # Use automatic method selection (often defaults to Simplex for small problems)
    # Barrier (Method=2) is optimized for large sparse problems, but Simplex is typically faster for small dense problems
    model.setParam("Method", -1)     # Auto method selection
    model.setParam("Presolve", 2)    # Aggressive presolving (default is 2, but explicit for clarity)
    model.setParam("Threads", 1)     # Use single thread for small problems (avoids overhead)
    model.Params.TimeLimit = 300     # Time limit

    try:
        # Solve
        model.optimize()

        lp_time = float(model.Runtime)

        # Return results
        if model.Status == GRB.OPTIMAL:
            objective_value = float(model.ObjVal)
            competitive_ratio = objective_value / opt_rev
            _append_objective_event(obj_history, history_time_offset + lp_time, competitive_ratio, history_source)
            return competitive_ratio, objective_value, lp_time
        elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
            objective_value = float(model.ObjVal)
            competitive_ratio = objective_value / opt_rev
            _append_objective_event(obj_history, history_time_offset + lp_time, competitive_ratio, history_source)
            return competitive_ratio, objective_value, lp_time
        else:
            return -np.inf, -np.inf, lp_time
    finally:
        model.dispose()


@dataclass
class LPState:
    """Reusable LP model for one active bundle set."""
    model: gp.Model
    p: dict
    s: dict
    upper_constrs: dict
    upper_bundle: dict
    F: tuple
    bundle_to_idx: dict
    Rs: np.ndarray
    cs: np.ndarray


class ReusableBundleLPCache:
    """
    Cache LP skeletons by active bundle set F.

    Fixed per F:
    - p/s variables
    - IC constraints
    - exact price-subadditivity constraints
    - p[0] = 0

    Dynamic per assignment:
    - objective coefficients and constant
    - m upper-bound constraints s_k + p[b_k] <= R[k,b_k]
    """

    def __init__(
        self,
        n,
        m,
        unit_cs,
        ship_cs,
        unit_us,
        Ns,
        opt_rev,
        stored_cs=None,
        stored_Rs=None,
        max_models=64,
    ):
        self.n = n
        self.m = m
        self.unit_cs = unit_cs
        self.ship_cs = ship_cs
        self.unit_us = unit_us
        self.Ns = Ns
        self.opt_rev = opt_rev
        self.stored_cs = stored_cs
        self.stored_Rs = stored_Rs
        self.max_models = int(max_models)
        self.cache = OrderedDict()
        self.models_built = 0
        self.cache_hits = 0

    def _build_model_for_F(self, F):
        F = tuple(sorted(int(b) for b in F))
        Rs, cs, bundle_to_idx, _ = get_predicted_Rs_cs(
            self.n,
            self.m,
            self.unit_cs,
            self.ship_cs,
            self.unit_us,
            F,
            self.stored_cs,
            self.stored_Rs,
        )

        model = gp.Model("Reusable Bundle LP-IC")
        model.Params.OutputFlag = 0
        model.Params.Method = 1
        model.Params.Presolve = 1
        model.Params.Threads = 1
        model.Params.TimeLimit = 300
        model.ModelSense = GRB.MAXIMIZE

        p = {b: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"p[{b}]") for b in F}
        s = {k: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s[{k}]") for k in range(self.m)}

        for k in range(self.m):
            for b in F:
                idx = bundle_to_idx[b]
                model.addLConstr(
                    s[k] + p[b],
                    GRB.GREATER_EQUAL,
                    float(Rs[k, idx]),
                    name=f"IC[{k},{b}]",
                )

        upper_constrs = {}
        upper_bundle = {}
        for k in range(self.m):
            upper_constrs[k] = model.addLConstr(
                s[k],
                GRB.LESS_EQUAL,
                0.0,
                name=f"Upper[{k}]",
            )
            upper_bundle[k] = None

        add_exact_price_subadditivity_constraints(model, p, F, self.n, name_prefix="lp_sa")

        if 0 in F:
            model.addConstr(p[0] == 0, name="Empty_bundle_price")

        model.update()
        self.models_built += 1
        return LPState(
            model=model,
            p=p,
            s=s,
            upper_constrs=upper_constrs,
            upper_bundle=upper_bundle,
            F=F,
            bundle_to_idx=bundle_to_idx,
            Rs=Rs,
            cs=cs,
        )

    def _get_state(self, assignment):
        F = tuple(sorted(set(int(b) for b in assignment.values()) | {0}))
        if F in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(F)
            return self.cache[F]

        state = self._build_model_for_F(F)
        self.cache[F] = state

        while len(self.cache) > self.max_models:
            _, old_state = self.cache.popitem(last=False)
            old_state.model.dispose()

        return state

    def solve(self, assignment, obj_history=None, history_time_offset=0.0, history_source="lp"):
        state = self._get_state(assignment)
        model = state.model

        price_coef = defaultdict(float)
        objective_constant = 0.0
        for k in range(self.m):
            b_k = int(assignment[k])
            idx = state.bundle_to_idx[b_k]
            Nk = float(self.Ns[k, 0])
            price_coef[b_k] += Nk
            objective_constant -= Nk * float(state.cs[k, idx])

        for b in state.F:
            state.p[b].Obj = float(price_coef.get(b, 0.0))
        model.ObjCon = float(objective_constant)

        for k in range(self.m):
            new_b = int(assignment[k])
            old_b = state.upper_bundle[k]
            constr = state.upper_constrs[k]

            if old_b is not None and old_b != new_b:
                model.chgCoeff(constr, state.p[old_b], 0.0)
            if old_b != new_b:
                model.chgCoeff(constr, state.p[new_b], 1.0)
                state.upper_bundle[k] = new_b

            idx = state.bundle_to_idx[new_b]
            constr.RHS = float(state.Rs[k, idx])

        model.optimize()
        lp_time = float(model.Runtime)

        if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
            competitive_ratio = float(model.ObjVal) / self.opt_rev
            objective_value = float(model.ObjVal)
            _append_objective_event(
                obj_history,
                history_time_offset + lp_time,
                competitive_ratio,
                history_source,
            )
            return True, competitive_ratio, objective_value, lp_time

        return False, -np.inf, -np.inf, lp_time

    def stats(self):
        return {
            "lp_cache_models": self.models_built,
            "lp_cache_hits": self.cache_hits,
            "lp_cache_size": len(self.cache),
        }

    def dispose(self):
        for state in self.cache.values():
            state.model.dispose()
        self.cache.clear()

    def __del__(self):
        try:
            self.dispose()
        except Exception:
            pass


def check_lp_feasibility_and_competitive_ratio(assignment, n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, stored_cs=None, stored_Rs=None):
    """
    Quickly check if given assignment is feasible under LP and return competitive ratio

    Returns:
        tuple: (is_feasible, competitive_ratio, objective_value, solve_time)
    """
    try:
        competitive_ratio, objective_value, solve_time = competitive_ratio_LP(n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, assignment, stored_cs, stored_Rs)
        
        if competitive_ratio == -np.inf:
            return False, -np.inf, -np.inf, solve_time
        else:
            return True, competitive_ratio, objective_value, solve_time

    except Exception as e:
        return False, -np.inf, -np.inf, 0.0


def predict_initial_bundles(dat, miscellaneous, model, device):
    """
    Use trained GCN model to generate initial pred_assort and probability matrix
    
    Args:
        dat: graph data
        miscellaneous: data parameters
        model: loaded model object
        device: torch device
    
    Returns:
        tuple: (initial_pred_assort, prob)
            - initial_pred_assort: [m, n] binary matrix
            - prob: [m, n] probability matrix
    """
    n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = miscellaneous
    m = segment_num

    # Move data to device
    dat = dat.to(device)

    with torch.no_grad():
        raw_out = model(dat)
        
    # Process model output (same logic as run_inference_only in test_PCP_multi_model_avg.py)
    if isinstance(raw_out, dict):
        # New edge-scoring model output
        if "logit_matrix" in raw_out:
            logits_nm = raw_out["logit_matrix"].detach().cpu().numpy()  # shape (n, m)
        elif "edge_logits" in raw_out:
            s = raw_out["edge_logits"].detach().cpu().numpy()
            logits_nm = s.reshape(n, m)  # shape (n, m)
        else:
            raise ValueError("Unexpected model output keys: " + ",".join(raw_out.keys()))
        
        # Apply sigmoid: shape (n, m)
        sigmoid_output = 1.0 / (1.0 + np.exp(-logits_nm))
    else:
        # Old node-level model output
        output = raw_out[:n, :].detach().cpu().numpy()  # shape (n, m)
        sigmoid_output = np.exp(output) / (np.exp(output) + np.exp(1))
    
    # Convert to [m, n] format for compatibility with local search
    # sigmoid_output is (n, m), transpose to (m, n)
    prob = sigmoid_output.T  # shape (m, n)
    
    # Generate initial pred_assort: shape (m, n)
    # Use threshold 0.5 for binary prediction
    initial_pred_assort = (prob >= 0.5).astype(int)

    return initial_pred_assort, prob


# ============================================================================
# Local Search Functions
# ============================================================================


def generate_neighbor_assignments_global_topk(current_assignment, prob, n, m):
    """
    Generate neighbor assignments using global Top-K strategy with mixed ordering
    
    Instead of generating 2*m neighbors (one Add and one Drop per segment),
    this function generates at most 2*K neighbors where K = ceil(sqrt(m)).
    
    Strategy:
    - Add candidates: score = prob[k, j] (higher is better)
    - Drop candidates: score = 1 - prob[k, j] (higher is better)
    - Mix both types and sort by score (descending)
    
    Args:
        current_assignment: dict, current segment-bundle assignment
        prob: [m, n] GCN output probability matrix
        n: number of products
        m: number of customer segments
    
    Returns:
        list of neighbor assignments, ordered by priority
    """
    current_pred_assort = assignment_to_pred_assort(current_assignment, n, m)
    K = int(ceil(sqrt(m)))
    
    neighbors = []
    add_candidates = []
    for k in range(m):
        for j in range(n):
            if current_pred_assort[k, j] == 0:
                score_add = prob[k, j]
                add_candidates.append((k, j, score_add, 'add'))
    
    add_candidates.sort(key=lambda x: x[2], reverse=True)
    add_list = add_candidates[:K]

    drop_candidates = []
    for k in range(m):
        for j in range(n):
            if current_pred_assort[k, j] == 1:
                score_drop = 1 - prob[k, j]
                drop_candidates.append((k, j, score_drop, 'drop'))
    
    drop_candidates.sort(key=lambda x: x[2], reverse=True)
    drop_list = drop_candidates[:K]

    mixed_candidates = add_list + drop_list
    mixed_candidates.sort(key=lambda x: x[2], reverse=True)
    
    for k, j, score, op_type in mixed_candidates:
        neighbor_pred = current_pred_assort.copy()
        if op_type == 'add':
            neighbor_pred[k, j] = 1
        else:
            neighbor_pred[k, j] = 0
        neighbor_assignment = convert_pred_assort_to_assignment(neighbor_pred)
        neighbors.append(neighbor_assignment)

    return neighbors


def local_search_with_lp_global_topk(
    initial_assignment,
    initial_milp_ratio,
    prob,
    meta,
    max_iterations=50,
    tolerance=1e-6,
    silent=False,
    plot_time_offset=0.0,
    record_history=False,
    wall_history=None,
    wall_time_start=None,
):
    """
    Local search function using global Top-K neighbor generation strategy
    
    Workflow:
    1. Cached LP solve to get current best competitive ratio from the initial MILP assignment
    2. Neighborhood search loop with global Top-K:
       - Generate at most 2*K neighbors (K = ceil(sqrt(m)))
       - LP feasibility check
       - Relative objective/profit improvement check
       - Update best solution
    3. Convert optimal assignment to pred_assort
    4. Final MILP solve (verify LP result)
    
    Args:
        initial_assignment: initial feasible segment-bundle assignment from MILP
        initial_milp_ratio: competitive ratio of the initial MILP solution
        prob: [m, n] GCN output probability matrix
        meta: data parameter tuple
        max_iterations: maximum number of iterations
        tolerance: relative objective/profit improvement tolerance
        silent: if True, suppress detailed output
    
    Returns:
        tuple: (final_pred_assort, final_competitive_ratio, search_info)
    """
    n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = meta
    m = segment_num
    K = int(ceil(sqrt(m)))
    current_assignment = initial_assignment.copy()

    if not silent:
        print(f"Top-K parameter: K={K} (max neighbors per iteration: {2*K})")
        print("Step 1: Initial cached LP solve...")
    search_info_events = [] if record_history else None
    plot_time_cursor = float(plot_time_offset)
    lp_cache = ReusableBundleLPCache(
        n,
        m,
        unit_cs,
        ship_cs,
        unit_us,
        Ns,
        opt_rev,
        stored_cs,
        stored_Rs,
    )
    is_initial_feasible, current_competitive_ratio, current_objective, initial_lp_time = lp_cache.solve(
        initial_assignment,
        obj_history=search_info_events,
        history_time_offset=plot_time_cursor,
        history_source="initial_lp",
    )
    if not is_initial_feasible:
        lp_cache.dispose()
        raise ValueError("Initial assignment is infeasible in cached LP solve")
    plot_time_cursor += initial_lp_time
    
    if not silent:
        print(f"Initial LP result: competitive ratio={current_competitive_ratio:.6f}, time={initial_lp_time:.4f}s")
    
    search_info = {
        'initial_milp_competitive_ratio': initial_milp_ratio,
        'initial_lp_competitive_ratio': current_competitive_ratio,
        'initial_lp_objective': current_objective,
        'iterations': 0,
        'improvements': 0,
        'lp_solver_calls': 1,
        'milp_solver_calls': 1,
        'K': K,
        'relative_objective_tolerance': float(tolerance),
    }
    if record_history:
        search_info['objective_events'] = search_info_events

    def _append_wall_event(value, source):
        if wall_history is None or wall_time_start is None:
            return
        _append_objective_event(
            wall_history,
            time.time() - wall_time_start,
            value,
            source,
        )

    _append_wall_event(current_competitive_ratio, 'initial_lp')
    
    if not silent:
        print("Step 2: Local Search loop...")
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        neighbors = generate_neighbor_assignments_global_topk(current_assignment, prob, n, m)
        
        actual_neighbors = len(neighbors)
        if not silent:
            print(f"Iteration {iteration}: Evaluating {actual_neighbors} neighbors (max: {2*K})")
        
        for neighbor_idx, neighbor_assignment in enumerate(neighbors):
            is_feasible, neighbor_competitive_ratio, neighbor_objective, neighbor_lp_time = lp_cache.solve(neighbor_assignment)
            plot_time_cursor += neighbor_lp_time
            
            search_info['lp_solver_calls'] += 1
            
            objective_improvement = neighbor_objective - current_objective
            objective_improvement_ratio = (
                objective_improvement / max(abs(current_objective), 1e-12)
                if np.isfinite(current_objective)
                else -np.inf
            )
            if is_feasible and objective_improvement_ratio > tolerance:
                current_assignment = neighbor_assignment
                current_competitive_ratio = neighbor_competitive_ratio
                current_objective = neighbor_objective
                improved = True
                search_info['improvements'] += 1
                if record_history:
                    _append_objective_event(
                        search_info['objective_events'],
                        plot_time_cursor,
                        current_competitive_ratio,
                        'local_search_lp',
                    )
                _append_wall_event(current_competitive_ratio, 'local_search_lp')
                if not silent:
                    print(f"Iteration {iteration}: Found improvement at neighbor {neighbor_idx+1}/{actual_neighbors}, "
                          f"competitive ratio={current_competitive_ratio:.6f}, "
                          f"objective improvement={objective_improvement:.6g}, "
                          f"relative objective improvement={objective_improvement_ratio:.6g}")
                break
        
        if not improved and not silent:
            print(f"Iteration {iteration}: No improvement found after evaluating {actual_neighbors} neighbors, search converged")
    
    search_info['iterations'] = iteration
    search_info['final_lp_competitive_ratio'] = current_competitive_ratio
    search_info['final_lp_objective'] = current_objective
    search_info['lp_improvement'] = current_competitive_ratio - search_info['initial_lp_competitive_ratio']
    search_info['lp_objective_improvement'] = current_objective - search_info['initial_lp_objective']
    search_info.update(lp_cache.stats())
    
    if not silent:
        print("Step 3: Converting optimal assignment to pred_assort...")
    final_pred_assort = assignment_to_pred_assort(current_assignment, n, m)

    if not silent:
        print("Step 4: Final MILP solve (verify LP result)...")
    final_milp_ratio, final_milp_time = competitive_ratio_with_optimal_bundle(
        n,
        m,
        unit_cs,
        ship_cs,
        unit_us,
        Ns,
        opt_rev,
        final_pred_assort,
        stored_cs,
        stored_Rs,
        obj_history=search_info.get('objective_events') if record_history else None,
        history_time_offset=plot_time_cursor,
        history_source_prefix="final_milp",
    )[:2]
    plot_time_cursor += final_milp_time
    
    search_info['final_milp_competitive_ratio'] = final_milp_ratio
    search_info['final_milp_time'] = final_milp_time
    search_info['milp_solver_calls'] += 1
    search_info['total_improvement'] = final_milp_ratio - search_info['initial_milp_competitive_ratio']
    search_info['plot_end_time'] = plot_time_cursor
    if record_history:
        _append_objective_event(
            search_info['objective_events'],
            plot_time_cursor,
            final_milp_ratio,
            'final_milp',
        )
    _append_wall_event(final_milp_ratio, 'final_milp')
    
    if not silent:
        print(f"Final MILP result: competitive ratio={final_milp_ratio:.6f}, time={final_milp_time:.4f}s")
        print(f"Total improvement: {search_info['total_improvement']:.6f}")
    
    lp_cache.dispose()
    return final_pred_assort, final_milp_ratio, search_info


def _save_layer_result(nl: int, m: int, n: int, results_list: list, result_dir: str, test_folder_name: str = None, silent: bool = False) -> None:
    """Save results for a single layer and problem size to CSV file"""
    import pandas as pd
    
    results_array = np.array(results_list)
    
    # Filename
    if test_folder_name:
        result_filename = f"test_result_FCPLS_{nl}layer_{test_folder_name}.csv"
    else:
        result_filename = f"test_result_FCPLS_{nl}layer_m{m}n{n}.csv"
    result_path = os.path.join(result_dir, result_filename)
    
    # Create DataFrame
    df = pd.DataFrame({
        'method': 'FCPLS',
        'layers': nl,
        'm_segments': m,
        'n_products': n,
        'competitive_ratio': results_array[:, 0],
        'runtime_ratio': results_array[:, 1],
        'total_time': results_array[:, 2],
        'gcn_time': results_array[:, 3],
        'initial_milp_time': results_array[:, 4],
        'local_search_time': results_array[:, 5],
        'gnn_gurobi_time': results_array[:, 6],
        'plot_axis_time': results_array[:, 6],
    })
    df.to_csv(result_path, index=False)
    if not silent:
        print(f"✅ Saved: {result_path} ({len(results_array)} samples)")


def _save_seed_averages(nl: int, m: int, n: int, seed_results: dict, result_dir: str, test_folder_name: str = None, silent: bool = False) -> None:
    """Save average results for each seed to CSV file"""
    import pandas as pd
    
    # Calculate average for each seed
    seed_avg_data = []
    for seed, results in sorted(seed_results.items()):
        if len(results) > 0:
            results_array = np.array(results)
            seed_avg_data.append({
                'seed': seed,
                'num_samples': len(results),
                'avg_competitive_ratio': float(np.mean(results_array[:, 0])),
                'std_competitive_ratio': float(np.std(results_array[:, 0])),
                'avg_time_ratio': float(np.mean(results_array[:, 1])),
                'std_time_ratio': float(np.std(results_array[:, 1])),
                'avg_total_time': float(np.mean(results_array[:, 2])),
                'avg_gcn_time': float(np.mean(results_array[:, 3])),
                'avg_initial_milp_time': float(np.mean(results_array[:, 4])),
                'avg_local_search_time': float(np.mean(results_array[:, 5])),
                'avg_gnn_gurobi_time': float(np.mean(results_array[:, 6])),
                'avg_plot_axis_time': float(np.mean(results_array[:, 6])),
            })
    
    if not seed_avg_data:
        return
    
    # Filename
    if test_folder_name:
        result_filename = f"test_result_FCPLS_{nl}layer_{test_folder_name}_seed_avg.csv"
    else:
        result_filename = f"test_result_FCPLS_{nl}layer_m{m}n{n}_seed_avg.csv"
    result_path = os.path.join(result_dir, result_filename)
    
    # Create DataFrame
    df = pd.DataFrame(seed_avg_data)
    
    # Add metadata columns
    df.insert(0, 'method', 'FCPLS')
    df.insert(1, 'layers', nl)
    df.insert(2, 'm_segments', m)
    df.insert(3, 'n_products', n)
    
    df.to_csv(result_path, index=False)
    if not silent:
        print(f"✅ Saved seed average results: {result_path} ({len(seed_avg_data)} seeds)")


def main():
    """
    Main function: Multi-model average evaluation for FCP Local Search
    """
    parser = argparse.ArgumentParser(description="Multi-model average evaluation (FCP Local Search with cached LP skeletons)")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--test_subdirs", type=str, default="dataset/test_BSP_m10n30_1e_3/;", 
                        help="Test data subdirectories (separated by semicolon)")
    parser.add_argument("--model_dir", type=str, default="models_multi_layer_edge_update")
    parser.add_argument("--layers", type=str, default="4", help="Layers list, comma separated")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10", help="Seeds list, comma separated")
    parser.add_argument("--result_dir", type=str, default="test_results_FCPLS_cached_lp_percent_1e_3", help="Results save directory")
    parser.add_argument("--save_result", type=bool, default=True, help="Whether to save result files")
    parser.add_argument("--max_iterations", type=int, default=50, help="Max local search iterations")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Relative objective improvement tolerance")
    parser.add_argument("--silent", type=bool, default=True, help="Silent mode: suppress iteration details")
    parser.add_argument(
        "--clear_internal_caches_each_seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear Rs/cs and subadditivity caches after each seed to avoid long-run memory growth.",
    )
    parser.add_argument(
        "--dispose_gurobi_env_every_samples",
        type=int,
        default=10,
        help="Dispose Gurobi default environment every N samples; use 0 to disable.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save the average objective-ratio-vs-time plot with std shading.",
    )
    args = parser.parse_args()

    dir_path = args.data_dir
    test_subdirs = parse_paths_arg(args.test_subdirs)
    model_root = os.path.join(dir_path, args.model_dir)
    result_dir = os.path.join(dir_path, args.result_dir) if (args.save_result or args.plot) else None

    layers = parse_list_arg(args.layers)
    seeds = parse_list_arg(args.seeds)

    # Create results directory
    if args.save_result or args.plot:
        os.makedirs(result_dir, exist_ok=True)

    # Always show configuration (not affected by silent mode)
    print("=" * 80)
    print("Global Top-K Local Search Multi-Model Average Evaluation (Cached LP Skeleton)")
    print("Strategy: K = ceil(sqrt(m)), max neighbors per iteration = 2*K")
    print("Add score = P[k,j], Drop score = 1-P[k,j], mixed and sorted by score")
    print("LP cache: keyed by active bundle set; updates objective and upper constraints only")
    print("=" * 80)
    print(f"📂 data_dir: {dir_path}")
    print(f"🧪 test_subdirs: {test_subdirs}")
    print(f"🗂 model_root: {model_root}")
    print(f"🔢 layers: {layers}")
    print(f"🌱 seeds: {seeds}")
    if args.save_result:
        print(f"💾 result_dir: {result_dir}")
    else:
        print(f"💾 Not saving result files")
    print(f"📈 plot objective curves: {'ON' if args.plot else 'OFF'}")
    print(
        "🧹 memory guard: "
        f"clear caches each seed={'ON' if args.clear_internal_caches_each_seed else 'OFF'}, "
        f"dispose Gurobi env every {args.dispose_gurobi_env_every_samples} samples"
    )
    if args.silent:
        print(f"🔇 Silent mode: ON (suppressing iteration details)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models (organized by layer)
    models_by_layer = load_models(model_root, layers, seeds, device, silent=args.silent)
    if not any(models_by_layer.values()):
        print("No models loaded, exiting.")
        return

    # Store results by layer (sample-level average)
    # Structure: {layer: {(m, n): [(competitive_ratio, time_ratio, total_time, gcn_time, initial_milp_time, local_search_time, gnn_gurobi_time), ...]}}
    results_by_layer = {nl: {} for nl in layers}
    
    # Store results by layer and seed (each seed separately)
    # Structure: {layer: {seed: {(m, n): [(competitive_ratio, time_ratio, total_time, gcn_time, initial_milp_time, local_search_time, gnn_gurobi_time), ...]}}}
    results_by_seed = {nl: {sd: {} for sd, _, _ in models_by_layer[nl]} for nl in layers}
    
    # Track saved (layer, m, n) combinations
    saved_keys = set()

    # Iterate through each dataset
    for test_subdir in test_subdirs:
        test_data_path = os.path.join(dir_path, test_subdir)

        if not os.path.exists(test_data_path):
            if not args.silent:
                print(f"⚠️ Test data path does not exist: {test_data_path}, skipping")
            continue

        dir_list = os.listdir(test_data_path)
        test_dataset = []
        misc_dataset = []
        file_names = []

        if not args.silent:
            print(f"\n📊 Loading test set: {test_subdir}")
        for fname in dir_list:
            if fname == ".DS_Store":
                continue
            fpath = os.path.join(test_data_path, fname)
            try:
                dat, misc = process_data(fpath)
                test_dataset.append(dat)
                misc_dataset.append(misc)
                file_names.append(fname)
            except Exception as e:
                if not args.silent:
                    print(f"Failed to read {fname}: {e}")
                continue

        sample_num = len(test_dataset)
        if not args.silent:
            print(f"✅ Loaded {sample_num} samples")
        if sample_num == 0:
            continue

        plot_histories_by_instance = None
        if args.plot:
            plot_histories_by_instance = {
                nl: {
                    "wall_clock": {},
                    "gnn_gurobi": {},
                }
                for nl in layers
            }

        # Use tqdm for progress tracking
        for i in tqdm(range(sample_num), desc=f"Evaluating {test_subdir}"):
            try:
                dat = test_dataset[i]
                misc = misc_dataset[i]
                n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = misc

                # For each layer, use all seeds for inference
                for nl in layers:
                    if nl not in models_by_layer or not models_by_layer[nl]:
                        continue
                    
                    model_competitive_ratios = []
                    model_time_ratios = []
                    model_total_times = []
                    model_gcn_times = []
                    model_initial_milp_times = []
                    model_local_search_times = []
                    model_gnn_gurobi_times = []

                    for sd, mdl, path in models_by_layer[nl]:
                        try:
                            strategy_start_time = time.time()
                            gnn_gurobi_history = []
                            wall_clock_history = []
                            if args.plot:
                                _append_objective_event(gnn_gurobi_history, 0.0, 0.0, "start")
                                _append_objective_event(wall_clock_history, 0.0, 0.0, "start")
                            
                            # Step 1: GCN preparation time before the first MILP
                            gcn_start = time.time()
                            initial_pred, prob = predict_initial_bundles(dat, misc, mdl, device)
                            gcn_time = time.time() - gcn_start
                            
                            # Step 2: Initial MILP solve
                            initial_milp_start = time.time()
                            initial_competitive_ratio, initial_milp_gurobi_time, initial_assignment = competitive_ratio_with_optimal_bundle(
                                n,
                                segment_num,
                                unit_cs,
                                ship_cs,
                                unit_us,
                                Ns,
                                opt_rev,
                                initial_pred,
                                stored_cs,
                                stored_Rs,
                                obj_history=gnn_gurobi_history if args.plot else None,
                                history_time_offset=gcn_time,
                                history_source_prefix="initial_milp",
                            )
                            initial_milp_time = time.time() - initial_milp_start
                            gnn_gurobi_time_cursor = gcn_time + initial_milp_gurobi_time
                            if args.plot:
                                _append_objective_event(
                                    wall_clock_history,
                                    time.time() - strategy_start_time,
                                    initial_competitive_ratio,
                                    "initial_milp",
                                )
                            
                            # Step 3: Local search optimization
                            local_search_start = time.time()
                            best_pred, best_rev, search_info = local_search_with_lp_global_topk(
                                initial_assignment,
                                initial_competitive_ratio,
                                prob,
                                misc,
                                args.max_iterations,
                                args.tolerance,
                                silent=args.silent,
                                plot_time_offset=gnn_gurobi_time_cursor,
                                record_history=args.plot,
                                wall_history=wall_clock_history if args.plot else None,
                                wall_time_start=strategy_start_time if args.plot else None,
                            )
                            local_search_time = time.time() - local_search_start
                            
                            total_time = time.time() - strategy_start_time
                            time_ratio = total_time / running_time if running_time > 0 else float("inf")
                            if args.plot:
                                gnn_gurobi_history.extend(search_info.get("objective_events", []))
                                gnn_gurobi_end_time = search_info.get("plot_end_time", gnn_gurobi_time_cursor)
                                _append_objective_event(
                                    gnn_gurobi_history,
                                    gnn_gurobi_end_time,
                                    best_rev,
                                    "end",
                                )
                                _append_objective_event(
                                    wall_clock_history,
                                    total_time,
                                    best_rev,
                                    "end",
                                )
                            else:
                                gnn_gurobi_end_time = search_info.get("plot_end_time", gnn_gurobi_time_cursor)

                            model_competitive_ratios.append(best_rev)
                            model_time_ratios.append(time_ratio)
                            model_total_times.append(total_time)
                            model_gcn_times.append(gcn_time)
                            model_initial_milp_times.append(initial_milp_time)
                            model_local_search_times.append(local_search_time)
                            model_gnn_gurobi_times.append(gnn_gurobi_end_time)
                            
                            key = (segment_num, n)
                            if key not in results_by_seed[nl][sd]:
                                results_by_seed[nl][sd][key] = []
                            results_by_seed[nl][sd][key].append([
                                best_rev,
                                time_ratio,
                                total_time,
                                gcn_time,
                                initial_milp_time,
                                local_search_time,
                                gnn_gurobi_end_time,
                            ])
                            if args.plot and plot_histories_by_instance is not None:
                                for time_kind, history in (
                                    ("wall_clock", wall_clock_history),
                                    ("gnn_gurobi", gnn_gurobi_history),
                                ):
                                    if key not in plot_histories_by_instance[nl][time_kind]:
                                        plot_histories_by_instance[nl][time_kind][key] = {}
                                    if i not in plot_histories_by_instance[nl][time_kind][key]:
                                        plot_histories_by_instance[nl][time_kind][key][i] = {}
                                    plot_histories_by_instance[nl][time_kind][key][i][sd] = history
                            
                        except Exception as e_model:
                            if not args.silent:
                                print(f"Model failed sample={file_names[i] if i < len(file_names) else i}, layer={nl}, seed={sd}, path={path}: {e_model}")
                                import traceback
                                traceback.print_exc()
                            continue
                        finally:
                            if args.clear_internal_caches_each_seed:
                                clear_internal_caches()
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                    if len(model_competitive_ratios) > 0:
                        key = (segment_num, n)
                        if key not in results_by_layer[nl]:
                            results_by_layer[nl][key] = []
                        
                        result_entry = [
                            float(np.mean(model_competitive_ratios)),
                            float(np.mean(model_time_ratios)),
                            float(np.mean(model_total_times)),
                            float(np.mean(model_gcn_times)),
                            float(np.mean(model_initial_milp_times)),
                            float(np.mean(model_local_search_times)),
                            float(np.mean(model_gnn_gurobi_times)),
                        ]
                        results_by_layer[nl][key].append(result_entry)
            except Exception as e:
                if not args.silent:
                    print(f"Sample {file_names[i] if i < len(file_names) else i} evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
                continue
            finally:
                clear_internal_caches()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (
                    args.dispose_gurobi_env_every_samples > 0
                    and (i + 1) % args.dispose_gurobi_env_every_samples == 0
                ):
                    gp.disposeDefaultEnv()

        # After processing this dataset, save and print results
        test_folder_name = os.path.basename(test_subdir.rstrip('/'))
        
        # Save results
        if args.save_result:
            if not args.silent:
                print(f"\n💾 Saving results for {test_folder_name}...")
            for nl in layers:
                for key, results in results_by_layer[nl].items():
                    full_key = (nl, key[0], key[1])  # (layer, m, n)
                    if full_key not in saved_keys and results:
                        m, n_prod = key
                        # Save sample-level average results
                        _save_layer_result(nl, m, n_prod, results, result_dir, test_folder_name, silent=args.silent)
                        
                        # Save per-seed average results
                        seed_results_for_key = {
                            sd: results_by_seed[nl][sd].get(key, [])
                            for sd in results_by_seed[nl].keys()
                        }
                        _save_seed_averages(nl, m, n_prod, seed_results_for_key, result_dir, test_folder_name, silent=args.silent)
                        
                        saved_keys.add(full_key)
        if args.plot and plot_histories_by_instance is not None:
            plot_specs = {
                "wall_clock": "Wall Clock Time (s)",
                "gnn_gurobi": "GNN + Gurobi Solve Time (s)",
            }
            for nl in layers:
                for time_kind, time_label in plot_specs.items():
                    for key in sorted(plot_histories_by_instance[nl][time_kind].keys()):
                        m, n_prod = key
                        _save_plot_for_key(
                            nl,
                            m,
                            n_prod,
                            plot_histories_by_instance[nl][time_kind][key],
                            result_dir,
                            test_folder_name,
                            time_kind=time_kind,
                            time_label=time_label,
                            silent=args.silent,
                            interval=5.0,
                        )
        
        # Print statistics for current dataset
        print(f"\n{'='*80}")
        print(f"📊 {test_folder_name} Evaluation Results")
        print(f"{'='*80}")
        for nl in layers:
            dataset_results = {}
            for key, results in results_by_layer[nl].items():
                full_key = (nl, key[0], key[1])
                # Only stats for current dataset (newly added)
                if full_key in saved_keys or not args.save_result:
                    dataset_results[key] = results
            
            if dataset_results:
                total_samples = sum(len(v) for v in dataset_results.values())
                print(f"\n【Layer {nl}】Sample count: {total_samples}")
                for (m, n_prod), results in sorted(dataset_results.items()):
                    if results:
                        results_array = np.array(results)
                        print(f"  m={m}, n={n_prod}: {len(results_array)} samples")
                        print(f"    Competitive Ratio - Mean: {np.mean(results_array[:, 0]):.4f}, Std: {np.std(results_array[:, 0]):.4f}, "
                              f"Min: {np.min(results_array[:, 0]):.4f}, Max: {np.max(results_array[:, 0]):.4f}")
                        print(f"    Time Ratio - Mean: {np.mean(results_array[:, 1]):.4f}, Std: {np.std(results_array[:, 1]):.4f}, "
                              f"Min: {np.min(results_array[:, 1]):.4f}, Max: {np.max(results_array[:, 1]):.4f}")
                        print(f"    Avg Times - Total: {np.mean(results_array[:, 2]):.4f}s, GCN: {np.mean(results_array[:, 3]):.4f}s, "
                              f"Initial MILP: {np.mean(results_array[:, 4]):.4f}s, Local Search: {np.mean(results_array[:, 5]):.4f}s, "
                              f"GNN+Gurobi: {np.mean(results_array[:, 6]):.4f}s")
                        print(f"    GNN+Gurobi Time - Mean: {np.mean(results_array[:, 6]):.4f}s, Std: {np.std(results_array[:, 6]):.4f}s, "
                              f"Min: {np.min(results_array[:, 6]):.4f}s, Max: {np.max(results_array[:, 6]):.4f}s")
        print(f"{'='*80}\n")

    # Final summary statistics (always shown)
    print("\n" + "="*80)
    print("🎉 All datasets evaluation complete!")
    print("="*80)
    print(f"Test datasets count: {len(test_subdirs)}")
    print(f"Layers: {layers}")
    print(f"Models per layer: {len(seeds)}")
    if args.save_result:
        print(f"Results directory: {result_dir}")
    else:
        print(f"Result files: Not saved")
    
    print("\nOverall statistics:")
    for nl in layers:
        total_samples = sum(len(v) for v in results_by_layer[nl].values())
        print(f"  【Layer {nl}】Total samples: {total_samples}")
    print("="*80)


if __name__ == "__main__":
    main()
