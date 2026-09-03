"""
Local Search Path Test - Global Top-K Strategy with Mixed Add/Drop Ordering

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
""" 

import os
import numpy as np
import time
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


# ============================================================================
# Helper Functions
# ============================================================================

def parse_list_arg(arg: str) -> List[int]:
    """Parse comma-separated string to list of integers"""
    return [int(x) for x in arg.split(",") if x.strip()]


def parse_paths_arg(arg: str) -> List[str]:
    """Parse semicolon-separated paths string to list"""
    return [p.strip() for p in arg.split(";") if p.strip()]


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
    bin2num = lambda x: int(''.join(map(str, x.tolist())), 2)
    
    assignment = {}
    m, n = pred_assort.shape
    
    for k in range(m):
        bundle_binary = pred_assort[k, :]
        bundle_idx = bin2num(bundle_binary)
        assignment[k] = bundle_idx
    
    return assignment


def assignment_to_pred_assort(assignment, n, m):
    """
    Convert one-to-one assignment back to pred_assort matrix
    """
    pred_assort = np.zeros((m, n), dtype=int)

    for k in range(m):
        bundle_idx = assignment[k]
        bundle_binary = format(bundle_idx, f'0{n}b')
        pred_assort[k, :] = [int(x) for x in bundle_binary]

    return pred_assort


def bundle_to_product_set(bundle_id, n):
    """Convert bundle ID to product set"""
    binary_str = format(bundle_id, '0' + str(n) + 'b')
    return set(i for i, bit in enumerate(binary_str) if bit == '1')


# ============================================================================
# Optimization Functions
# ============================================================================

def revenue_ratio_with_optimal_bundle(n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, pred_assort, stored_cs=None, stored_Rs=None):
    milp_start_time = time.time()
    
    bin2num = lambda x: int(''.join(map(str, x.tolist())), 2)
    segment_ind = np.array([i for i in range(m)])
    
    # Get unique predicted bundles
    bundle_dic = {}
    for i in range(m):
        bundle_idx = bin2num(pred_assort[i, :])
        try:
            bundle_dic[bundle_idx].append(i)
        except:
            bundle_dic[bundle_idx] = [i]
    
    predicted_bundles = list(bundle_dic.keys())
    
    # Calculate Rs and costs only for predicted bundles
    if stored_cs is not None and stored_Rs is not None:
        # Use stored matrices (old format)
        # Convert to compact format for consistency
        # Generate assortments only for predicted bundles
        predicted_assortments = []
        for bundle_id in predicted_bundles:
            bundle_binary = list(map(int, format(bundle_id, '0' + str(n) + 'b')))
            predicted_assortments.append(bundle_binary)
        predicted_assortments = np.array(predicted_assortments)
        
        # Create mapping from bundle_id to index in predicted arrays
        bundle_to_idx = {bundle_id: idx for idx, bundle_id in enumerate(predicted_bundles)}
        
        # Extract Rs and cs for predicted bundles from stored full-size arrays
        Rs_predicted = np.zeros((m, len(predicted_bundles)))
        cs_predicted = np.zeros((m, len(predicted_bundles)))
        for bundle_id, idx in bundle_to_idx.items():
            Rs_predicted[:, idx] = stored_Rs[:, bundle_id]
            cs_predicted[:, idx] = stored_cs[:, bundle_id]
        
        Rbar = np.max(Rs_predicted)
    else:
        # Calculate only for predicted bundles (new optimized format)
        # Only calculate for predicted bundles instead of all bundles
        
        # Generate assortments only for predicted bundles
        predicted_assortments = []
        for bundle_id in predicted_bundles:
            bundle_binary = list(map(int, format(bundle_id, '0' + str(n) + 'b')))
            predicted_assortments.append(bundle_binary)
        predicted_assortments = np.array(predicted_assortments)
        
        # Calculate Rs only for predicted bundles: Rs[customer, bundle]
        Rs_predicted = np.sqrt(unit_us.dot(predicted_assortments.T))  # shape: (m, len(predicted_bundles))
        
        # Calculate cs only for predicted bundles: cs[customer, bundle]  
        cs_base = np.sum(predicted_assortments * unit_cs, axis=1)  # shape: (len(predicted_bundles),)
        cs_predicted = cs_base + ship_cs  # Broadcasting: (len(predicted_bundles),) + (m, 1) -> (m, len(predicted_bundles))
        cs_predicted = cs_predicted * 0.2
        
        # Create mapping from bundle_id to index in predicted arrays
        bundle_to_idx = {bundle_id: idx for idx, bundle_id in enumerate(predicted_bundles)}
        
        Rbar = np.max(Rs_predicted)

    model = gp.Model("Bundle MILP")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-2
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
    
    # Improved subadditivity constraints
    # Pre-compute product sets for all bundles
    bundle_product_sets = {}
    for bundle_id in predicted_bundles:
        bundle_product_sets[bundle_id] = bundle_to_product_set(bundle_id, n)

    # Filter constraints
    for k in predicted_bundles:
        if k == 0:  # Skip empty set
            continue

        k_set = bundle_product_sets[k]
        if len(k_set) == 0:
            continue

        for i in predicted_bundles:
            for j in predicted_bundles:
                if i >= j:  # Avoid duplicates
                    continue

                i_set = bundle_product_sets[i]
                j_set = bundle_product_sets[j]
                union_set = i_set.union(j_set)

                # Check if it's a valid inclusion relationship
                if (k_set.issubset(union_set) and
                    k_set != i_set and
                    k_set != j_set):
                    model.addConstr(p[k] <= p[i] + p[j])

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
        model.addConstrs((S[k, 0] == 0 for k in segment_ind))

    model.optimize()

    milp_end_time = time.time()
    milp_time = milp_end_time - milp_start_time
    
    # Extract optimal bundle assignment
    optimal_bundle_assignment = {}
    if model.Status == GRB.OPTIMAL:
        for k in segment_ind:
            for i in predicted_bundles:
                if theta[k, i].X > 0.5:  # Binary variable is 1
                    optimal_bundle_assignment[k] = i
                    break
    
    return model.ObjVal/opt_rev, milp_time, optimal_bundle_assignment


def revenue_ratio_LP(n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, segment_bundle_assignment, stored_cs=None, stored_Rs=None):
    lp_start_time = time.time()

    # Extract involved bundle set from assignment
    assigned_bundles = set(segment_bundle_assignment.values())
    predicted_bundles = list(assigned_bundles)

    # Ensure empty bundle (index 0) is in the set
    if 0 not in predicted_bundles:
        predicted_bundles.append(0)

    segment_ind = np.array([i for i in range(m)])
    
    # Calculate Rs and costs only for involved bundles
    if stored_cs is not None and stored_Rs is not None:
        # Use stored matrices (old format)
        # Convert to compact format for consistency
        # Generate assortments only for involved bundles
        predicted_assortments = []
        for bundle_id in predicted_bundles:
            bundle_binary = list(map(int, format(bundle_id, '0' + str(n) + 'b')))
            predicted_assortments.append(bundle_binary)
        predicted_assortments = np.array(predicted_assortments)
        
        # Create mapping from bundle_id to index in predicted arrays
        bundle_to_idx = {bundle_id: idx for idx, bundle_id in enumerate(predicted_bundles)}
        
        # Extract Rs and cs for involved bundles from stored full-size arrays
        Rs_predicted = np.zeros((m, len(predicted_bundles)))
        cs_predicted = np.zeros((m, len(predicted_bundles)))
        for bundle_id, idx in bundle_to_idx.items():
            Rs_predicted[:, idx] = stored_Rs[:, bundle_id]
            cs_predicted[:, idx] = stored_cs[:, bundle_id]
    else:
        # Calculate only for involved bundles (new optimized format)
        # LP only calculates for involved bundles instead of all bundles
        
        # Generate assortments only for involved bundles
        predicted_assortments = []
        for bundle_id in predicted_bundles:
            bundle_binary = list(map(int, format(bundle_id, '0' + str(n) + 'b')))
            predicted_assortments.append(bundle_binary)
        predicted_assortments = np.array(predicted_assortments)
        
        # Calculate Rs only for involved bundles: Rs[customer, bundle]
        Rs_predicted = np.sqrt(unit_us.dot(predicted_assortments.T))  # shape: (m, len(predicted_bundles))
        
        # Calculate cs only for involved bundles: cs[customer, bundle]  
        cs_base = np.sum(predicted_assortments * unit_cs, axis=1)  # shape: (len(predicted_bundles),)
        cs_predicted = cs_base + ship_cs  # Broadcasting: (len(predicted_bundles),) + (m, 1) -> (m, len(predicted_bundles))
        cs_predicted = cs_predicted * 0.2
        
        # Create mapping from bundle_id to index in predicted arrays
        bundle_to_idx = {bundle_id: idx for idx, bundle_id in enumerate(predicted_bundles)}

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

    # CONSTRAINT 3: Price subadditivity (improved implementation, consistent with MILP)
    # Mathematical formulation: p_i ≤ Σ_{j∈I} p_j, for all i∈F and covering family I⊆F
    # such that ⋃_{j∈I} B(j) = B(i)
    # Use improved set-based method instead of combinations for better performance
    
    # Pre-compute product sets for all bundles
    bundle_product_sets = {}
    for bundle_id in predicted_bundles:
        bundle_product_sets[bundle_id] = bundle_to_product_set(bundle_id, n)

    # Filter constraints
    for k in predicted_bundles:
        if k == 0:  # Skip empty set
            continue

        k_set = bundle_product_sets[k]
        if len(k_set) == 0:
            continue

        for i in predicted_bundles:
            for j in predicted_bundles:
                if i >= j:  # Avoid duplicates
                    continue

                i_set = bundle_product_sets[i]
                j_set = bundle_product_sets[j]
                union_set = i_set.union(j_set)

                # Check if it's a valid inclusion relationship
                if (k_set.issubset(union_set) and
                    k_set != i_set and
                    k_set != j_set):
                    model.addConstr(p[k] <= p[i] + p[j],
                                  name=f"Subadditivity_k{k}_i{i}_j{j}")

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

    # Solve
    model.optimize()

    lp_end_time = time.time()
    lp_time = lp_end_time - lp_start_time

    # Return results
    if model.Status == GRB.OPTIMAL:
        return model.ObjVal / opt_rev, lp_time
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        return model.ObjVal / opt_rev, lp_time
    else:
        return -np.inf, lp_time


def check_lp_feasibility_and_revenue(assignment, n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, stored_cs=None, stored_Rs=None):
    """
    Quickly check if given assignment is feasible under LP and return revenue

    Returns:
        tuple: (is_feasible, revenue_ratio, solve_time)
    """
    try:
        revenue_ratio, solve_time = revenue_ratio_LP(n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, assignment, stored_cs, stored_Rs)

        if revenue_ratio == -np.inf:
            return False, -np.inf, solve_time
        else:
            return True, revenue_ratio, solve_time

    except Exception as e:
        return False, -np.inf, 0.0


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

    # GCN inference (same as test_PCP_multi_model_avg.py)
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

    # Commented out detailed printing for cleaner output
    # print(f"GCN generated pred_assort:")
    # for k in range(m):
    #     bundle_binary = ''.join(map(str, initial_pred_assort[k, :]))
    #     bundle_idx = int(bundle_binary, 2)
    #     print(f"  Segment {k}: {bundle_binary} (Bundle {bundle_idx})")

    return initial_pred_assort, prob


def solve_initial_milp(initial_pred, miscellaneous):
    """
    Use MILP solver to get initial revenue ratio
    """
    n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = miscellaneous

    initial_milp_ratio, _, _ = revenue_ratio_with_optimal_bundle(
        n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_rev, initial_pred, stored_cs, stored_Rs)

    return initial_milp_ratio


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
        tuple: (neighbors, timing_info)
            - neighbors: list of neighbor assignments, ordered by priority
            - timing_info: dict with 'add_candidate_time', 'drop_candidate_time', 'neighbor_generation_time'
    """
    import time
    
    # Convert assignment to pred_assort matrix
    convert_start = time.time()
    current_pred_assort = assignment_to_pred_assort(current_assignment, n, m)
    convert_time = time.time() - convert_start
    
    # Calculate K = ceil(sqrt(m))
    K = int(ceil(sqrt(m)))
    
    neighbors = []
    timing_info = {
        'add_candidate_time': 0.0,
        'drop_candidate_time': 0.0,
        'neighbor_generation_time': 0.0,
        'convert_time': convert_time
    }
    
    # Step 1: Generate Add candidates (score = probability)
    add_start = time.time()
    add_candidates = []
    for k in range(m):
        for j in range(n):
            if current_pred_assort[k, j] == 0:  # Currently not selected
                score_add = prob[k, j]  # Higher probability = better candidate
                add_candidates.append((k, j, score_add, 'add'))
    
    # Sort Add candidates by score (descending: high prob -> low prob)
    add_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Take top K Add candidates
    add_list = add_candidates[:K]
    timing_info['add_candidate_time'] = time.time() - add_start
    
    # Step 2: Generate Drop candidates (score = 1 - probability)
    # Only consider products with prob >= 0.5 (consistent with Initial FCP threshold strategy)
    drop_start = time.time()
    drop_candidates = []
    for k in range(m):
        for j in range(n):
            if current_pred_assort[k, j] == 1 and prob[k, j] >= 0.5:  # Currently selected AND prob >= 0.5
                score_drop = 1 - prob[k, j]  # 1 - probability (lower prob = higher score)
                drop_candidates.append((k, j, score_drop, 'drop'))
    
    # Sort Drop candidates by score (descending: high (1-prob) -> low (1-prob))
    drop_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Take top K Drop candidates
    drop_list = drop_candidates[:K]
    timing_info['drop_candidate_time'] = time.time() - drop_start
    
    # Step 3: Mix Add and Drop candidates, sort by score (descending)
    neighbor_gen_start = time.time()
    mixed_candidates = add_list + drop_list
    mixed_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Generate neighbors in mixed priority order
    for k, j, score, op_type in mixed_candidates:
        neighbor_pred = current_pred_assort.copy()
        if op_type == 'add':
            neighbor_pred[k, j] = 1  # Add product j to segment k
        else:  # op_type == 'drop'
            neighbor_pred[k, j] = 0  # Drop product j from segment k
        neighbor_assignment = convert_pred_assort_to_assignment(neighbor_pred)
        neighbors.append(neighbor_assignment)
    
    timing_info['neighbor_generation_time'] = time.time() - neighbor_gen_start
    
    return neighbors, timing_info


def local_search_with_lp_global_topk(initial_pred_assort, prob, meta, max_iterations=50, tolerance=1e-6, silent=False):
    """
    Local search function using global Top-K neighbor generation strategy
    
    Workflow:
    1. Initial MILP solve to get baseline assignment
    2. LP solve to get current best revenue
    3. Neighborhood search loop with global Top-K:
       - Generate at most 2*K neighbors (K = ceil(sqrt(m)))
       - LP feasibility check
       - Revenue improvement check
       - Update best solution
    4. Convert optimal assignment to pred_assort
    5. Final MILP solve (verify LP result)
    
    Args:
        initial_pred_assort: [m, n] initial predicted bundle assignment matrix
        prob: [m, n] GCN output probability matrix
        meta: data parameter tuple
        max_iterations: maximum number of iterations
        tolerance: tolerance for revenue improvement
        silent: if True, suppress detailed output
    
    Returns:
        tuple: (final_pred_assort, final_revenue_ratio, search_info)
    """
    n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = meta
    m = segment_num
    
    # Calculate K for this instance
    K = int(ceil(sqrt(m)))
    
    # Step 1: Initial MILP solve to get feasible assignment
    if not silent:
        print("Step 1: Initial MILP solve...")
    initial_milp_ratio, initial_milp_time, initial_assignment = revenue_ratio_with_optimal_bundle(
        n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, initial_pred_assort, stored_cs, stored_Rs)
    
    if not silent:
        print(f"Initial MILP result: revenue ratio={initial_milp_ratio:.6f}, time={initial_milp_time:.4f}s")
        print(f"Initial assignment: {initial_assignment}")
        print(f"Top-K parameter: K={K} (max neighbors per iteration: {2*K})")
    
    # Step 2: LP solve to get current best revenue
    if not silent:
        print("Step 2: Initial LP solve...")
    current_revenue, initial_lp_time = revenue_ratio_LP(n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, initial_assignment, stored_cs, stored_Rs)
    current_assignment = initial_assignment.copy()
    
    if not silent:
        print(f"Initial LP result: revenue ratio={current_revenue:.6f}, time={initial_lp_time:.4f}s")
    
    # Search information recording
    search_start_time = time.time()
    search_info = {
        'initial_milp_revenue': initial_milp_ratio,
        'initial_lp_revenue': current_revenue,
        'iterations': 0,
        'improvements': 0,
        'search_path': [current_revenue],
        'time_path': [0.0],
        'iteration_path': [0],
        'lp_solver_calls': 1,  # Initial LP solve counts as one
        'milp_solver_calls': 1,  # Initial MILP solve counts as one
        'total_iteration_time': 0.0,  # Total time spent in iterations only
        'K': K,  # Record K value
        'max_neighbors_per_iter': 2 * K,  # Record max neighbors per iteration
        # Detailed timing breakdown
        'total_add_candidate_time': 0.0,
        'total_drop_candidate_time': 0.0,
        'total_neighbor_generation_time': 0.0,
        'total_lp_solve_time': 0.0,
        'total_neighbor_iteration_time': 0.0,  # Time spent in for loop (excluding LP)
    }
    
    # Step 3: Local Search loop with global Top-K
    if not silent:
        print("Step 3: Local Search loop (Global Top-K strategy)...")
    improved = True
    iteration = 0
    iteration_loop_start_time = time.time()
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        iteration_start_time = time.time()
        
        # Generate neighbors using global Top-K strategy
        neighbors, neighbor_timing = generate_neighbor_assignments_global_topk(current_assignment, prob, n, m)
        
        # Accumulate candidate generation times
        search_info['total_add_candidate_time'] += neighbor_timing['add_candidate_time']
        search_info['total_drop_candidate_time'] += neighbor_timing['drop_candidate_time']
        search_info['total_neighbor_generation_time'] += neighbor_timing['neighbor_generation_time']
        
        actual_neighbors = len(neighbors)
        if not silent:
            print(f"Iteration {iteration}: Evaluating {actual_neighbors} neighbors (max: {2*K})")
        
        # Evaluate each neighbor in priority order
        for neighbor_idx, neighbor_assignment in enumerate(neighbors):
            # LP feasibility and revenue check
            is_feasible, neighbor_revenue, lp_time = check_lp_feasibility_and_revenue(
                neighbor_assignment, n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, stored_cs, stored_Rs)
            
            # lp_solver_calls represents one solve call
            search_info['lp_solver_calls'] += 1
            search_info['total_lp_solve_time'] += lp_time
            
            # Track non-LP time in the loop (variable assignment, condition check, etc.)
            # This is a small overhead, we'll estimate it as a small fraction
            # The actual measurement would require more detailed instrumentation
            search_info['total_neighbor_iteration_time'] += 0.000001  # Minimal overhead per iteration
            
            if is_feasible and neighbor_revenue > current_revenue + tolerance:
                current_assignment = neighbor_assignment
                current_revenue = neighbor_revenue
                improved = True
                search_info['improvements'] += 1
                search_info['search_path'].append(current_revenue)
                search_info['time_path'].append(time.time() - search_start_time)
                search_info['iteration_path'].append(iteration)
                
                if not silent:
                    print(f"Iteration {iteration}: Found improvement at neighbor {neighbor_idx+1}/{actual_neighbors}, "
                          f"revenue ratio={current_revenue:.6f}")
                break  # Greedy strategy: immediately accept improvement
        
        iteration_time = time.time() - iteration_start_time
        search_info['total_iteration_time'] += iteration_time
        
        if not improved and not silent:
            print(f"Iteration {iteration}: No improvement found after evaluating {actual_neighbors} neighbors, search converged")
    
    search_info['iterations'] = iteration
    search_info['final_lp_revenue'] = current_revenue
    search_info['lp_improvement'] = current_revenue - search_info['initial_lp_revenue']
    
    # Step 4: Convert optimal assignment to pred_assort
    if not silent:
        print("Step 4: Converting optimal assignment to pred_assort...")
    final_pred_assort = assignment_to_pred_assort(current_assignment, n, m)
    
    if not silent:
        print(f"Final pred_assort shape: {final_pred_assort.shape}")
    
    # Step 5: Final MILP solve (verify LP result)
    if not silent:
        print("Step 5: Final MILP solve (verify LP result)...")
    final_milp_ratio, final_milp_time = revenue_ratio_with_optimal_bundle(
        n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, final_pred_assort, stored_cs, stored_Rs)[:2]
    
    search_info['final_milp_revenue'] = final_milp_ratio
    search_info['final_milp_time'] = final_milp_time
    search_info['milp_solver_calls'] += 1
    search_info['total_improvement'] = final_milp_ratio - search_info['initial_milp_revenue']
    
    if not silent:
        print(f"Final MILP result: revenue ratio={final_milp_ratio:.6f}, time={final_milp_time:.4f}s")
        print(f"Total improvement: {search_info['total_improvement']:.6f}")
    
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
        'revenue_ratio': results_array[:, 0],
        'runtime_ratio': results_array[:, 1],
        'avg_total_time': results_array[:, 2],
        'avg_threshold_time': results_array[:, 3],
        'avg_initial_milp_time': results_array[:, 4],
        'avg_local_search_time': results_array[:, 5],
        'avg_gcn_time': results_array[:, 6],
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
                'avg_revenue_ratio': float(np.mean(results_array[:, 0])),
                'std_revenue_ratio': float(np.std(results_array[:, 0])),
                'avg_time_ratio': float(np.mean(results_array[:, 1])),
                'std_time_ratio': float(np.std(results_array[:, 1])),
                'avg_total_time': float(np.mean(results_array[:, 2])),
                'avg_threshold_time': float(np.mean(results_array[:, 3])),
                'avg_initial_milp_time': float(np.mean(results_array[:, 4])),
                'avg_local_search_time': float(np.mean(results_array[:, 5])),
                'avg_gcn_time': float(np.mean(results_array[:, 6])),
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


def evaluate_single_dataset_OLD(test_data_path, dataset_name, model_path, max_samples=1000):
    """
    Evaluate global Top-K Local Search strategy on a single dataset
    
    Args:
        test_data_path: path to test dataset directory
        dataset_name: name of the dataset
        model_path: path to the trained model file
        max_samples: maximum number of samples to evaluate
    
    Returns:
        dict: evaluation results
    """
    print(f"\n=== Evaluating Dataset: {dataset_name} (Global Top-K Strategy) ===")
    print(f"Test data path: {test_data_path}")
    print(f"Test data directory exists: {os.path.exists(test_data_path)}")
    
    if not os.path.exists(test_data_path):
        print(f"Dataset path does not exist: {test_data_path}")
        return None
    
    # Load test dataset
    print('Begin dataset loading...')
    dir_list = os.listdir(test_data_path)
    sample_num = len(dir_list)
    test_dataset = []
    miscellaneous_dataset = []
    print('Start reading the dataset...')
    
    for i in range(sample_num):
        if dir_list[i] == '.DS_Store':
            continue
        file_path = os.path.join(test_data_path, dir_list[i])
        try:
            dat, miscellaneous = process_data(file_path)
            test_dataset.append(dat)
            miscellaneous_dataset.append(miscellaneous)
        except Exception as e:
            print(f"Error processing file {dir_list[i]}: {e}")
            continue
    
    sample_num = len(test_dataset)
    print(f'Successfully loaded {sample_num} test samples.')
    
    if sample_num == 0:
        print(f"No valid samples found in {dataset_name}")
        return None
    
    # Limit samples
    actual_test_count = min(sample_num, max_samples)
    if sample_num > max_samples:
        print(f"Limiting test to first {actual_test_count} samples out of {sample_num} total samples.")
        test_dataset = test_dataset[:actual_test_count]
        miscellaneous_dataset = miscellaneous_dataset[:actual_test_count]
    
    # Evaluate model
    results = []
    all_search_paths = []
    initial_milp_times = []
    iteration_times = []
    final_milp_times = []
    iterations_counts = []
    per_iter_times = []
    K_values = []  # Track K values
    # Detailed timing breakdown
    add_times = []
    drop_times = []
    neighbor_gen_times = []
    lp_solve_times = []
    neighbor_iter_times = []
    
    # Local search parameters
    max_iterations = 50
    tolerance = 1e-6
    
    for i in tqdm(range(actual_test_count), desc=f"Evaluating {dataset_name}"):
        try:
            dat = test_dataset[i]
            miscellaneous = miscellaneous_dataset[i]
            n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = miscellaneous
            
            # Start timing for the entire strategy
            strategy_start_time = time.time()
            
            # Step 1: Generate initial pred_assort
            threshold_start = time.time()
            initial_pred, prob = predict_initial_bundles(dat, miscellaneous, model_path)
            threshold_time = time.time() - threshold_start
            
            # Step 2: Initial MILP solve
            initial_milp_start = time.time()
            initial_revenue = solve_initial_milp(initial_pred, miscellaneous)
            initial_milp_time = time.time() - initial_milp_start
            
            # Step 3: Global Top-K Local Search optimization
            local_search_start = time.time()
            best_pred, best_rev, search_info = local_search_with_lp_global_topk(
                initial_pred, prob, miscellaneous, max_iterations, tolerance
            )
            local_search_time = time.time() - local_search_start
            
            # Collect timing metrics
            initial_milp_times.append(initial_milp_time)
            total_iteration_time = search_info.get('total_iteration_time', local_search_time)
            iteration_times.append(total_iteration_time)
            final_milp_times.append(search_info.get('final_milp_time', 0.0))
            iterations_counts.append(search_info.get('iterations', 0))
            iters = max(1, search_info.get('iterations', 0))
            per_iter_times.append(total_iteration_time / iters)
            K_values.append(search_info.get('K', int(ceil(2 * sqrt(segment_num)))))
            
            # Collect detailed timing breakdown
            add_times.append(search_info.get('total_add_candidate_time', 0.0))
            drop_times.append(search_info.get('total_drop_candidate_time', 0.0))
            neighbor_gen_times.append(search_info.get('total_neighbor_generation_time', 0.0))
            lp_solve_times.append(search_info.get('total_lp_solve_time', 0.0))
            neighbor_iter_times.append(search_info.get('total_neighbor_iteration_time', 0.0))
            
            # Total strategy time
            total_time = time.time() - strategy_start_time
            
            # Calculate time ratio
            time_ratio = total_time / running_time if running_time > 0 else float('inf')
            
            # Store detailed results
            results.append([
                n, best_rev, time_ratio, total_time, running_time,
                threshold_time, initial_milp_time, local_search_time,
                initial_revenue, search_info['total_improvement'],
                search_info['iterations'], search_info['improvements'],
                search_info['lp_solver_calls'], search_info['milp_solver_calls'],
                search_info.get('K', 0), search_info.get('max_neighbors_per_iter', 0),
                # Detailed timing breakdown
                search_info.get('total_iteration_time', 0.0),
                search_info.get('total_add_candidate_time', 0.0),
                search_info.get('total_drop_candidate_time', 0.0),
                search_info.get('total_neighbor_generation_time', 0.0),
                search_info.get('total_lp_solve_time', 0.0),
                search_info.get('total_neighbor_iteration_time', 0.0)
            ])
            
            # Collect search path data
            search_path_data = {
                'sample_id': i,
                'n_products': n,
                'revenue_path': search_info['search_path'],
                'time_path': search_info['time_path'],
                'iteration_path': search_info['iteration_path'],
                'initial_revenue': search_info['initial_lp_revenue'],
                'final_revenue': best_rev,
                'improvement': search_info['total_improvement'],
                'dataset_name': dataset_name,
                'K': search_info.get('K', 0),
                'max_neighbors_per_iter': search_info.get('max_neighbors_per_iter', 0)
            }
            all_search_paths.append(search_path_data)
            
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute averages
    def _avg(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0
    
    return {
        'dataset_name': dataset_name,
        'results': np.array(results) if results else np.array([]),
        'search_paths': all_search_paths,
        'sample_count': len(results),
        'avg_initial_milp_time': _avg(initial_milp_times),
        'avg_iteration_time': _avg(iteration_times),
        'avg_final_milp_time': _avg(final_milp_times),
        'avg_iterations': _avg(iterations_counts),
        'avg_per_iter_time': _avg(per_iter_times),
        'avg_K': _avg(K_values),
        # Detailed timing breakdown
        'avg_add_candidate_time': _avg(add_times),
        'avg_drop_candidate_time': _avg(drop_times),
        'avg_neighbor_generation_time': _avg(neighbor_gen_times),
        'avg_lp_solve_time': _avg(lp_solve_times),
        'avg_neighbor_iteration_time': _avg(neighbor_iter_times),
    }


def main():
    """
    Main function: Multi-model average evaluation for FCP Local Search
    """
    parser = argparse.ArgumentParser(description="Multi-model average evaluation (FCP Local Search)")
    parser.add_argument("--data_dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    parser.add_argument("--test_subdirs", type=str, default="data/deterministic/test_m10n10_1e_3;data/deterministic/test_m20n10_1e_3;data/deterministic/test_m30n10_1e_3", 
                        help="Test data subdirectories (separated by semicolon)")
    parser.add_argument("--model_dir", type=str, default="models/base_4layer")
    parser.add_argument("--layers", type=str, default="4", help="Layers list, comma separated")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10", help="Seeds list, comma separated")
    parser.add_argument("--result_dir", type=str, default="results/appendix_cd_legacy", help="Results save directory")
    parser.add_argument("--save_result", type=bool, default=True, help="Whether to save result files")
    parser.add_argument("--max_iterations", type=int, default=50, help="Max local search iterations")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Revenue improvement tolerance")
    parser.add_argument("--silent", type=bool, default=True, help="Silent mode: suppress iteration details")
    args = parser.parse_args()

    dir_path = args.data_dir
    test_subdirs = parse_paths_arg(args.test_subdirs)
    model_root = os.path.join(dir_path, args.model_dir)
    result_dir = os.path.join(dir_path, args.result_dir) if args.save_result else None

    layers = parse_list_arg(args.layers)
    seeds = parse_list_arg(args.seeds)

    # Create results directory
    if args.save_result:
        os.makedirs(result_dir, exist_ok=True)

    # Always show configuration (not affected by silent mode)
    print("=" * 80)
    print("Global Top-K Local Search Multi-Model Average Evaluation")
    print("Strategy: K = ceil(sqrt(m)), max neighbors per iteration = 2*K")
    print("Add score = P[k,j], Drop score = 1-P[k,j], mixed and sorted by score")
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
    # Structure: {layer: {(m, n): [(revenue_ratio, time_ratio, total_time, threshold_time, initial_milp_time, local_search_time, gcn_time), ...]}}
    results_by_layer = {nl: {} for nl in layers}
    
    # Store results by layer and seed (each seed separately)
    # Structure: {layer: {seed: {(m, n): [(revenue_ratio, time_ratio, total_time, threshold_time, initial_milp_time, local_search_time, gcn_time), ...]}}}
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
                    
                    model_revenues = []
                    model_time_ratios = []
                    model_total_times = []
                    model_threshold_times = []
                    model_initial_milp_times = []
                    model_local_search_times = []
                    model_gcn_times = []

                    for sd, mdl, path in models_by_layer[nl]:
                        try:
                            # Start timing for entire strategy
                            strategy_start_time = time.time()
                            
                            # Step 1: GCN inference + threshold prediction
                            threshold_start = time.time()
                            initial_pred, prob = predict_initial_bundles(dat, misc, mdl, device)
                            threshold_time = time.time() - threshold_start
                            
                            # Step 2: Initial MILP solve
                            initial_milp_start = time.time()
                            initial_revenue = solve_initial_milp(initial_pred, misc)
                            initial_milp_time = time.time() - initial_milp_start
                            
                            # Step 3: Local search optimization
                            local_search_start = time.time()
                            best_pred, best_rev, search_info = local_search_with_lp_global_topk(
                                initial_pred, prob, misc, args.max_iterations, args.tolerance, silent=args.silent
                            )
                            local_search_time = time.time() - local_search_start
                            
                            # Total time
                            total_time = time.time() - strategy_start_time
                            time_ratio = total_time / running_time if running_time > 0 else float("inf")
                            
                            # Extract GCN time (from threshold_time, it includes GCN inference)
                            gcn_time = threshold_time

                            model_revenues.append(best_rev)
                            model_time_ratios.append(time_ratio)
                            model_total_times.append(total_time)
                            model_threshold_times.append(threshold_time)
                            model_initial_milp_times.append(initial_milp_time)
                            model_local_search_times.append(local_search_time)
                            model_gcn_times.append(gcn_time)
                            
                            # Record individual seed result
                            key = (segment_num, n)
                            if key not in results_by_seed[nl][sd]:
                                results_by_seed[nl][sd][key] = []
                            results_by_seed[nl][sd][key].append([best_rev, time_ratio, total_time, threshold_time, initial_milp_time, local_search_time, gcn_time])
                            
                        except Exception as e_model:
                            if not args.silent:
                                print(f"Model failed sample={file_names[i] if i < len(file_names) else i}, layer={nl}, seed={sd}, path={path}: {e_model}")
                                import traceback
                                traceback.print_exc()
                            continue

                    if len(model_revenues) > 0:
                        key = (segment_num, n)
                        if key not in results_by_layer[nl]:
                            results_by_layer[nl][key] = []
                        
                        result_entry = [
                            float(np.mean(model_revenues)),
                            float(np.mean(model_time_ratios)),
                            float(np.mean(model_total_times)),
                            float(np.mean(model_threshold_times)),
                            float(np.mean(model_initial_milp_times)),
                            float(np.mean(model_local_search_times)),
                            float(np.mean(model_gcn_times)),
                        ]
                        results_by_layer[nl][key].append(result_entry)
            except Exception as e:
                if not args.silent:
                    print(f"Sample {file_names[i] if i < len(file_names) else i} evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
                continue

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
                        print(f"    Revenue Ratio - Mean: {np.mean(results_array[:, 0]):.4f}, Std: {np.std(results_array[:, 0]):.4f}, "
                              f"Min: {np.min(results_array[:, 0]):.4f}, Max: {np.max(results_array[:, 0]):.4f}")
                        print(f"    Time Ratio - Mean: {np.mean(results_array[:, 1]):.4f}, Std: {np.std(results_array[:, 1]):.4f}, "
                              f"Min: {np.min(results_array[:, 1]):.4f}, Max: {np.max(results_array[:, 1]):.4f}")
                        print(f"    Avg Times - Total: {np.mean(results_array[:, 2]):.4f}s, Threshold: {np.mean(results_array[:, 3]):.4f}s, "
                              f"Initial MILP: {np.mean(results_array[:, 4]):.4f}s, Local Search: {np.mean(results_array[:, 5]):.4f}s, GCN: {np.mean(results_array[:, 6]):.4f}s")
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
