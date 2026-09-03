import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
import os
import numpy as np
import msgpack
import msgpack_numpy as mnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
from contextlib import nullcontext

import torch
from torch_geometric.nn import GENConv
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import gurobipy as gp
from gurobipy import GRB
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,edge_dim):
        super().__init__()
        self.lr_conv1 = GENConv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.lr_conv2 = GENConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.rl_conv1 = GENConv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.rl_conv2 = GENConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.lr_conv3 = GENConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.rl_conv3 = GENConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.FC = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.5) # 尝试改Dropout的数据（0.3 0.2）
        # 尝试加层数

    def forward(self, data):
        x, edge_index, edge_weight, side_ind = data.x, data.edge_index, data.edge_attr, data.side_ind   #side_ind: 0 for customer, 1 for product
        reverse_edge_index = edge_index.flip(dims=[0])
        
        x1 = self.lr_conv1(x, edge_index, edge_weight)
        x2 = self.rl_conv1(x, reverse_edge_index, edge_weight)
        x = F.relu((1-side_ind)*x1 + side_ind*x2)
        x = self.dropout(x)
        
        x1 = self.lr_conv2(x, edge_index, edge_weight)
        x2 = self.rl_conv2(x, reverse_edge_index, edge_weight)
        x = F.relu((1-side_ind)*x1 + side_ind*x2)
        x = self.dropout(x)
        
        # Third GENConv layer
        x1 = self.lr_conv3(x, edge_index, edge_weight)
        x2 = self.rl_conv3(x, reverse_edge_index, edge_weight)
        x = F.relu((1-side_ind)*x1 + side_ind*x2)
        x = self.dropout(x)
        
        x = self.FC(x)
        return x
    
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


# 修复PyTorch 2.6的weights_only问题 - 添加安全全局变量
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([EdgeScoringGCN])
    
def process_data(file_path):
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
    
    # Note: cs and Rs will be calculated on-demand for predicted bundles only
    # This saves memory and computation for large n values
    cs = None  # Will be calculated in revenue_ratio function
    Rs = None  # Will be calculated in revenue_ratio function

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

    # Build bipartite edges: product i -> segment j
    prods = []
    custs = []
    edge_weights = []
    for i in range(product_num):
        for j in range(segment_num):
            prods.append(i)
            custs.append(j + product_num)
            edge_weights.append([float(unit_us[j, i])])

    edge_index = torch.tensor([prods, custs], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    side_ind = torch.tensor([1] * product_num + [0] * segment_num, dtype=torch.long).view(-1, 1)

    # Labels (unused in evaluation)
    prod_labels = np.array(opt_bundles).T
    seg_labels = -np.ones((segment_num, segment_num), dtype=int)  # placeholder
    y = np.append(prod_labels, seg_labels, axis=0)
    y = torch.tensor(y, dtype=torch.long)

    # Include product/segment counts to help edge models reshape logits
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        side_ind=side_ind,
        y=y,
        product_num=product_num,
        segment_num=segment_num,
    )

    miscellaneous = (
        product_num,
        segment_num,
        unit_cs,
        ship_cs,
        unit_us,
        Ns,
        opt_bundles,
        opt_prices,
        opt_rev,
        running_time,
        gap,
    )
    return data, miscellaneous


def load_web_gcn_results(results_file_path):
    """
    Load GCN inference results from Web-based computation

    Args:
        results_file_path: Path to the msgpack file containing Web GCN results

    Returns:
        dict: Dictionary mapping file names to their results
    """
    # Determine file format based on extension
    if results_file_path.endswith('.json'):
        # JSON format (legacy support)
        with open(results_file_path, 'r') as f:
            web_results = json.load(f)
    else:
        # msgpack format (default)
        with open(results_file_path, 'rb') as f:
            web_results = msgpack.load(f, object_hook=mnp.decode, strict_map_key=False)

    # Create a mapping from file names to results for easy lookup
    results_map = {}
    for result in web_results['results']:
        file_name = result['file_name']
        results_map[file_name] = {
            'pred_assort': np.array(result['pred_assort']),
            'gcn_inference_time': result['gcn_inference_time'],
            'n_products': result['n_products']
        }

    return results_map, web_results


# def stable_sigmoid(x):
#     """
#     Numerically stable sigmoid function to avoid overflow

#     Args:
#         x: Input array

#     Returns:
#         Sigmoid output without numerical overflow
#     """
#     # Clip extreme values to prevent overflow
#     x_clipped = np.clip(x, -500, 500)

#     # Use different formulations for positive and negative values
#     # For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
#     # For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
#     return np.where(x_clipped >= 0,
#                     1 / (1 + np.exp(-x_clipped)),
#                     np.exp(x_clipped) / (1 + np.exp(x_clipped)))


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


def revenue_ratio(n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, pred_assort, time_limit=600):
    """Optimized approach: only create variables and constraints for predicted bundles"""
    milp_start_time = time.time()

    segment_ind = np.array([i for i in range(m)])
    
    # Get unique predicted bundles
    # pred_assort (预测的产品组合矩阵)并生成对应可行bundle
    bundle_dic = {}
    for i in range(m):
        bundle_idx = binary_vector_to_bundle_id(pred_assort[i, :])
        try:
            bundle_dic[bundle_idx].append(i)
        except:
            bundle_dic[bundle_idx] = [i]
    
    predicted_bundles = list(bundle_dic.keys())
    originally_predicted = set(predicted_bundles)  # 记录原始预测的 bundles
    
    # Always include bundle 0 (empty bundle) to ensure feasibility
    # This provides a fallback solution where everyone chooses nothing
    if 0 not in predicted_bundles:
        predicted_bundles.append(0)
    
    
    # Calculate Rs and cs only for predicted bundles
    # Generate assortments only for predicted bundles
    predicted_assortments = bundle_ids_to_assortment_matrix(predicted_bundles, n)
    
    # Calculate Rs only for predicted bundles: Rs[customer, bundle]
    Rs_predicted = np.log1p(unit_us.dot(predicted_assortments.T))  # shape: (m, len(predicted_bundles))
    
    # Calculate cs only for predicted bundles: cs[customer, bundle]  
    cs_base = np.sum(predicted_assortments * unit_cs, axis=1)  # shape: (len(predicted_bundles),)
    cs_predicted = cs_base + ship_cs  # Broadcasting: (len(predicted_bundles),) + (m, 1) -> (m, len(predicted_bundles))
    cs_predicted = cs_predicted * 0.2
    
    # Create mapping from bundle_id to index in predicted arrays
    bundle_to_idx = {bundle_id: idx for idx, bundle_id in enumerate(predicted_bundles)}
    
    Rbar = np.max(Rs_predicted)

    model = gp.Model("Bundle MILP Optimized")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-3
    model.Params.TimeLimit = time_limit
    model.Params.DualReductions = 0  # 强制 Gurobi 区分不可行和无界
    
    # For debugging infeasibility - can compute IIS (Irreducible Inconsistent Subsystem)
    # model.Params.OutputFlag = 1  # Uncomment to see detailed output

    # Create variables ONLY for predicted bundles
    p = model.addVars(predicted_bundles, vtype=GRB.CONTINUOUS, lb=0, name="p")
    theta = model.addVars(m, predicted_bundles, vtype=GRB.BINARY, name="theta")
    s = model.addVars(m, vtype=GRB.CONTINUOUS, name="s")
    S = model.addVars(m, predicted_bundles, vtype=GRB.CONTINUOUS, lb=0, name='S')
    Z = model.addVars(m, predicted_bundles, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Z')
    P = model.addVars(m, predicted_bundles, vtype=GRB.CONTINUOUS, lb=0, name='P')

    # Objective: only consider predicted bundles
    model.setObjective(gp.quicksum(Ns[k, 0]*Z[k, i] for k in segment_ind for i in predicted_bundles), GRB.MAXIMIZE)

    # Standard constraints (only for predicted bundles)
    model.addConstrs((s[k] >= Rs_predicted[k, bundle_to_idx[i]] - p[i] for i in predicted_bundles for k in segment_ind))
    
    #     # 改进的次可加性约束
    # # 只对原始预测的 bundles 添加次可加性约束，不包括手动添加的 bundle 0
    # # 预计算所有bundle的产品集合
    # bundle_product_sets = {}
    # for bundle_id in originally_predicted:  # 只对原始预测的 bundles
    #     bundle_product_sets[bundle_id] = bundle_to_product_set(bundle_id, n)

    # # 筛选约束
    # added_constraints = set()  # 避免重复约束

    # for k in originally_predicted:  # 只对原始预测的 bundles
    #     if k == 0:  # 跳过空集
    #         continue

    #     k_set = bundle_product_sets[k]
    #     if len(k_set) == 0:
    #         continue

    #     for i in originally_predicted:  # 只对原始预测的 bundles
    #         for j in originally_predicted:  # 只对原始预测的 bundles
    #             if i >= j:  # 避免重复
    
    # K-way cover subadditivity constraints (free disposal, minimal covers only).
    # This implementation stays exact but avoids Python set operations and avoids
    # enumerating all bundle subsets before checking whether they can cover k.
    predicted_nonempty = [b for b in originally_predicted if b != 0]
    added_constraints = set()
    subadd_ctr = 0
    item_bits = [1 << bit_idx for bit_idx in range(n)]

    def add_subadditivity_constraint(target_bundle, cover):
        nonlocal subadd_ctr
        ordered_cover = tuple(sorted(cover))
        key = (target_bundle, ordered_cover)
        if key in added_constraints:
            return
        model.addConstr(
            p[target_bundle] <= gp.quicksum(p[b] for b in ordered_cover),
            name=f"sa{subadd_ctr}"
        )
        added_constraints.add(key)
        subadd_ctr += 1

    for k in predicted_nonempty:
        target_mask = k
        singleton_covers = []
        multi_cover_candidates = []

        for b in predicted_nonempty:
            if b == k:
                continue
            contribution = b & target_mask
            if contribution == 0:
                continue
            if contribution == target_mask:
                singleton_covers.append(b)
            else:
                multi_cover_candidates.append((b, contribution))

        for b in singleton_covers:
            add_subadditivity_constraint(k, (b,))

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
                add_subadditivity_constraint(k, tuple(bundle_id for bundle_id, _ in chosen))
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
                    
    # Remaining constraints (only for predicted bundles)
    model.addConstrs((P[k, i] >= p[i] - Rbar*(1-theta[k, i]) for i in predicted_bundles for k in segment_ind))
    model.addConstrs((P[k, i] <= p[i] for i in predicted_bundles for k in segment_ind))
    
    # Modified constraint: only sum over predicted bundles
    model.addConstrs((s[k] >= gp.quicksum(Rs_predicted[k, bundle_to_idx[i]]*theta[j, i] - P[j, i] for i in predicted_bundles) for k in segment_ind for j in segment_ind))
    
    model.addConstrs((Z[k, i] == P[k, i] - cs_predicted[k, bundle_to_idx[i]] * theta[k, i] for i in predicted_bundles for k in segment_ind))
    model.addConstrs((S[k, i] == Rs_predicted[k, bundle_to_idx[i]]*theta[k, i] - P[k, i] for i in predicted_bundles for k in segment_ind))
    model.addConstrs((s[k] == gp.quicksum(S[k, i] for i in predicted_bundles) for k in segment_ind))
    model.addConstrs((gp.quicksum(theta[k, i] for i in predicted_bundles) == 1 for k in segment_ind))
    
    # Special handling for empty bundle (bundle 0) if it exists in predicted bundles
    if 0 in predicted_bundles:
        model.addConstr(p[0] == 0)  # Empty bundle must have zero price
        model.addConstrs((S[k, 0] == 0 for k in segment_ind))

    model.optimize()

    milp_end_time = time.time()
    milp_time = milp_end_time - milp_start_time

    # Check if optimization was successful
    if model.Status == GRB.OPTIMAL:
        return model.ObjVal/opt_rev, milp_time
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        # Time limit reached but at least one feasible solution found
        return model.ObjVal/opt_rev, milp_time
    elif model.Status == GRB.INFEASIBLE:
        # Model is infeasible - try to diagnose
        # Uncomment below to compute IIS for debugging
        # print(f"Model infeasible! Computing IIS...")
        # model.computeIIS()
        # model.write("infeasible_model.ilp")
        # print(f"IIS written to infeasible_model.ilp")
        raise ValueError(f"Optimization failed with status {model.Status} (INFEASIBLE)")
    else:
        # Optimization failed (unbounded or other status)
        raise ValueError(f"Optimization failed with status {model.Status}")


def main(use_web_gcn=False, web_results_path=None):
    """
    Main function with option to use Web-based GCN results

    Args:
        use_web_gcn: If True, use pre-computed Web GCN results instead of local inference
        web_results_path: Path to the Web GCN results JSON file
    """
    # Import config to get paths
    import sys
    import os

    # Add parent directory to path to import config
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)

    print(f"Parent directory: {parent_dir}")
    print(f"Python path: {sys.path[-1]}")

    # Configuration paths - using hardcoded paths since config.py doesn't exist
    # from config import BASE_DIR, MODEL_PATH, DATASET_PATHS

    # Set paths using hardcoded values
    # dir_path = BASE_DIR
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_path = os.path.join(dir_path, "models", "base_4layer", "best_model_edge_4layer_seed1.pt")
    # test_data_path = DATASET_PATHS.get('test_n10', os.path.join(BASE_DIR, "dataset_bundle/dataset/test/test_DLY/"))
    # test_data_path = dir_path + '/dataset/test_m10n10_beta_half_half/'
    test_data_path = os.path.join(dir_path, "data", "ood", "test_m10n10_log_correct_1e_3")
    
    
    # test_data_path = dir_path + '/m20_n10_sample_100/'
    # test_data_path = dir_path + '/m30_n10_sample_100/'
    # test_data_path = dir_path + '/m10_n15_sample_100_BSP/'
    # test_data_path = dir_path + '/test-BSP-m20n15-0.2-nolim/'
    # test_data_path = dir_path + '/test-BSP-m10n25-0.2-nolim/'
    result_path = os.path.join(dir_path, 'test_result_FCP_m10n10.csv')
    
    # Set default web results path if not provided
    if web_results_path is None:
        web_results_path = os.path.join(dir_path, 'gcn_inference_results.msgpack')

    # Debug: Print paths                                                    
    print(f"Base directory: {dir_path}")
    print(f"Model path: {model_path}")
    print(f"Test data path: {test_data_path}")
    print(f"Use Web GCN: {use_web_gcn}")
    if use_web_gcn:
        print(f"Web results path: {web_results_path}")

    # Check if files exist
    print(f"Model file exists: {os.path.exists(model_path)}")
    print(f"Test data directory exists: {os.path.exists(test_data_path)}")
    if use_web_gcn:
        print(f"Web results file exists: {os.path.exists(web_results_path)}")

    # Load Web GCN results if using web mode
    web_gcn_results = None
    web_gcn_metadata = None
    if use_web_gcn:
        print('Loading Web GCN results...')
        web_gcn_results, web_gcn_metadata = load_web_gcn_results(web_results_path)
        print(f'Loaded {len(web_gcn_results)} Web GCN results')

    # Load the trained model (only needed for local inference)
    model = None
    device = None
    if not use_web_gcn:
        print('Loading trained model...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        import __main__
        __main__.EdgeScoringGCN = EdgeScoringGCN
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
    
    # Load test dataset
    print('\nBegin model evaluation...')
    dir_list = os.listdir(test_data_path)
    sample_num = len(dir_list)
    test_dataset = []
    miscellaneous_dataset = []
    file_names = []
    print('Start reading the dataset...')

    for i in range(sample_num):
        if dir_list[i] == '.DS_Store':
            continue
        file_path = test_data_path + dir_list[i]
        try:
            dat, miscellaneous = process_data(file_path)
            test_dataset.append(dat)
            miscellaneous_dataset.append(miscellaneous)
            file_names.append(dir_list[i])
        except Exception as e:
            print(f"Error processing file {dir_list[i]}: {e}")
            continue

    sample_num = len(test_dataset)
    print(f'Successfully loaded {sample_num} test samples.')
    if sample_num == 0:
        print('No valid samples found. Exiting without evaluation.')
        return

    # Evaluate model
    ratios = []
    time_ratios = []

    if not use_web_gcn:
        model.eval()

    with torch.no_grad() if not use_web_gcn else nullcontext():
        for i in tqdm(range(sample_num), desc="Evaluating"):
            try:
                miscellaneous = miscellaneous_dataset[i]
                n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap = miscellaneous
                current_file_name = file_names[i]

                if use_web_gcn:
                    # Use Web GCN results
                    if current_file_name not in web_gcn_results:
                        print(f"Warning: No Web GCN result found for {current_file_name}, skipping...")
                        continue

                    web_result = web_gcn_results[current_file_name]
                    pred_assort = web_result['pred_assort']
                    gcn_inference_time = web_result['gcn_inference_time']

                    # Start timing from MILP (since GCN is already done)
                    milp_start_time = time.time()

                    # MILP solving (timed internally)
                    ratio, milp_time = revenue_ratio(n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_rev, pred_assort)

                    # Total time is GCN time + MILP time
                    total_time = gcn_inference_time + milp_time

                else:
                    # Local GCN inference
                    dat = test_dataset[i].to(device)

                    # Start timing from GCN inference
                    gcn_start_time = time.time()

                    # GCN inference (supports both node-level and edge-scoring models)
                    raw_out = model(dat)

                    if isinstance(raw_out, dict):
                        # EdgeScoringGCN-style output
                        if 'logit_matrix' in raw_out:
                            # shape (n, m)
                            logits_nm = raw_out['logit_matrix'].detach().cpu().numpy()
                        elif 'edge_logits' in raw_out:
                            s = raw_out['edge_logits'].detach().cpu().numpy()
                            logits_nm = s.reshape(n, segment_num)
                        else:
                            raise ValueError('Unexpected model output keys for edge scoring: ' + ','.join(raw_out.keys()))

                        # Convert logits to binary assortment per segment: shape (m, n)
                        pred_assort = (logits_nm.T >= 0.0).astype(int)
                    else:
                        # Legacy node-output model: keep original thresholding logic
                        output = raw_out[:n, :].detach().cpu().numpy()
                        pred_assort = (np.exp(output) / (np.exp(output) + np.exp(1)) >= 0.5).astype(int).T

                    # MILP solving (also timed internally)
                    ratio, milp_time = revenue_ratio(n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_rev, pred_assort)

                    # Total time from GCN start to MILP end
                    total_time = time.time() - gcn_start_time

                # Calculate time ratio with respect to default running time
                time_ratio = total_time / running_time if running_time > 0 else float('inf')

                ratios.append([n, ratio])
                time_ratios.append([n, time_ratio, total_time, running_time, milp_time])

            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                continue
    
    # Save results
    ratios = np.array(ratios)
    time_ratios = np.array(time_ratios)
    
    # Combine revenue and time ratios into one array
    if len(ratios) == 0:
        print('No successful evaluations to save.')
        return
    combined_results = np.column_stack((
        ratios[:, 0],  # n_products
        ratios[:, 1],  # revenue_ratio
        time_ratios[:, 1]  # runtime_ratio
    ))
    
    # Save with proper headers
    header = 'n_products,revenue_ratio,runtime_ratio'
    # np.savetxt(result_path, combined_results, delimiter=',', header=header, comments='')
    
    # Print summary statistics
    print(f'\nEvaluation completed!')
    print(f'Results saved to: {result_path}')
    print(f'Number of samples evaluated: {len(ratios)}')
    
    # Revenue ratio statistics
    if len(ratios) > 0:
        print(f'\n=== REVENUE RATIO STATISTICS ===')
        print(f'Average revenue ratio: {np.mean(ratios[:, 1]):.4f}')
        print(f'Std revenue ratio: {np.std(ratios[:, 1]):.4f}')
        print(f'Min revenue ratio: {np.min(ratios[:, 1]):.4f}')
        print(f'Max revenue ratio: {np.max(ratios[:, 1]):.4f}')
    
    # Time ratio statistics
    if len(time_ratios) > 0:
        print(f'\n=== TIME RATIO STATISTICS ===')
        print(f'Average time ratio (GCN+MILP/Default): {np.mean(time_ratios[:, 1]):.4f}')
        print(f'Std time ratio: {np.std(time_ratios[:, 1]):.4f}')
        print(f'Min time ratio: {np.min(time_ratios[:, 1]):.4f}')
        print(f'Max time ratio: {np.max(time_ratios[:, 1]):.4f}')
        
        print(f'\n=== ABSOLUTE TIME STATISTICS ===')
        print(f'Average total time (GCN+MILP): {np.mean(time_ratios[:, 2]):.4f} seconds')
        print(f'Average default time: {np.mean(time_ratios[:, 3]):.4f} seconds')
        print(f'Average MILP time: {np.mean(time_ratios[:, 4]):.4f} seconds')
        print(f'Average GCN time: {np.mean(time_ratios[:, 2] - time_ratios[:, 4]):.4f} seconds')


def main_with_local_gcn():
    """Run evaluation with local GCN inference"""
    main(use_web_gcn=False)


def main_with_web_gcn(web_results_path=None):
    """Run evaluation with Web-based GCN results"""
    main(use_web_gcn=True, web_results_path=web_results_path)


if __name__ == "__main__":
    import sys

    # Check command line arguments to determine mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "--web-gcn":
            # Use Web GCN mode
            web_path = sys.argv[2] if len(sys.argv) > 2 else None
            print("Running with Web-based GCN results...")
            main_with_web_gcn(web_path)
        elif sys.argv[1] == "--local-gcn":
            # Use local GCN mode
            print("Running with local GCN inference...")
            main_with_local_gcn()
        else:
            print("Usage:")
            print("  python test_FCP.py --local-gcn")
            print("  python test_FCP.py --web-gcn [path_to_results.msgpack]")
    else:
        # Default to local GCN mode
        print("Running with local GCN inference (default mode)...")
        main_with_local_gcn()
