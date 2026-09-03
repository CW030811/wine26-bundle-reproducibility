from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
import os
import numpy as np
import msgpack
import msgpack_numpy as mnp
from tqdm import tqdm
import time
import sys

from torch_geometric.nn import GENConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import gurobipy as gp
from gurobipy import GRB


class EdgeScoringGCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 128,
        num_layers: int = 2,
        edge_dim: int = 1,
        score_type: str = 'bilinear',  # 'bilinear' | 'dot' | 'mlp'
        proj_dim: int | None = None,   # for 'dot'
        use_edge_attr: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.score_type = score_type

        # Directional GENConv stacks: left->right and right->left
        self.l2r = nn.ModuleList()
        self.r2l = nn.ModuleList()
        c_in = in_channels
        for _ in range(num_layers):
            self.l2r.append(GENConv(c_in, hidden_channels, edge_dim=edge_dim))
            self.r2l.append(GENConv(c_in, hidden_channels, edge_dim=edge_dim))
            c_in = hidden_channels

        self.dropout = nn.Dropout(dropout)

        # Edge-scoring head
        if score_type == 'bilinear':
            self.U = nn.Parameter(torch.empty(hidden_channels, hidden_channels))
            nn.init.xavier_uniform_(self.U)
        elif score_type == 'dot':
            d = proj_dim or hidden_channels
            self.proj_p = nn.Linear(hidden_channels, d, bias=False)
            self.proj_s = nn.Linear(hidden_channels, d, bias=False)
        elif score_type == 'mlp':
            d_in = hidden_channels * 2 + (edge_dim if use_edge_attr else 0)
            self.mlp = nn.Sequential(
                nn.Linear(d_in, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            )
        else:
            raise ValueError(f"Unknown score_type={score_type}")

        if use_edge_attr and score_type in ('bilinear', 'dot'):
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            )

    def forward(self, data):
        x, edge_index, edge_attr, side_ind = data.x, data.edge_index, data.edge_attr, data.side_ind
        rev_edge_index = edge_index.flip(dims=[0])

        # Encoder: blend left-to-right and right-to-left messages
        for l2r_conv, r2l_conv in zip(self.l2r, self.r2l):
            r_x = l2r_conv(x, edge_index, edge_attr)
            l_x = r2l_conv(x, rev_edge_index, edge_attr)
            x = F.relu(r_x * (1 - side_ind) + l_x * side_ind)
            x = self.dropout(x)

        z = x  # node embeddings, shape (N, H)
        src, dst = edge_index  # edge endpoints

        # Edge scoring
        if self.score_type == 'bilinear':
            # s_e = h_src^T U h_dst
            s = torch.einsum('ei,ij,ej->e', z[src], self.U, z[dst])
            if self.use_edge_attr:
                s = s + self.edge_mlp(edge_attr).squeeze(-1)
        elif self.score_type == 'dot':
            s = (self.proj_p(z[src]) * self.proj_s(z[dst])).sum(dim=-1)
            if self.use_edge_attr:
                s = s + self.edge_mlp(edge_attr).squeeze(-1)
        else:  # 'mlp'
            feats = [z[src], z[dst]]
            if self.use_edge_attr:
                feats.append(edge_attr)
            s = self.mlp(torch.cat(feats, dim=-1)).squeeze(-1)

        out = { 'edge_logits': s }

        # Optional matrix view for single-graph inference
        if hasattr(data, 'product_num') and hasattr(data, 'segment_num'):
            try:
                n = int(data.product_num)
                m = int(data.segment_num)
                if s.numel() == n * m:
                    out['logit_matrix'] = s.view(n, m)
            except Exception:
                pass

        return out

# Allow torch.load to resolve EdgeScoringGCN safely (PyTorch 2.6+)
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([EdgeScoringGCN])
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

def bundle_to_product_set(bundle_id, n):
    """Convert bundle ID to product set"""
    binary_str = format(bundle_id, '0' + str(n) + 'b')
    return set(i for i, bit in enumerate(binary_str) if bit == '1')

def revenue_ratio(n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, feasible_bundles, selected_products=None, stored_cs=None, stored_Rs=None):
    """
    Optimized MILP: Use progressive strategy with chain structure to reduce subadditivity constraints
    Only calculates Rs and costs for feasible bundles to improve efficiency

    Args:
        n: number of products
        m: number of segments
        unit_cs: unit costs [1, n]
        ship_cs: shipping costs [m, 1]
        unit_us: unit utilities [m, n]
        Ns: demand matrix
        opt_rev: optimal revenue
        feasible_bundles: feasible bundle set
        selected_products: product selection list for each segment (used to build chain constraints)
        stored_cs: stored cost matrix (old format compatibility)
        stored_Rs: stored valuation matrix (old format compatibility)
    """

    segment_ind = np.arange(m)

    # Convert feasible_bundles to list for Gurobi
    feasible_bundles_list = list(feasible_bundles)
    
    # Calculate Rs and costs only for feasible bundles
    if stored_cs is not None and stored_Rs is not None:
        # Use stored matrices (old format)
        costs = stored_cs
        Rs = stored_Rs
        Rbar = np.max(Rs)
    else:
        # Calculate only for feasible bundles (new optimized format)
        # Only calculate for feasible bundles instead of all bundles
        
        # Generate assortments only for feasible bundles
        feasible_assortments = []
        for bundle_id in feasible_bundles_list:
            bundle_binary = list(map(int, format(bundle_id, '0' + str(n) + 'b')))
            feasible_assortments.append(bundle_binary)
        feasible_assortments = np.array(feasible_assortments)
        
        # Calculate Rs only for feasible bundles: Rs[customer, bundle]
        Rs_feasible = np.sqrt(unit_us.dot(feasible_assortments.T))  # shape: (m, len(feasible_bundles))
        
        # Calculate cs only for feasible bundles: cs[customer, bundle]  
        cs_base = np.sum(feasible_assortments * unit_cs, axis=1)  # shape: (len(feasible_bundles),)
        cs_feasible = cs_base + ship_cs  # Broadcasting: (len(feasible_bundles),) + (m, 1) -> (m, len(feasible_bundles))
        cs_feasible = cs_feasible * 0.2
        
        # Create mapping from bundle_id to index in feasible arrays
        bundle_to_idx = {bundle_id: idx for idx, bundle_id in enumerate(feasible_bundles_list)}
        
        # For legacy compatibility, create full-size arrays (this could be optimized further)
        costs = np.zeros((m, 2**n))
        Rs = np.zeros((m, 2**n))
        
        for bundle_id, idx in bundle_to_idx.items():
            costs[:, bundle_id] = cs_feasible[:, idx]
            Rs[:, bundle_id] = Rs_feasible[:, idx]
        
        Rbar = np.max(Rs_feasible)

    model = gp.Model('optimized_MILP_v2')
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-2
    model.Params.TimeLimit = 600

    # Create variables ONLY for feasible bundles
    p = model.addVars(feasible_bundles_list, lb=0.0, vtype=GRB.CONTINUOUS, name='p')
    theta = model.addVars(m, feasible_bundles_list, vtype=GRB.BINARY, name='theta')
    s = model.addVars(m, vtype=GRB.CONTINUOUS, name='s')
    S = model.addVars(m, feasible_bundles_list, lb=0.0, vtype=GRB.CONTINUOUS, name='S')
    Z = model.addVars(m, feasible_bundles_list, vtype=GRB.CONTINUOUS, name='Z')
    P = model.addVars(m, feasible_bundles_list, lb=0.0, vtype=GRB.CONTINUOUS, name='P')

    # Objective: only consider feasible bundles
    model.setObjective(gp.quicksum(Ns[k, 0] * Z[k, i] for k in segment_ind for i in feasible_bundles_list), GRB.MAXIMIZE)

    # Constraints (only for feasible bundles)
    model.addConstrs((s[k] >= Rs[k, i] - p[i] for i in feasible_bundles_list for k in segment_ind))

    # Optimized subadditivity constraint construction
    if selected_products is not None:
        # Method 1: Use progressive chain structure
        _add_progressive_chain_constraints(model, p, selected_products, n, feasible_bundles_list)

        # Method 2: Traditional cross-segment constraints (keep original logic but reduce computation)
        _add_cross_segment_constraints(model, p, feasible_bundles_list, n)
    else:
        # Fall back to original method
        _add_traditional_subadditivity_constraints(model, p, feasible_bundles_list, n)

    # Remaining constraints (only for feasible bundles)
    model.addConstrs((P[k, i] >= p[i] - Rbar * (1 - theta[k, i]) for i in feasible_bundles_list for k in segment_ind))
    model.addConstrs((P[k, i] <= p[i] for i in feasible_bundles_list for k in segment_ind))
    model.addConstrs((s[k] >= gp.quicksum(Rs[k, i] * theta[j, i] - P[j, i] for i in feasible_bundles_list) for k in segment_ind for j in segment_ind))
    model.addConstrs((Z[k, i] == P[k, i] - costs[k, i] * theta[k, i] for i in feasible_bundles_list for k in segment_ind))
    model.addConstrs((S[k, i] == Rs[k, i] * theta[k, i] - P[k, i] for i in feasible_bundles_list for k in segment_ind))
    model.addConstrs((s[k] == gp.quicksum(S[k, i] for i in feasible_bundles_list) for k in segment_ind))
    model.addConstrs((gp.quicksum(theta[k, i] for i in feasible_bundles_list) == 1 for k in segment_ind))

    if 0 in feasible_bundles:
        model.addConstrs((S[k, 0] == 0 for k in segment_ind))

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.ObjVal / opt_rev, model.Runtime
    elif model.status == GRB.TIME_LIMIT:
        return model.ObjBound / opt_rev, model.Runtime
    else:
        print(f"Optimized approach v2 - Optimization failed with status: {model.status}")
        return 0, model.Runtime


def _add_progressive_chain_constraints(model, p, selected_products, n, feasible_bundles_list):
    """
    Add monotonic constraints for progressive chains within each segment
    p({p1}) <= p({p1,p2}) <= p({p1,p2,p3}) <= ...
    """
    bin2num = lambda x: int(''.join(map(str, x.tolist())), 2)

    for segment_idx, selected_list in enumerate(selected_products):
        if len(selected_list) <= 1:
            continue

        # Build progressive bundles for this segment
        progressive_bundles = []
        for i in range(1, len(selected_list) + 1):
            bundle_array = np.zeros(n, dtype=int)
            for product_idx in selected_list[:i]:
                bundle_array[product_idx] = 1
            bundle_idx = bin2num(bundle_array)
            if bundle_idx in feasible_bundles_list:
                progressive_bundles.append(bundle_idx)

        # Add chain monotonic constraints: p[bundle_i] <= p[bundle_{i+1}]
        for i in range(len(progressive_bundles) - 1):
            current_bundle = progressive_bundles[i]
            next_bundle = progressive_bundles[i + 1]
            model.addConstr(p[current_bundle] <= p[next_bundle],
                          name=f"progressive_chain_s{segment_idx}_b{i}")


def _add_cross_segment_constraints(model, p, feasible_bundles_list, n):
    """
    Add cross-segment subadditivity constraints with more efficient filtering strategy
    """
    # Pre-compute product sets for all bundles
    bundle_product_sets = {}
    for bundle_id in feasible_bundles_list:
        bundle_product_sets[bundle_id] = bundle_to_product_set(bundle_id, n)

    # Only add constraints for different "families" of bundles, avoid redundant constraints within the same progressive chain
    added_constraints = set()

    for k in feasible_bundles_list:
        if k == 0:  # Skip empty set
            continue

        k_set = bundle_product_sets[k]
        if len(k_set) == 0:
            continue

        for i in feasible_bundles_list:
            for j in feasible_bundles_list:
                if i >= j:  # Avoid duplicates
                    continue

                i_set = bundle_product_sets[i]
                j_set = bundle_product_sets[j]
                union_set = i_set.union(j_set)

                # Check if it's a valid inclusion relationship
                if (k_set.issubset(union_set) and
                    k_set != i_set and
                    k_set != j_set and
                    not _is_progressive_redundant(k_set, i_set, j_set)):  # New: avoid progressive internal redundancy

                    constraint_key = (k, min(i, j), max(i, j))
                    if constraint_key not in added_constraints:
                        model.addConstr(p[k] <= p[i] + p[j],
                                      name=f"cross_segment_{k}_{i}_{j}")
                        added_constraints.add(constraint_key)


def _is_progressive_redundant(k_set, i_set, j_set):
    """
    Check if the constraint is redundant within the progressive chain
    If k, i, j all come from the same progressive chain and there are already chain constraints, then this constraint is redundant
    """
    # Simplified version: if k is a proper subset of i or j, it might be a progressive internal constraint
    return (k_set.issubset(i_set) and k_set != i_set) or (k_set.issubset(j_set) and k_set != j_set)


def _add_traditional_subadditivity_constraints(model, p, feasible_bundles_list, n):
    """
    Traditional subadditivity constraint addition method (fallback option)
    """
    bundle_product_sets = {}
    for bundle_id in feasible_bundles_list:
        bundle_product_sets[bundle_id] = bundle_to_product_set(bundle_id, n)

    added_constraints = set()

    for k in feasible_bundles_list:
        if k == 0:
            continue

        k_set = bundle_product_sets[k]
        if len(k_set) == 0:
            continue

        for i in feasible_bundles_list:
            for j in feasible_bundles_list:
                if i >= j:
                    continue

                i_set = bundle_product_sets[i]
                j_set = bundle_product_sets[j]
                union_set = i_set.union(j_set)

                if (k_set.issubset(union_set) and
                    k_set != i_set and
                    k_set != j_set):

                    constraint_key = (k, min(i, j), max(i, j))
                    if constraint_key not in added_constraints:
                        model.addConstr(p[k] <= p[i] + p[j])
                        added_constraints.add(constraint_key)



def top_m_selection(output, m, threshold=0.5):
    """
    Apply top-M selection to the output matrix for each segment with strict probability threshold.
    For each column (segment), first select top M products, then filter out those with probability < threshold.
    This ensures only high-quality products (prob >= threshold) are considered, even if it results in
    fewer than M products per segment.

    Args:
        output: n x m matrix where n is products, m is segments
        m: number of segments (also the number of top products to select per segment)
        threshold: probability threshold (default: 0.5)

    Returns:
        selected_products: list of lists, each containing the selected product indices
                          for each segment, sorted by probability (descending).
                          Note: lists may have different lengths if some segments have
                          fewer than M products above threshold.
    """
    n, m_segments = output.shape

    selected_products = []

    for j in range(m_segments):  # for each segment (column)
        # Step 1: Get top M products by probability (descending order)
        sorted_indices = np.argsort(output[:, j])[::-1]
        top_m_indices = sorted_indices[:n]

        # Step 2: Filter out products with probability < threshold
        filtered_indices = []
        for idx in top_m_indices:
            if output[idx, j] >= threshold:
                filtered_indices.append(idx)

        selected_products.append(filtered_indices)

    return selected_products


def generate_progressive_bundles(selected_products, n):
    """
    Generate progressive bundles from the selected products for each segment.
    For each segment, create bundles: {p1}, {p1,p2}, {p1,p2,p3}, ..., {p1,p2,...,pM}
    where p1, p2, ..., pM are the products sorted by probability (descending).

    Args:
        selected_products: list of lists, each containing selected product indices
                          for each segment, sorted by probability (descending)
        n: total number of products

    Returns:
        feasible_bundles: set of all feasible bundle indices (as integers)
    """
    feasible_bundles = set()

    # Always include the empty bundle (no products selected)
    feasible_bundles.add(0)
    
    bin2num = lambda x: int(''.join(map(str, x.tolist())), 2)
    for selected_list in selected_products:
        # Generate progressive bundles: {p1}, {p1,p2}, {p1,p2,p3}, ..., {p1,p2,...,pM}
        for i in range(1, len(selected_list) + 1):  # from 1 to len(selected_list)
            # Create bundle with first i products
            bundle_array = np.zeros(n, dtype=int)
            for product_idx in selected_list[:i]:
                bundle_array[product_idx] = 1
            bundle_idx = bin2num(bundle_array)
            feasible_bundles.add(bundle_idx)

    return feasible_bundles


def main():
    # Set default paths
    _base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(_base, "models_multi_layer_edge_update", "model_edge_4layer_seed1.pt")
    dataset_dir = os.path.join(_base, "dataset2_4_2026")
    
    # MB subdirs in dataset2_4_2026 (subdir structure, not flat test_data/)
    mb_subdirs = ["test_m10n10_1e_3", "test_m20n10_1e_3", "test_m30n10_1e_3"]
    result_path = 'test_result_topm_vs_optimal.csv'
    
    # Debug: Print paths
    print(f"Base directory: {_base}")
    print(f"Model path: {model_path}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"MB subdirs: {mb_subdirs}")

    # Set M = number of segments

    print('Loading trained model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load(model_path, map_location=device)
    model.to(device)
    
    # Load test dataset from MB subdirs
    print('\nBegin model evaluation...')
    test_dataset = []
    miscellaneous_dataset = []
    print('Start reading the dataset...')
    
    for subdir in mb_subdirs:
        sub_path = os.path.join(dataset_dir, subdir)
        if not os.path.exists(sub_path):
            continue
        for fname in sorted(os.listdir(sub_path)):
            if fname == '.DS_Store' or not fname.endswith('.msgpack'):
                continue
            file_path = os.path.join(sub_path, fname)
            try:
                dat, miscellaneous = process_data(file_path)
                test_dataset.append(dat)
                miscellaneous_dataset.append(miscellaneous)
            except Exception as e:
                print(f"Error processing file {fname}: {e}")
                continue
    
    sample_num = len(test_dataset)
    print(f'Successfully loaded {sample_num} test samples.')
    
    # Evaluate model
    results = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(sample_num), desc="Evaluating"):
            try:
                dat = test_dataset[i].to(device)
                miscellaneous = miscellaneous_dataset[i]
                n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = miscellaneous
                
                start_time = time.time()
                
                # Inference with support for EdgeScoringGCN and legacy node-output models
                raw_out = model(dat)
                if isinstance(raw_out, dict):
                    # EdgeScoringGCN-style output
                    if 'logit_matrix' in raw_out:
                        logits_nm = raw_out['logit_matrix'].detach().cpu().numpy()
                    elif 'edge_logits' in raw_out:
                        s = raw_out['edge_logits'].detach().cpu().numpy()
                        logits_nm = s.reshape(n, segment_num)
                    else:
                        raise ValueError('Unexpected model output keys for edge scoring: ' + ','.join(raw_out.keys()))
                    # Convert logits to probabilities (sigmoid)
                    sigmoid_output = 1.0 / (1.0 + np.exp(-logits_nm))
                else:
                    # Legacy node-output model
                    output = raw_out[:n, :].detach().cpu().numpy()
                    sigmoid_output = np.exp(output) / (np.exp(output) + np.exp(1))
                
                # Apply top-M selection where M = number of segments with threshold=0.5
                selected_products = top_m_selection(sigmoid_output, m=n, threshold=0.5)
                
                # Generate progressive bundles for MILP optimization
                feasible_bundles = generate_progressive_bundles(selected_products, n)

                ratio_optimized, optimized_runtime = revenue_ratio(n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_rev, feasible_bundles, selected_products, stored_cs, stored_Rs)
                
                end_time = time.time()
                topm_runtime = end_time - start_time
                
                runtime_ratio = topm_runtime / running_time if running_time > 0 else float('inf')
                
                # Check for invalid revenue ratios
                if ratio_optimized < 0 or np.isinf(opt_rev) or opt_rev == 0:
                    print(f"\nProblematic case detected in sample {i}:")
                    print(f"Optimized Revenue ratio: {ratio_optimized}")
                    print(f"Optimal revenue: {opt_rev}")
                    print(f"Number of products (n): {n}")
                    print(f"Number of segments: {segment_num}")
                    print(f"Selected products: {selected_products}")
                    print(f"Top-M runtime: {topm_runtime:.4f}s")
                    continue
                
                results.append([n, ratio_optimized, runtime_ratio, topm_runtime, optimized_runtime])
                
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                continue
    
    # Save results
    results = np.array(results)
    
    header = 'n_products,revenue_ratio_optimized,runtime_ratio,topm_runtime,optimized_runtime'
    # np.savetxt(result_path, results, delimiter=',', header=header, comments='')
     
    # Print summary statistics
    print(f'\nEvaluation completed!')
    print(f'Progressive bundles from products with prob >= 0.5 (m = n)')
    print(f'Results saved to: {result_path}')
    print(f'Number of samples evaluated: {len(results)}')
    if len(results) > 0:
        print(f'  Revenue Ratio Statistics:')
        print(f'  Average revenue ratio: {np.mean(results[:, 1]):.4f}')
        print(f'  Std revenue ratio: {np.std(results[:,1]):.4f}')
        print(f'  Min revenue ratio: {np.min(results[:, 1]):.4f}')
        print(f'  Max revenue ratio: {np.max(results[:, 1]):.4f}')
        print('')
        print(f'  Runtime Ratio Statistics:')
        print(f'  Average runtime ratio: {np.mean(results[:, 2]):.4f}')
        print(f'  Std runtime ratio: {np.std(results[:, 2]):.4f}')
        print(f'  Min runtime ratio: {np.min(results[:, 2]):.4f}')
        print(f'  Max runtime ratio: {np.max(results[:, 2]):.4f}')
        
        # Calculate and print actual runtime statistics
        topm_runtimes = results[:, 3]  # Get actual top-M runtimes
        optimized_runtimes = results[:, 4]  # Get actual optimized MILP runtimes
        print(f'\n Actual Runtime Statistics:')
        print(f'  Top-M approach runtime:')
        print(f'    Mean: {np.mean(topm_runtimes):.4f} seconds')
        print(f'    Std: {np.std(topm_runtimes):.4f} seconds')
        print(f'  Optimized MILP runtime:')
        print(f'    Mean: {np.mean(optimized_runtimes):.4f} seconds')
        print(f'    Std: {np.std(optimized_runtimes):.4f} seconds')


if __name__ == "__main__":
    main()
