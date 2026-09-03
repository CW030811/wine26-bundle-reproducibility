from __future__ import annotations

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


def bundle_to_product_set(bundle_id, n):
    """Convert bundle ID to product set"""
    binary_str = format(bundle_id, '0' + str(n) + 'b')
    return set(i for i, bit in enumerate(binary_str) if bit == '1')

def revenue_ratio(n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, pred_assort):
    """Optimized approach: only create variables and constraints for predicted bundles"""
    milp_start_time = time.time()

    bin2num = lambda x: int(''.join(map(str, x.tolist())), 2)
    segment_ind = np.array([i for i in range(m)])
    
    # Get unique predicted bundles
    # pred_assort (predicted product combination matrix) and generate corresponding feasible bundles
    bundle_dic = {}
    for i in range(m):
        bundle_idx = bin2num(pred_assort[i, :])
        try:
            bundle_dic[bundle_idx].append(i)
        except:
            bundle_dic[bundle_idx] = [i]
    
    predicted_bundles = list(bundle_dic.keys())
    # Only calculate for predicted bundles instead of all bundles
    
    # Calculate Rs and cs only for predicted bundles
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
    added_constraints = set()  # Avoid duplicate constraints

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
    
    # Special handling for empty bundle (bundle 0) if it exists in predicted bundles
    if 0 in predicted_bundles:
        model.addConstrs((S[k, 0] == 0 for k in segment_ind))

    model.optimize()

    milp_end_time = time.time()
    milp_time = milp_end_time - milp_start_time

    return model.ObjVal/opt_rev, milp_time


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
    _base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(_base, "models_multi_layer_edge_update", "model_edge_4layer_seed1.pt")
    dataset_dir = os.path.join(_base, "dataset2_4_2026")

    # MB subdirs in dataset2_4_2026 (subdir structure, not flat test_data/)
    mb_subdirs = ["test_m10n10_1e_3", "test_m20n10_1e_3", "test_m30n10_1e_3"]
    result_path = 'test_result_threshold_vs_optimal.csv'
    
    # Set default web results path if not provided
    if web_results_path is None:
        web_results_path = os.path.join(dataset_dir, 'gcn_inference_results.msgpack')

    # Debug: Print paths
    print(f"Base directory: ./")
    print(f"Model path: {model_path}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"MB subdirs: {mb_subdirs}")
    print(f"Use Web GCN: {use_web_gcn}")
    if use_web_gcn:
        print(f"Web results path: {web_results_path}")

    # Check if files exist
    print(f"Model file exists: {os.path.exists(model_path)}")
    for sub in mb_subdirs:
        sub_path = os.path.join(dataset_dir, sub)
        print(f"  {sub} exists: {os.path.exists(sub_path)}")
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
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()
    
    # Load test dataset from MB subdirs
    print('\nBegin model evaluation...')
    test_dataset = []
    miscellaneous_dataset = []
    file_names = []
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
                file_names.append(f"{subdir}/{fname}")
            except Exception as e:
                print(f"Error processing file {fname}: {e}")
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
    np.savetxt(result_path, combined_results, delimiter=',', header=header, comments='')
    
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
            print("  python test_bundle_threshold.py --local-gcn")
            print("  python test_bundle_threshold.py --web-gcn [path_to_results.msgpack]")
    else:
        # Default to local GCN mode
        print("Running with local GCN inference (default mode)...")
        main_with_local_gcn()
