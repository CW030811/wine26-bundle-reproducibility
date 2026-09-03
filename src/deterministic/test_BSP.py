import numpy as np
import argparse
import csv
import msgpack
import msgpack_numpy as mnp
import os
import time
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations

bin2num = lambda x: int(''.join(map(str, x.tolist())), 2)


def process_data(file_path):
    """
    Load and process data from a msgpack file
    动态计算 cs / Rs（不依赖存储值）
    """
    with open(file_path, 'rb') as f:
        data = msgpack.load(f, object_hook=mnp.decode, strict_map_key=False)
    
    product_num = data['product_num']
    segment_num = data['segment_num']
    unit_cs = data['unit_cs']
    ship_cs = data['ship_cs']
    unit_us = data['unit_us']
    Ns = data['Ns']          # segment sizes (m, 1)
    opt_bundles = data['opt_bundles']
    opt_prices = data['opt_prices']
    opt_rev = data['opt_rev']
    running_time = data['running_time']
    gap = data['gap']

    # ---------- 动态计算成本时所需的基础信息 ----------
    n = int(product_num)
    m = int(segment_num)
    
    # 处理 unit_cs 形状，拉平成 (n,)
    unit_cs_vec = np.asarray(unit_cs).reshape(-1)[:n]
    # 处理 ship_cs 形状，拉平成 (m,)
    ship_cs_vec = np.asarray(ship_cs).reshape(-1)[:m]

    # 处理 unit_us 形状，确保 (m, n)
    unit_us_mat = np.asarray(unit_us)
    if unit_us_mat.ndim == 1:
        unit_us_mat = unit_us_mat.reshape(1, -1)
    unit_us_mat = unit_us_mat[:m, :n]
    
    return {
        'product_num': product_num,
        'segment_num': segment_num,
        'unit_cs': unit_cs,
        'ship_cs': ship_cs,
        'unit_us': unit_us,
        'Ns': Ns,
        'opt_bundles': opt_bundles,
        'opt_prices': opt_prices,
        'opt_rev': opt_rev,
        'running_time': running_time,
        'gap': gap
    }


def solve_bsp_for_evaluation(n, m, unit_us, unit_cs, ship_cs, Ns):
    """
    BSP solver using size-based variables for evaluation.
    """
    segment_ind = range(m)
    
    # 处理 unit_us 形状，确保 (m, n)
    unit_us_mat = np.asarray(unit_us)
    if unit_us_mat.ndim == 1:
        unit_us_mat = unit_us_mat.reshape(1, -1)
    unit_us_mat = unit_us_mat[:m, :n]
    
    # 处理 unit_cs 形状，拉平成 (n,)
    unit_cs_vec = np.asarray(unit_cs).reshape(-1)[:n]
    
    # 处理 ship_cs 形状，拉平成 (m,)
    ship_cs_vec = np.asarray(ship_cs).reshape(-1)[:m]
    
    max_size = n
    size_indices = range(max_size + 1)
    
    # For each size, compute maximum valuation and corresponding cost for each customer
    v_ks = np.zeros((m, max_size + 1))  # customer k's max valuation for size s
    c_ks = np.zeros((m, max_size + 1))  # cost for customer k to get size s
    
    for s in range(max_size + 1):
        for k in segment_ind:
            if s == 0:
                v_ks[k, s] = 0
                c_ks[k, s] = 0
            else:
                # Select top s products by utility for customer k
                top_s_indices = np.argsort(unit_us_mat[k, :])[-s:]
                # Valuation = sqrt(sum of top s utilities)
                v_ks[k, s] = np.sqrt(np.sum(unit_us_mat[k, top_s_indices]))
                # Cost = sum of unit costs for those s products + shipping cost, times 0.2
                c_ks[k, s] = (np.sum(unit_cs_vec[top_s_indices]) + ship_cs_vec[k]) * 0.2
    
    Rbar = np.max(np.sum(unit_us_mat, axis=1))

    model = gp.Model("BSP Evaluation")
    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 1e-3)
    model.Params.TimeLimit = 600

    # Variables:
    # p[s] = price for bundles of size s
    p = model.addVars(max_size + 1, vtype=GRB.CONTINUOUS, lb=0, name="p")
    # theta[k,s] = 1 if customer k purchases a bundle of size s, 0 otherwise  
    theta = model.addVars(m, max_size + 1, vtype=GRB.BINARY, name="theta")
    # P[k,s] = price paid by customer k for bundle of size s
    P = model.addVars(m, max_size + 1, vtype=GRB.CONTINUOUS, lb=0, name="P")
    # S[k,s] = surplus of customer k from bundle of size s
    S = model.addVars(m, max_size + 1, vtype=GRB.CONTINUOUS, lb=0, name="S")
    # Z[k,s] = profit from customer k choosing bundle of size s
    Z = model.addVars(m, max_size + 1, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name="Z")
    # surplus[k] = total surplus of customer k  
    surplus = model.addVars(m, vtype=GRB.CONTINUOUS, name="s")

    # Objective: maximize total profit
    model.setObjective(gp.quicksum(Ns[k, 0]*Z[k, s] for k in segment_ind for s in size_indices), GRB.MAXIMIZE)

    # Constraints:
    
    # IC constraint: surplus[k] >= v_ks[k,s] - p[s] for all k, s
    model.addConstrs((surplus[k] >= v_ks[k, s] - p[s] for k in segment_ind for s in size_indices))
    
    # Each customer chooses exactly one bundle size  
    model.addConstrs((gp.quicksum(theta[k, s] for s in size_indices) == 1 for k in segment_ind))
    
    # Price consistency: P[k,s] >= p[s] - M*(1-theta[k,s])
    model.addConstrs((P[k, s] >= p[s] - Rbar * (1 - theta[k, s]) 
                     for k in segment_ind for s in size_indices))
    
    # Price upper bound: P[k,s] <= p[s]
    model.addConstrs((P[k, s] <= p[s] for k in segment_ind for s in size_indices))
    
    # Surplus definition: S[k,s] = v_ks[k,s] * theta[k,s] - P[k,s]
    model.addConstrs((S[k, s] == v_ks[k, s] * theta[k, s] - P[k, s] 
                     for k in segment_ind for s in size_indices))
    
    # Total surplus: surplus[k] = sum_s S[k,s]
    model.addConstrs((surplus[k] == gp.quicksum(S[k, s] for s in size_indices) 
                     for k in segment_ind))
    
    # IC constraint between customers: surplus[k] >= sum_s (v_ks[k,s] * theta[j,s] - P[j,s])
    # for all k, j != k
    model.addConstrs((surplus[k] >= gp.quicksum(v_ks[k, s] * theta[j, s] - P[j, s] for s in size_indices)
                     for k in segment_ind for j in segment_ind if j != k))
    
    # Profit definition: Z[k,s] = P[k,s] - c_ks[k,s] * theta[k,s]  
    model.addConstrs((Z[k, s] == P[k, s] - c_ks[k, s] * theta[k, s] for k in segment_ind for s in size_indices))
    
    # Subadditivity constraints for size-based pricing
    for s1 in range(max_size + 1):
        for s2 in range(max_size + 1):
            if s1 + s2 <= max_size:
                model.addConstr(p[s1 + s2] <= p[s1] + p[s2])

    # Monotonicity constraints: larger bundles should have higher or equal prices
    for s in range(max_size):
        model.addConstr(p[s + 1] >= p[s])
    
    # No surplus from size 0 (empty bundle)
    model.addConstrs((S[k, 0] == 0 for k in segment_ind))

    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 1e-3)
    model.Params.TimeLimit = 600
    model.optimize()
    
    # Check if we have a valid solution
    if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
        # Extract solution - only return prices for actually selected sizes
        selected_sizes = set()
        size_prices = {}
        solution_sizes = {}  # segment -> selected size
        
        # First, identify which sizes are actually selected by customers
        for k in segment_ind:
            for s in size_indices:
                if theta[k, s].X >= 1 - 1e-2:
                    selected_sizes.add(s)
                    solution_sizes[k] = s
                    break
        
        # Only return prices for selected sizes
        for s in selected_sizes:
            size_prices[s] = p[s].X
        
        return model.ObjVal, model.Runtime, solution_sizes, size_prices
    else:
        # No valid solution found
        return None, model.Runtime, None, None


def evaluate_sample(data_dict):
    """
    Evaluate a single sample: compare BSP solution with optimal solution
    """
    n = data_dict['product_num']
    m = data_dict['segment_num']
    unit_us = data_dict['unit_us']
    unit_cs = data_dict['unit_cs']
    ship_cs = data_dict['ship_cs']
    Ns = data_dict['Ns']
    opt_rev = data_dict['opt_rev']
    opt_time = data_dict['running_time']
    
    # Solve BSP problem on this instance
    bsp_rev, bsp_time, solution_sizes, size_prices = solve_bsp_for_evaluation(n, m, unit_us, unit_cs, ship_cs, Ns)
    if bsp_rev is None:  # infeasible
        return None, None, None, None, None, None
        
    # Package solution
    bsp_solution = (solution_sizes, size_prices)
    
    # Calculate revenue ratio (BSP revenue / optimal revenue)
    revenue_ratio = bsp_rev / opt_rev if opt_rev > 0 else 0
    
    # Calculate time ratio (BSP time / optimal time)
    time_ratio = bsp_time / opt_time if opt_time > 0 else 0
    
    return revenue_ratio, time_ratio, bsp_time, bsp_rev, None, bsp_solution


def main():
    parser = argparse.ArgumentParser(description="Bundle Size Pricing evaluation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    )
    parser.add_argument(
        "--test_subdirs",
        type=str,
        default="data/deterministic/test_BSP_m10n100_1e_3",
        help="Test data subdirectories separated by semicolons.",
    )
    parser.add_argument("--result_dir", type=str, default="results/bsp")
    args = parser.parse_args()

    dir_path = os.path.abspath(args.data_dir)
    result_dir = args.result_dir if os.path.isabs(args.result_dir) else os.path.join(dir_path, args.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    datasets = {
        os.path.basename(subdir.rstrip("/")): (
            subdir if os.path.isabs(subdir) else os.path.join(dir_path, subdir)
        )
        for subdir in (part.strip() for part in args.test_subdirs.split(";"))
        if subdir
    }
    
    print("=" * 80)
    print("Bundle Size Pricing (BSP) Evaluation")
    print("=" * 80)
    print(f"Directory path: {dir_path}")
    print(f"Datasets to test: {list(datasets.keys())}")
    
    for dataset_name, test_data_path in datasets.items():
        print(f"\n{'='*80}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*80}")
        run_single_dataset(test_data_path, dataset_name, result_dir)


def _save_sample_results(results, result_dir, dataset_name):
    """Save the auditable BSP sample-level result table."""
    if not results:
        return None

    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"test_result_BSP_{dataset_name}.csv")
    columns = [
        "filename",
        "product_num",
        "segment_num",
        "revenue_ratio",
        "time_ratio",
        "bsp_time",
        "bsp_rev",
        "opt_rev",
        "opt_time",
    ]
    with open(result_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    return result_path


def run_single_dataset(test_data_path, dataset_name, result_dir):
    """Run evaluation on a single dataset"""
    
    if not os.path.exists(test_data_path):
        print(f"Dataset path does not exist: {test_data_path}")
        return
    
    print(f'Begin dataset loading...')
    print('Start reading the dataset...')
    
    # Load test dataset
    dir_list = sorted(os.listdir(test_data_path))
    sample_num = len([f for f in dir_list if f.endswith('.msgpack')])
    print(f'Found {sample_num} test samples.')
    print("Evaluating Bundle Size Pricing algorithm...")
    
    # Evaluate samples
    results = []
    failed_samples = 0
    
    for filename in tqdm(dir_list, desc="Evaluating BSP"):
        if not filename.endswith('.msgpack'):
            continue
            
        file_path = os.path.join(test_data_path, filename)
        
        try:
            # Load and process data
            data_dict = process_data(file_path)
            
            # Evaluate sample: BSP vs optimal
            result = evaluate_sample(data_dict)
            revenue_ratio, time_ratio, bsp_time, bsp_rev, bsp_price, bsp_solution = result
            
            if revenue_ratio is None:  # infeasible
                failed_samples += 1
                continue
            
            results.append({
                'filename': filename,
                'product_num': data_dict['product_num'],
                'segment_num': data_dict['segment_num'],
                'revenue_ratio': revenue_ratio,
                'time_ratio': time_ratio,
                'bsp_time': bsp_time,
                'bsp_rev': bsp_rev,
                'opt_rev': data_dict['opt_rev'],
                'opt_time': data_dict['running_time']
            })
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            failed_samples += 1
            continue
    
    # Save results
    if results:
        # Convert to arrays for analysis
        revenue_ratios = np.array([r['revenue_ratio'] for r in results])
        time_ratios = np.array([r['time_ratio'] for r in results])
        product_nums = np.array([r['product_num'] for r in results])
        bsp_times = np.array([r['bsp_time'] for r in results])
        opt_times = np.array([r['opt_time'] for r in results])
        
        result_path_final = _save_sample_results(results, result_dir, dataset_name)
        
        # Print summary statistics
        print(f'\n=== BUNDLE SIZE PRICING RESULTS ===')
        print(f'Test completed successfully')
        print(f'Results saved to: {result_path_final}')
        print(f'Number of samples evaluated: {len(results)}')
        print(f'Number of failed samples: {failed_samples}')
        
        # Revenue ratio statistics
        print(f'\n=== REVENUE RATIO STATISTICS ===')
        print(f'Mean: {np.mean(revenue_ratios):.4f}')
        print(f'Std:  {np.std(revenue_ratios):.4f}')
        print(f'Min:  {np.min(revenue_ratios):.4f}')
        print(f'Max:  {np.max(revenue_ratios):.4f}')
        
        # Time ratio statistics
        print(f'\n=== TIME RATIO STATISTICS ===')
        print(f'Mean: {np.mean(time_ratios):.4f}')
        print(f'Std:  {np.std(time_ratios):.4f}')
        print(f'Min:  {np.min(time_ratios):.4f}')
        print(f'Max:  {np.max(time_ratios):.4f}')
        
        # Absolute time statistics
        print(f'\n=== ABSOLUTE TIME STATISTICS ===')
        print(f'BSP Runtime:')
        print(f'  Mean: {np.mean(bsp_times):.4f} seconds')
        print(f'  Std:  {np.std(bsp_times):.4f} seconds')
        print(f'  Min:  {np.min(bsp_times):.4f} seconds')
        print(f'  Max:  {np.max(bsp_times):.4f} seconds')
        print(f'Optimal Runtime:')
        print(f'  Mean: {np.mean(opt_times):.4f} seconds')
        print(f'  Std:  {np.std(opt_times):.4f} seconds')
        print(f'  Min:  {np.min(opt_times):.4f} seconds')
        print(f'  Max:  {np.max(opt_times):.4f} seconds')
    
    else:
        print("No samples were successfully evaluated.")
        if failed_samples > 0:
            print(f"Total failed samples: {failed_samples}")


if __name__ == "__main__":
    main()
