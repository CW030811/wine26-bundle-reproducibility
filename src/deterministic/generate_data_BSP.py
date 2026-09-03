import numpy as np
import pandas as pd
import msgpack
import msgpack_numpy as mnp
import os
import torch
from torch_geometric.data import Data
import torch_geometric.utils
import gurobipy as gp
from gurobipy import GRB
import time
from itertools import combinations
import shutil

bin2num = lambda x: int(''.join(map(str, x.tolist())), 2)

# def solve_bundle_size_pricing_MILP(n, m, B, assortments, costs, Rs, Rbar, Ns):
#     """
#     Bundle size pricing: bundles of the same size have the same price
#     """
#     product_ind = np.array([i for i in range(n)])
#     segment_ind = np.array([i for i in range(m)])
#     bundle_ind = np.array([i for i in range(B)])
    
#     # Calculate bundle sizes
#     bundle_sizes = np.sum(assortments, axis=1)  # size of each bundle
#     max_size = int(np.max(bundle_sizes))
#     size_indices = range(max_size + 1)  # 0, 1, 2, ..., max_size

#     # Initialize objective history tracking
#     obj_hist = []
#     start_time = time.time()
#     last_record_time = 0
    
#     # Define callback function to record objective every 5 seconds
#     def callback_func(model, where):
#         nonlocal last_record_time, obj_hist, start_time
        
#         if where == GRB.Callback.MIP:
#             current_time = time.time()
#             elapsed_time = current_time - start_time
            
#             # Record every 5 seconds
#             if elapsed_time - last_record_time >= 5.0:
#                 obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
#                 obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
                
#                 # Record the better of the two (for maximization problem, we want the best incumbent)
#                 if obj_best != GRB.INFINITY:
#                     current_obj = obj_best
#                 else:
#                     current_obj = obj_bound if obj_bound != -GRB.INFINITY else None
                
#                 if current_obj is not None:
#                     obj_hist.append({
#                         'time': elapsed_time,
#                         'objective': current_obj
#                     })
                
#                 last_record_time = elapsed_time

#     model = gp.Model("Bundle Size Pricing MILP")

#     # Price variables: p[s] = price for bundles of size s
#     p = model.addVars(max_size + 1, vtype=GRB.CONTINUOUS, lb=0, name="p")
#     theta = model.addVars(m, B, vtype=GRB.BINARY, name="theta")
#     s = model.addVars(m, vtype=GRB.CONTINUOUS, name="s")
#     S = model.addVars(m, B, vtype=GRB.CONTINUOUS, lb=0, name='S')
#     Z = model.addVars(m, B, vtype=GRB.CONTINUOUS, name='Z')
#     P = model.addVars(m, B, vtype=GRB.CONTINUOUS, lb=0, name='P')

#     model.setObjective(gp.quicksum(Ns[k, 0]*Z[k, i] for k in segment_ind for i in bundle_ind), GRB.MAXIMIZE)

#     # Link bundle prices to size-based prices
#     # for i in bundle_ind:
#     #     bundle_size = int(bundle_sizes[i])
#     #     for k in segment_ind:
#     #         # model.addConstr(P[k, i] == p[bundle_size] * theta[k, i])
#     #         model.addConstr(P[k, i] <= p[bundle_size])
#     #         model.addConstr(P[k, i] >= p[bundle_size] - Rbar * (1 - theta[k, i]))

#     model.addConstrs((s[k] >= Rs[k, i] - p[int(bundle_sizes[i])] for i in bundle_ind for k in segment_ind))

#     # Subadditivity constraints for size-based pricing
#     for size1 in range(max_size + 1):
#         for size2 in range(max_size + 1):
#             if size1 + size2 <= max_size:
#                 model.addConstr(p[size1 + size2] <= p[size1] + p[size2])

#     # Monotonicity constraints: larger bundles should have higher or equal prices
#     for size in range(max_size):
#         model.addConstr(p[size + 1] >= p[size])

#     model.addConstrs((P[k, i] >= p[int(bundle_sizes[i])] - Rbar*(1-theta[k, i]) for i in bundle_ind for k in segment_ind))
#     model.addConstrs((P[k, i] <= p[int(bundle_sizes[i])] for i in bundle_ind for k in segment_ind))

#     model.addConstrs((s[k] >= gp.quicksum(Rs[k, i]*theta[j, i] - P[j, i] for i in bundle_ind) for k in segment_ind for j in segment_ind))

#     model.addConstrs((Z[k, i] == P[k, i] - costs[k, i] * theta[k, i] for i in bundle_ind for k in segment_ind))
#     model.addConstrs((S[k, i] == Rs[k, i]*theta[k, i] - P[k, i] for i in bundle_ind for k in segment_ind))
#     model.addConstrs((s[k] == gp.quicksum(S[k, i] for i in bundle_ind) for k in segment_ind))
#     model.addConstrs((gp.quicksum(theta[k, i] for i in bundle_ind) == 1 for k in segment_ind))
#     model.addConstrs((S[k, 0] == 0 for k in segment_ind))

#     model.setParam("OutputFlag", 1)
#     model.setParam("MIPGap", 1e-2)
#     # model.Params.TimeLimit = 600*2
#     t1 = time.time()
#     model.optimize(callback_func)
#     t2 = time.time()
    
#     # Record final objective if available
#     if model.SolCount > 0:
#         final_time = t2 - start_time
#         obj_hist.append({
#             'time': final_time,
#             'objective': model.ObjVal
#         })

#     # Check feasibility
#     feasible = model.SolCount > 0

#     if not feasible:
#         return None, None, None, model.Runtime, None, False, obj_hist

#     opt_bundles = []
#     opt_prices = {}
#     size_prices = {}
    
#     for size in range(max_size + 1):
#         size_prices[size] = p[size].X
    
#     for k in segment_ind:
#         for i in bundle_ind:
#             if theta[k, i].X >= 1 - 1e-2:
#                 opt_bundles.append(assortments[i, :].tolist())
#                 bundle_size = int(bundle_sizes[i])
#                 opt_prices[i] = size_prices[bundle_size]

#     return opt_bundles, opt_prices, model.ObjVal, model.Runtime, model.MIPGap, True, obj_hist


def solve_bundle_size_pricing_MILP_v2(n, m, unit_us, unit_cs, ship_cs, Ns):
    """
    Bundle size pricing: bundles of the same size have the same price
    New implementation using size-based variables (theta[k,s], P[k,s]) instead of bundle-based
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
    size_indices = range(max_size + 1)  # 0, 1, 2, ..., max_size
    
    # For each size, compute maximum valuation and corresponding cost for each customer
    # v_ks: customer k's maximum valuation for size s
    # c_ks: cost of size s for customer k
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
    
    # Initialize objective history tracking
    obj_hist = []
    start_time = time.time()
    last_record_time = 0
    
    # Define callback function to record objective every 5 seconds
    def callback_func(model, where):
        nonlocal last_record_time, obj_hist, start_time
        
        if where == GRB.Callback.MIP:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Record every 5 seconds
            if elapsed_time - last_record_time >= 5.0:
                obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
                
                # Record the better of the two (for maximization problem, we want the best incumbent)
                if obj_best != GRB.INFINITY:
                    current_obj = obj_best
                else:
                    current_obj = obj_bound if obj_bound != -GRB.INFINITY else None
                
                if current_obj is not None:
                    obj_hist.append({
                        'time': elapsed_time,
                        'objective': current_obj
                    })
                
                last_record_time = elapsed_time

    model = gp.Model("Bundle Size Pricing MILP v2")

    # Variables (using original naming convention):
    # p[s] = price for bundles of size s
    p = model.addVars(max_size + 1, vtype=GRB.CONTINUOUS, lb=0, name="p")
    # theta[k,s] = 1 if customer k purchases a bundle of size s, 0 otherwise  
    theta = model.addVars(m, max_size + 1, vtype=GRB.BINARY, name="theta")
    # P[k,s] = price paid by customer k for bundle of size s
    P = model.addVars(m, max_size + 1, vtype=GRB.CONTINUOUS, lb=0, name="P")
    # S[k,s] = surplus of customer k from bundle of size s
    S = model.addVars(m, max_size + 1, vtype=GRB.CONTINUOUS, lb=0, name="S")
    # Z[k,s] = profit from customer k choosing bundle of size s
    Z = model.addVars(m, max_size + 1, lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Z")
    # surplus[k] = total surplus of customer k  
    surplus = model.addVars(m, vtype=GRB.CONTINUOUS, name="s")

    # Objective: maximize total profit (using original format)
    model.setObjective(gp.quicksum(Ns[k, 0]*Z[k, s] for k in segment_ind for s in size_indices), GRB.MAXIMIZE)

    # Constraints (using original format and variable names):
    
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
    t1 = time.time()
    model.optimize(callback_func)
    t2 = time.time()
    
    # Check if we have a valid solution
    if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
        final_time = t2 - start_time
        obj_hist.append({
            'time': final_time,
            'objective': model.ObjVal
        })
        
        # Extract solution - only return prices for actually selected sizes
        selected_sizes = set()
        size_prices = {}
        solution_sizes = {}  # customer -> selected size
        
        # First, identify which sizes are actually selected by customers
        for k in segment_ind:
            for s in size_indices:
                if theta[k, s].X >= 1 - 1e-2:
                    selected_sizes.add(s)
                    solution_sizes[k] = s
        
        # Only return prices for selected sizes
        for s in selected_sizes:
            size_prices[s] = p[s].X
        
        # Note: In the new version, we return both customer size choices and size prices
        return solution_sizes, size_prices, model.ObjVal, model.Runtime, model.MIPGap, True, obj_hist
    else:
        # No valid solution found
        return None, None, None, model.Runtime, None, False, obj_hist


def generate_sample(m, l, u, sample_num, folder_path):
    '''
    Generate small-size samples for bundle size pricing
    '''
    import os
    import shutil

    def delete_all_files(folder_path):
        """
        删除指定文件夹下的所有文件（保留空文件夹）
        """
        if os.path.exists(folder_path):
            # 文件夹已存在，询问用户是否删除
            print(f"\n⚠️  警告: 文件夹 '{folder_path}' 已存在！")
            confirm = input("是否删除该文件夹？(yes/no): ").strip().lower()
            
            if confirm in ['yes', 'y']:
                try:
                    shutil.rmtree(folder_path)
                    print(f"✓ 文件夹 '{folder_path}' 删除成功。")
                    time.sleep(2)
                except Exception as e:
                    print(f"❌ 删除失败: {e}")
                    return
            else:
                print("❌ 取消删除。程序退出。")
                exit(0)
        else:
            print(f"文件夹 '{folder_path}' 不存在，将创建新文件夹。")
        
        os.makedirs(folder_path, exist_ok=True)
        print(f"✓ 文件夹 '{folder_path}' 创建成功。")

    folder_to_clean = folder_path
    delete_all_files(folder_to_clean)
    
    for iter in range(sample_num):
        if np.mod(iter + 1, 10) == 0:
            print('Generating samples: {}/{}'.format(iter+1, sample_num))
        
        # Generate ship_cs for each sample
        ship_cs = np.random.rand(m, 1)
        
        n = np.random.randint(l, u) # product number
        unit_cs = np.random.rand(1, n)
        
        unit_us = np.random.rand(m, n)  
        
        Xs = np.random.rand(m, 1)
        Ns = Xs/(np.sum(Xs))

        result = solve_bundle_size_pricing_MILP_v2(n, m, unit_us, unit_cs, ship_cs, Ns)
        solution_sizes, size_prices, opt_rev, runtime, gap, feasible, obj_hist = result

        if not feasible:
            print(f"Sample {iter+1}: infeasible model – skipping save.")
            continue

        # Convert to bundle format
        opt_bundles = []
        opt_prices = {}
        
        for chosen_size, price in size_prices.items():
            opt_prices[chosen_size] = price
        
        for k in range(m):
            if k in solution_sizes:
                chosen_size = solution_sizes[k]
                # Create bundle with top chosen_size products by valuation
                product_valuation = unit_us[k, :]
                if chosen_size > 0:
                    top_product_indices = np.argsort(product_valuation)[-chosen_size:]
                    bundle_binary = np.zeros(n, dtype=int)
                    bundle_binary[top_product_indices] = 1
                    opt_bundles.append(bundle_binary.tolist())
                else:
                    opt_bundles.append([0] * n)
            else:
                opt_bundles.append([0] * n)

        data_to_pack = {
            'product_num': n,
            'segment_num': m,
            'unit_cs': unit_cs,
            'ship_cs': ship_cs,
            'unit_us': unit_us,
            'Ns': Ns,
            'opt_bundles': opt_bundles,          # List format for threshold compatibility
            'opt_prices': opt_prices,
            'opt_rev': opt_rev,
            'running_time': runtime,
            'gap': gap,
            'obj_hist': obj_hist,
            'solution_sizes': solution_sizes,    # Size-based solution (customer -> size)
            'size_prices': size_prices,          # Size-based prices (size -> price)
        }

        path = folder_path + 'sample_data_{}_size_{}_sizepricing.msgpack'.format(iter+1, n)

        with open(path, 'wb') as f:
            msgpack.dump(data_to_pack, f, default=mnp.encode)
    return


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.abspath(__file__))
    # m = 10  # segment number


    # generate_sample(m, 20, 21, 100, dir_path + '/dataset/test_BSP_m10n20_correct_1e_3/') 
    # generate_sample(m, 30, 31, 100, dir_path + '/dataset/test_BSP_m10n30_correct_1e_3/')
    # generate_sample(m, 40, 41, 100, dir_path + '/dataset/test_BSP_m10n40_correct_1e_3/') 
    # generate_sample(m, 50, 51, 100, dir_path + '/dataset/test_BSP_m10n50_correct_1e_3/')
    # generate_sample(m, 60, 61, 100, dir_path + '/dataset/test_BSP_m10n60_correct_1e_3/')
    # generate_sample(m, 80, 81, 100, dir_path + '/dataset/test_BSP_m10n80_correct_1e_3/') 
    # generate_sample(m, 100, 101, 100, dir_path + '/dataset/test_BSP_m10n100_correct_1e_3/')
    # m = 20  # segment number
    # generate_sample(m, 20, 21, 100, dir_path + '/dataset/test_BSP_m20n20_correct_1e_3/') 
    # generate_sample(m, 30, 31, 100, dir_path + '/dataset/test_BSP_m20n30_correct_1e_3/')
    # generate_sample(m, 40, 41, 100, dir_path + '/dataset/test_BSP_m20n40_correct_1e_3/') 
    # generate_sample(m, 60, 61, 100, dir_path + '/dataset/test_BSP_m20n60_correct_1e_3/') 
    # generate_sample(m, 80, 81, 100, dir_path + '/dataset/test_BSP_m20n80_correct_1e_3/') 
    # generate_sample(m, 100, 101, 100, dir_path + '/dataset/test_BSP_m20n100_correct_1e_3/')
    
    # m = 30
    # generate_sample(m, 20, 21, 10, dir_path + '/dataset/test_BSP_m30n20_1e_3/') 
    m = 40
    generate_sample(m, 20, 21, 90, dir_path + '/dataset/test_BSP_m40n20_correct_1e_3_2/') 
    # m = 50
    # generate_sample(m, 20, 21, 10, dir_path + '/dataset/test_BSP_m50n20_1e_3/') 
    m = 60
    generate_sample(m, 20, 21, 40, dir_path + '/dataset/test_BSP_m60n20_correct_1e_3_2/') 
    # m = 70
    # generate_sample(m, 20, 21, 10, dir_path + '/dataset/test_BSP_m70n20_1e_3/') 
    m = 80
    generate_sample(m, 20, 21, 20, dir_path + '/dataset/test_BSP_m80n20_correct_1e_3_2/') 
    # m = 90
    # generate_sample(m, 20, 21, 10, dir_path + '/dataset/test_BSP_m90n20_1e_3/') 
    # m = 100
    # generate_sample(m, 20, 21, 10, dir_path + '/dataset/test_BSP_m100n20_1e_3/') 
    
    # generate_sample(m, 40, 41, 100, dir_path + '/dataset/test_BSP_m20n40_1e_3/') 
    # generate_sample(m, 60, 61, 100, dir_path + '/dataset/test_BSP_m20n60_1e_3/') 
    # generate_sample(m, 80, 81, 100, dir_path + '/dataset/test_BSP_m20n80_1e_3/') 
    # generate_sample(m, 100, 101, 100, dir_path + '/dataset/test_BSP_m20n100_1e_3/') 
    

    
    
    
    
    # generate_sample(m, 20, 21, 2000, dir_path + '/dataset/test-BSP-m10n20-0.2-nolim/') 
    # generate_sample(m, 30, 31, 2, dir_path + '/dataset/test-BSP-m10n30-0.2-nolim/') 
    # generate_sample(m, 30, 31, 2000, dir_path + '/dataset/test-BSP-m10n30-0.2-nolim/') 
    
    m = 15
    # generate_sample(m, 15, 16, 100, dir_path + '/dataset/test_BSP_m15n15/') 
    
    m = 20
    # generate_sample(m, 15, 16, 100, dir_path + '/dataset/test_BSP_m20n15/') 
    
    
    