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

bin2num = lambda x: int(''.join(map(str, x.tolist())), 2)

def solve_bundle_MILP(n, m, B, assortments, costs, Rs, Rbar, Ns):
    product_ind = np.array([i for i in range(n)])
    segment_ind = np.array([i for i in range(m)])
    bundle_ind = np.array([i for i in range(B)])

    model = gp.Model("Bundle MILP")

    p = model.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="p")
    theta = model.addVars(m, B, vtype=GRB.BINARY, name="theta")
    s = model.addVars(m, vtype=GRB.CONTINUOUS, name="s")
    S = model.addVars(m, B, vtype=GRB.CONTINUOUS, lb=0, name='S')
    Z = model.addVars(m, B, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='Z')
    P = model.addVars(m, B, vtype=GRB.CONTINUOUS, lb=0, name='P')

    model.setObjective(gp.quicksum(Ns[k, 0]*Z[k, i] for k in segment_ind for i in bundle_ind), GRB.MAXIMIZE)

    model.addConstrs((s[k] >= Rs[k, i] - p[i] for i in bundle_ind for k in segment_ind))
    for i in bundle_ind:
        tmp_assort = assortments[i, :]
        set_inds = np.where(tmp_assort)[0]
        for num in range(1, sum(tmp_assort)//2+1):
            for inds in combinations(set_inds, num):
                assort1 = np.zeros(n, dtype=int)
                assort1[list(inds)] = 1
                assort2 = tmp_assort - assort1
                model.addConstr(p[bin2num(tmp_assort)] <= p[bin2num(assort1)] + p[bin2num(assort2)])

    model.addConstrs((P[k, i] >= p[i] - Rbar*(1-theta[k, i]) for i in bundle_ind for k in segment_ind))
    model.addConstrs((P[k, i] <= p[i] for i in bundle_ind for k in segment_ind))

    model.addConstrs((s[k] >= gp.quicksum(Rs[k, i]*theta[j, i] - P[j, i] for i in bundle_ind) for k in segment_ind for j in segment_ind))

    model.addConstrs((Z[k, i] == P[k, i] - costs[k, i] * theta[k, i] for i in bundle_ind for k in segment_ind))
    model.addConstrs((S[k, i] == Rs[k, i]*theta[k, i] - P[k, i] for i in bundle_ind for k in segment_ind))
    model.addConstrs((s[k] == gp.quicksum(S[k, i] for i in bundle_ind) for k in segment_ind))
    model.addConstrs((gp.quicksum(theta[k, i] for i in bundle_ind) == 1 for k in segment_ind))
    model.addConstrs((S[k, 0] == 0 for k in segment_ind))

    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 1e-3)
    # model.setParam("MIPGap", 1e-7)
    
    model.Params.TimeLimit = 600
    t1 = time.time()
    model.optimize()
    t2 = time.time()

    # Check feasibility: if no feasible solution found, simply indicate infeasibility
    feasible = model.SolCount > 0

    if not feasible:
        # Return placeholders when infeasible; caller will handle
        return None, None, None, model.Runtime, None, False

    opt_bundles = []
    opt_prices = {}
    for k in segment_ind:
        for i in bundle_ind:
            # Guard against attributes being None in edge cases
            if theta[k, i].X >= 1 - 1e-2:
                opt_bundles.append(assortments[i, :].tolist())
                opt_prices[i] = p[i].X

    return opt_bundles, opt_prices, model.ObjVal, model.Runtime, model.MIPGap, True
        


def generate_sample(m, l, u, sample_num, folder_path):
    '''
    Generate small-size samples (with at most l products) for bundle pricing
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

    # 示例用法
    folder_to_clean = folder_path
    delete_all_files(folder_to_clean)
    for iter in range(sample_num):
        if np.mod(iter + 1, 10) == 0:
            print('Generating samples: {}/{}'.format(iter+1, sample_num))
        n = np.random.randint(l, u) # product number
        B = 2**n # bundle number
        unit_cs = np.random.rand(1, n)
        ship_cs = np.random.rand(m, 1)  # Generate ship_cs for each sample
        
        assortments = np.array([list(map(int, format(num, '0' + str(n) + 'b'))) for num in range(2**n)], dtype=int)
        costs = np.sum(assortments * unit_cs, axis=1) + ship_cs 
        costs = costs*0.2
        unit_us = np.random.rand(m, n)
        
        Rs = (unit_us.dot(assortments.T))**(1/3)
        # Rs = unit_us.dot(assortments.T)
        Rbar = np.max(Rs)
        Xs = np.random.rand(m, 1)
        Ns = Xs/(np.sum(Xs))
    
    
        opt_bundles, opt_prices, opt_rev, runtime, gap, feasible = solve_bundle_MILP(n, m, B, assortments, costs, Rs, Rbar, Ns)

        # Skip saving if the model was infeasible
        if not feasible:
            print(f"Sample {iter+1}: infeasible model – skipping save.")
            continue

        data_to_pack = {
            'product_num': n,
            'segment_num': m,
            'unit_cs': unit_cs,
            'ship_cs': ship_cs,
            'unit_us': unit_us,
            'cs': costs, 
            'Rs': Rs,
            'Ns': Ns,
            'opt_bundles': opt_bundles,
            'opt_prices': opt_prices,
            'opt_rev': opt_rev,
            'running_time': runtime,
            'gap': gap
        }

        path = folder_path + 'sample_data_{}_size_{}.msgpack'.format(iter+1, n)
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            msgpack.dump(data_to_pack, f, default=mnp.encode)
    return


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    # Test with m=20 and n=10
    m = 20  # segment number
    generate_sample(m, 10, 11, 100, dir_path + '/dataset/test_OOD/test_m20n10_f0.33_correct_1e_3/')
    m = 30
    generate_sample(m, 10, 11, 100, dir_path + '/dataset/test_OOD/test_m30n10_f0.33_correct_1e_3/')


    


    



    # np.random.seed(60)
    # n = 40
    # k = 5
    # rs = np.exp(np.random.beta(1, 1, (n, 1)))
    # vs = np.random.rand(n, k)*0.3 / rs
    # alp_vec = np.random.rand(k, 1)
    # alp_vec = alp_vec/sum(alp_vec)


    
    # np.random.seed(60)
    # n = 40
    # k = 5
    # rs = np.exp(np.random.beta(1, 1, (n, 1)))
    # vs = np.random.beta(0.1, 0.1, (n, k))*0.3
    # alp_vec = np.random.rand(k, 1)
    # alp_vec = alp_vec/sum(alp_vec)
    # # generate_sample(rs, vs, alp_vec, 5, 15, 2000, dir_path + '/dataset/train/')
    # generate_sample(rs, vs, alp_vec, 15, 25, 1000, dir_path + '/dataset/test/')
    # generate_sample(rs, vs, alp_vec, 25, 35, 200, dir_path + '/dataset/large_scale_test/')

    # n = 3
    # k = 2
    # rs = np.array([8, 4, 3])[:, np.newaxis]
    # alp_vec = np.array([1/2, 1/2])[:, np.newaxis]
    # vs = np.array([[5, 1/5], [20, 10], [1, 10]])
    # generate_sample(rs, vs, alp_vec, 3, 4, 1, dir_path + '/dataset/train/')


# def old_solve_MMNL(rs, vs, alp_vec, card_const):
#     '''
#     Solve MMNL by brute-force

#     '''
#     n = len(rs)
#     bst_assort = None
#     bst_rev = -1
#     for num in range(1, 2**n):
#         S = np.array([int(x) for x in format(num, '0'+str(n)+'b')][::-1])[:, np.newaxis]
#         if sum(S)[0] > card_const:
#             continue
#         rev = np.sum(vs*rs*S/(1+np.sum(vs*S, axis=0))[np.newaxis], axis=0).dot(alp_vec)[0]
#         if rev > bst_rev:
#             bst_rev = rev
#             bst_assort = S
#     base_rev = -1
#     for i in range(n):
#         S = rs >= rs[i]
#         if sum(S)[0] > card_const:
#             continue
#         rev = np.sum(vs*rs*S/(1+np.sum(vs*S, axis=0))[np.newaxis], axis=0).dot(alp_vec)[0]
#         if rev >= base_rev:
#             base_rev = rev
#     base_ratio = base_rev / bst_rev
#     return bst_assort, bst_rev, base_ratio


# def solve_MMNL(rs, vs, alp_vec, card_const, label):
#     n = len(rs)
    
#     if label in ['train', 'test']:
#         # Enumerate all possible assortments
#         assortments = np.array([list(map(int, format(num, '0' + str(n) + 'b'))) for num in range(1, 2**n)])

#         valid_assortments = assortments[np.sum(assortments, axis=1) <= card_const]
#         valid_revs = (valid_assortments.dot(vs * rs)/(1+valid_assortments.dot(vs))).dot(alp_vec)

#         bst_rev_idx = np.argmax(valid_revs)  
#         bst_assort = valid_assortments[bst_rev_idx, :]  
#         bst_assort = bst_assort[:, np.newaxis]
#         bst_rev = valid_revs[bst_rev_idx, 0]  
#     else:
#         bst_assort = None
#         bst_rev = 1
    
#     if label != 'train':
#         # Base revenue calculation: max revenue for assortments with at most `card_const` items
#         rev_assortments = np.array([(rs >= rs[i]).T[0].tolist() for i in range(n)])
#         valid_rev_assortments = rev_assortments[np.sum(rev_assortments, axis=1) <= card_const]
#         valid_rev_revs = (valid_rev_assortments.dot(vs * rs)/(1+valid_rev_assortments.dot(vs))).dot(alp_vec)
    
#         base_rev = np.max(valid_rev_revs)
#         base_ratio = base_rev / bst_rev
#     else:
#         base_ratio = None
    
#     return bst_assort, bst_rev, base_ratio
