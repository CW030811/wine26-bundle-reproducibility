"""
使用最新的 test_PCP_cp.py 方法生成 PCP 数据集。

更新内容：
1. 使用 PCP cutting-plane / lazy-constraint 的最新求解方法
2. 默认使用 seed=6
3. 输出生成所有数据的总时间
4. 若程序被中断，输出截至中断的累计时间和已生成 data 数量
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import time

import gurobipy as gp
import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
from gurobipy import GRB
from torch_geometric.data import Data
from tqdm import tqdm

from test_PCP_cp import (
    EdgeScoringGCN,
    _add_all_singleton_monotonicity_constraints,
    _add_progressive_chain_constraints,
    _build_segment_chain_data,
    _two_segment_lb_exceeds_target,
    cheapest_cover_dp_pruned,
    cheapest_cover_segments_dp,
    generate_progressive_bundles,
    top_m_selection,
)


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([EdgeScoringGCN])


def build_graph_data(n, m, unit_cs, ship_cs, unit_us, Ns):
    node_num = n + m
    feature = np.zeros((node_num, 4), dtype=float)

    uc = unit_cs[0] if isinstance(unit_cs, np.ndarray) and unit_cs.ndim == 2 else unit_cs
    feature[:n, 0] = np.asarray(uc).reshape(-1)[:n]
    feature[:n, 1] = np.asarray(np.average(unit_us, axis=0)).reshape(-1)[:n]

    if isinstance(Ns, np.ndarray) and Ns.ndim == 2:
        ns_vec = Ns[:m, 0]
    else:
        ns_vec = np.asarray(Ns).reshape(-1)[:m]
    feature[n:, 2] = ns_vec

    sc_vec = ship_cs[:m, 0] if isinstance(ship_cs, np.ndarray) and ship_cs.ndim == 2 else ship_cs
    feature[n:, 3] = np.asarray(sc_vec).reshape(-1)[:m]

    x = torch.tensor(feature, dtype=torch.float)

    prods = []
    custs = []
    edge_weights = []
    for i in range(n):
        for j in range(m):
            prods.append(i)
            custs.append(j + n)
            edge_weights.append([float(unit_us[j, i])])

    edge_index = torch.tensor([prods, custs], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    side_ind = torch.tensor([1] * n + [0] * m, dtype=torch.long).view(-1, 1)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        side_ind=side_ind,
        product_num=n,
        segment_num=m,
    )


def predict_progressive_bundles_with_model(model, data, n, m, device, threshold=0.5):
    data = data.to(device)

    inference_start = time.time()
    with torch.no_grad():
        raw_out = model(data)
        if isinstance(raw_out, dict):
            if "logit_matrix" in raw_out:
                logits_nm = raw_out["logit_matrix"].detach().cpu().numpy()
            elif "edge_logits" in raw_out:
                logits_nm = raw_out["edge_logits"].detach().cpu().numpy().reshape(n, m)
            else:
                raise ValueError("Unexpected model output keys: " + ",".join(raw_out.keys()))
            sigmoid_output = 1.0 / (1.0 + np.exp(-logits_nm))
        else:
            output = raw_out[:n, :].detach().cpu().numpy()
            sigmoid_output = np.exp(output) / (np.exp(output) + np.exp(1))
    gcn_time = time.time() - inference_start

    selected_products = top_m_selection(sigmoid_output, m=n, threshold=threshold)
    feasible_bundles = generate_progressive_bundles(selected_products, n)
    return selected_products, feasible_bundles, sigmoid_output, gcn_time


def bundle_id_to_binary(bundle_id, n):
    bits = format(int(bundle_id), f"0{n}b")
    return np.array([int(bit) for bit in bits], dtype=int)


def solve_progressive_choice_pricing_milp_cp(
    n,
    m,
    unit_us,
    unit_cs,
    ship_cs,
    Ns,
    feasible_bundles,
    selected_products,
):
    total_time_limit = 120.0
    viol_tol = 1e-6
    max_lazy_per_callback = 1

    segment_ind = np.arange(m)
    feasible_bundles_list = sorted(int(b) for b in feasible_bundles)

    assortments = np.zeros((len(feasible_bundles_list), n), dtype=int)
    for row_idx, bundle_id in enumerate(feasible_bundles_list):
        remaining = int(bundle_id)
        while remaining:
            lowbit = remaining & -remaining
            bit_pos = lowbit.bit_length() - 1
            assortments[row_idx, n - 1 - bit_pos] = 1
            remaining ^= lowbit

    Rs = np.sqrt(unit_us.dot(assortments.T))
    cs_base = np.sum(assortments * unit_cs, axis=1)
    cs = (cs_base + ship_cs) * 0.2
    Rbar = np.max(Rs)
    bundle_to_idx = {b: idx for idx, b in enumerate(feasible_bundles_list)}

    obj_hist = []
    last_record_time = 0.0

    model = None
    try:
        model = gp.Model("PCP_lazy_generation")
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 1e-3
        model.Params.TimeLimit = total_time_limit
        model.Params.LazyConstraints = 1
        model.Params.MIPFocus = 1
        model.Params.Cuts = 2

        p = model.addVars(feasible_bundles_list, lb=0.0, ub=float(Rbar), vtype=GRB.CONTINUOUS, name="p")
        theta = model.addVars(m, feasible_bundles_list, vtype=GRB.BINARY, name="theta")
        s = model.addVars(m, vtype=GRB.CONTINUOUS, name="s")
        S = model.addVars(m, feasible_bundles_list, lb=0.0, vtype=GRB.CONTINUOUS, name="S")
        Z = model.addVars(m, feasible_bundles_list, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Z")
        P = model.addVars(m, feasible_bundles_list, lb=0.0, vtype=GRB.CONTINUOUS, name="P")

        model.setObjective(
            gp.quicksum(Ns[k, 0] * Z[k, i] for k in segment_ind for i in feasible_bundles_list),
            GRB.MAXIMIZE,
        )

        model.addConstrs(
            (s[k] >= Rs[k, bundle_to_idx[i]] - p[i] for i in feasible_bundles_list for k in segment_ind)
        )
        model.addConstrs(
            (P[k, i] >= p[i] - Rbar * (1 - theta[k, i]) for i in feasible_bundles_list for k in segment_ind)
        )
        model.addConstrs((P[k, i] <= p[i] for i in feasible_bundles_list for k in segment_ind))
        model.addConstrs(
            (
                s[k]
                >= gp.quicksum(Rs[k, bundle_to_idx[i]] * theta[j, i] - P[j, i] for i in feasible_bundles_list)
                for k in segment_ind
                for j in segment_ind
            )
        )
        model.addConstrs(
            (Z[k, i] == P[k, i] - cs[k, bundle_to_idx[i]] * theta[k, i] for i in feasible_bundles_list for k in segment_ind)
        )
        model.addConstrs(
            (S[k, i] == Rs[k, bundle_to_idx[i]] * theta[k, i] - P[k, i] for i in feasible_bundles_list for k in segment_ind)
        )
        model.addConstrs((s[k] == gp.quicksum(S[k, i] for i in feasible_bundles_list) for k in segment_ind))
        model.addConstrs((gp.quicksum(theta[k, i] for i in feasible_bundles_list) == 1 for k in segment_ind))

        _add_progressive_chain_constraints(model, p, selected_products, n, feasible_bundles_list)
        segment_chains, bundle_to_segs, bundle_primary_seg = _build_segment_chain_data(selected_products, n)

        if 0 in feasible_bundles_list:
            model.addConstr(p[0] == 0, name="p0_zero")
            model.addConstrs((S[k, 0] == 0 for k in segment_ind))

        nonempty_bundles = [b for b in feasible_bundles_list if b != 0]
        lazy_check_bundles = nonempty_bundles
        per_k_segment_options = None
        if selected_products is not None:
            per_k_segment_options = {}
            lazy_check_bundles = []
            for k in nonempty_bundles:
                segment_options = []
                for seg_idx, chain in enumerate(segment_chains):
                    seen_masks = set()
                    opts = []
                    for b in chain:
                        if b == k or bundle_primary_seg.get(b) != seg_idx:
                            continue
                        overlap = b & k
                        if overlap == 0 or overlap in seen_masks:
                            continue
                        seen_masks.add(overlap)
                        opts.append((overlap, int(b)))
                    if opts:
                        seg_union = 0
                        for overlap, _ in opts:
                            seg_union |= overlap
                        segment_options.append((tuple(opts), seg_union))

                if len(segment_options) < 2:
                    continue

                segment_options.sort(
                    key=lambda item: (len(item[0]), -bin(int(item[1])).count("1"))
                )

                ordered_options = [opts for opts, _ in segment_options]

                suffix_union = [0] * (len(ordered_options) + 1)
                for idx in range(len(ordered_options) - 1, -1, -1):
                    suffix_union[idx] = suffix_union[idx + 1] | segment_options[idx][1]

                if suffix_union[0] == k:
                    per_k_segment_options[int(k)] = (tuple(ordered_options), tuple(suffix_union))
                    lazy_check_bundles.append(int(k))

        _add_all_singleton_monotonicity_constraints(
            model,
            p,
            feasible_bundles_list,
            bundle_to_segs=bundle_to_segs,
        )

        added_lazy = set()

        def callback(cb_model, where):
            nonlocal last_record_time

            if where == GRB.Callback.MIP:
                elapsed = float(cb_model.cbGet(GRB.Callback.RUNTIME))
                if elapsed - last_record_time >= 5.0:
                    obj_best = cb_model.cbGet(GRB.Callback.MIP_OBJBST)
                    obj_bound = cb_model.cbGet(GRB.Callback.MIP_OBJBND)
                    current_obj = None
                    if obj_best != GRB.INFINITY:
                        current_obj = obj_best
                    elif obj_bound != -GRB.INFINITY:
                        current_obj = obj_bound
                    if current_obj is not None:
                        obj_hist.append({
                            "time": elapsed,
                            "objective": current_obj,
                        })
                    last_record_time = elapsed
                return

            if where != GRB.Callback.MIPSOL:
                return

            p_values = cb_model.cbGetSolution([p[b] for b in feasible_bundles_list])
            p_sol = dict(zip(feasible_bundles_list, p_values))
            bundles_by_price = sorted(
                (b for b in lazy_check_bundles if p_sol[b] > 1e-8),
                key=lambda b: p_sol[b],
                reverse=True,
            )

            cuts_added = 0
            for k in bundles_by_price:
                if per_k_segment_options is not None:
                    segment_options, _ = per_k_segment_options[k]
                    if _two_segment_lb_exceeds_target(segment_options, p_sol, p_sol[k], viol_tol):
                        continue
                    best_cost, best_cover = cheapest_cover_segments_dp(
                        k,
                        per_k_segment_options[k],
                        p_sol,
                        viol_tol=viol_tol,
                    )
                else:
                    best_cost, best_cover = cheapest_cover_dp_pruned(
                        k,
                        nonempty_bundles,
                        p_sol,
                        n,
                        viol_tol=viol_tol,
                    )

                if best_cover is None:
                    continue

                if len(best_cover) <= 1:
                    continue

                if best_cost < p_sol[k] - viol_tol:
                    key = (int(k), tuple(best_cover))
                    if key in added_lazy:
                        continue

                    cb_model.cbLazy(
                        p[k] <= gp.quicksum(p[b] for b in best_cover)
                    )
                    added_lazy.add(key)
                    cuts_added += 1

                    if cuts_added >= max_lazy_per_callback:
                        break

        model.optimize(callback)
        runtime = float(model.Runtime)
        status = int(model.Status)
        sol_count = int(model.SolCount)

        if sol_count <= 0:
            return None, None, 0.0, runtime, 1.0, False, obj_hist, None, cs, Rs

        opt_prices = {int(b): float(p[b].X) for b in feasible_bundles_list}
        solution_bundles = {}
        opt_bundles = np.zeros((m, n), dtype=int)
        for seg_idx in segment_ind:
            chosen_bundle = 0
            for bundle_id in feasible_bundles_list:
                if theta[seg_idx, bundle_id].X > 0.5:
                    chosen_bundle = int(bundle_id)
                    break
            solution_bundles[int(seg_idx)] = chosen_bundle
            opt_bundles[seg_idx] = bundle_id_to_binary(chosen_bundle, n)

        gap = float(model.MIPGap) if status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} else 1.0
        feasible = status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
        return (
            opt_bundles,
            opt_prices,
            float(model.ObjVal),
            runtime,
            gap,
            feasible,
            obj_hist,
            solution_bundles,
            cs,
            Rs,
        )
    finally:
        if model is not None:
            model.dispose()


def prepare_output_folder(folder_path):
    if os.path.exists(folder_path):
        print(f"\n⚠️  警告: 文件夹 '{folder_path}' 已存在！")
        confirm = input("是否删除该文件夹？(yes/no): ").strip().lower()
        if confirm in ["yes", "y"]:
            shutil.rmtree(folder_path)
            print(f"✓ 文件夹 '{folder_path}' 删除成功。")
            time.sleep(1)
        else:
            print("❌ 取消删除。程序退出。")
            raise SystemExit(0)
    else:
        print(f"文件夹 '{folder_path}' 不存在，将创建新文件夹。")

    os.makedirs(folder_path, exist_ok=True)
    print(f"✓ 文件夹 '{folder_path}' 创建成功。")


def print_generation_summary(start_time, successful_samples, interrupted=False):
    elapsed = time.time() - start_time
    status = "程序被中断" if interrupted else "生成完成"
    print(f"\n{'=' * 60}")
    print(f"{status}")
    print(f"总共生成 sample 的时间: {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")
    print(f"成功生成 sample 数量: {successful_samples}")
    print(f"{'=' * 60}")


def generate_sample(m, l, u, sample_num, folder_path, model_path, device, threshold=0.5):
    print(f"\n加载训练好的模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return 0

    import __main__
    __main__.EdgeScoringGCN = EdgeScoringGCN
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    print("模型加载成功")

    prepare_output_folder(folder_path)

    print(f"\n开始生成 {sample_num} 个 PCP 样本...")
    print(f"参数: m={m}, n范围=[{l}, {u}), threshold={threshold}")

    successful_samples = 0
    generation_start = time.time()

    try:
        for iter_idx in tqdm(range(sample_num), desc="生成样本"):
            n = np.random.randint(l, u)
            unit_cs = np.random.rand(1, n)
            ship_cs = np.random.rand(m, 1)
            unit_us = np.random.rand(m, n)
            Xs = np.random.rand(m, 1)
            Ns = Xs / np.sum(Xs)

            graph_data = build_graph_data(n, m, unit_cs, ship_cs, unit_us, Ns)

            try:
                selected_products, feasible_bundles, sigmoid_output, gcn_time = predict_progressive_bundles_with_model(
                    model,
                    graph_data,
                    n,
                    m,
                    device,
                    threshold,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\nSample {iter_idx + 1}: GCN 推理失败: {e}")
                continue

            try:
                result = solve_progressive_choice_pricing_milp_cp(
                    n,
                    m,
                    unit_us,
                    unit_cs,
                    ship_cs,
                    Ns,
                    feasible_bundles,
                    selected_products,
                )
                opt_bundles, opt_prices, opt_rev, runtime, gap, feasible, obj_hist, solution_bundles, cs, Rs = result
                solve_time = runtime
                total_runtime = gcn_time + solve_time
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\nSample {iter_idx + 1}: MILP 求解失败: {e}")
                continue

            if not feasible:
                print(f"\nSample {iter_idx + 1}: 无可行解，跳过保存")
                continue

            data_to_pack = {
                "product_num": n,
                "segment_num": m,
                "unit_cs": unit_cs,
                "ship_cs": ship_cs,
                "unit_us": unit_us,
                "Ns": Ns,
                "opt_bundles": opt_bundles,
                "opt_prices": opt_prices,
                "opt_rev": opt_rev,
                "running_time": total_runtime,
                "gcn_time": gcn_time,
                "solve_time": solve_time,
                "gap": gap,
                "obj_hist": obj_hist,
                "solution_bundles": solution_bundles,
                "selected_products": selected_products,
                "feasible_bundles": list(feasible_bundles),
                "sigmoid_output": sigmoid_output.tolist(),
                "threshold": threshold,
                "cs": cs,
                "Rs": Rs,
            }

            path = os.path.join(folder_path, f"sample_data_{successful_samples + 1}_size_{n}_pcp.msgpack")
            with open(path, "wb") as f:
                msgpack.dump(data_to_pack, f, default=mnp.encode)

            successful_samples += 1

            del graph_data
            del selected_products, feasible_bundles, sigmoid_output
            del opt_bundles, opt_prices, opt_rev, runtime, gap, feasible, obj_hist, solution_bundles, cs, Rs
            del gcn_time, solve_time, total_runtime
            del ship_cs, unit_cs, unit_us, Xs, Ns, data_to_pack
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print_generation_summary(generation_start, successful_samples, interrupted=True)
        raise

    print_generation_summary(generation_start, successful_samples, interrupted=False)
    return successful_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PCP dataset with PCP-cp method.")
    parser.add_argument(
        "--folder_name",
        type=str,
        default="train_PCP_m10n50_correct_4layer_seed9_lr3",
        help="输出文件夹名称，位置固定在项目根目录下的 dataset/ 中。",
    )
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.abspath(__file__))

    m = 10
    num_layers = 4
    seed = 9
    threshold = 0.5

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_dir = os.path.join(dir_path, "models_multi_layer_edge_update_correct_lr_3")
    model_candidates = [
        os.path.join(model_dir, f"best_model_edge_{num_layers}layer_seed{seed}.pt"),
        os.path.join(model_dir, f"model_edge_{num_layers}layer_seed{seed}.pt"),
        os.path.join(model_dir, f"model-{num_layers}layer_seed{seed}.pt"),
    ]

    model_path = None
    for path in model_candidates:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("未找到模型文件！搜索路径：")
        for path in model_candidates:
            print(f"  - {path}")
        raise SystemExit(1)

    print(f"找到模型: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"默认 seed: {seed}")

    try:
        generate_sample(
            m,
            50,
            51,
            3000,
            os.path.join(dir_path, "dataset", args.folder_name),
            model_path,
            device,
            threshold,
        )
    except KeyboardInterrupt:
        print("已停止生成。")
