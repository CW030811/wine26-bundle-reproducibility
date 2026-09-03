import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple

# Prefer explicit academic license file when present
if not os.environ.get("GRB_LICENSE_FILE") and Path.home().joinpath(".gurobi", "gurobi.lic").exists():
    os.environ["GRB_LICENSE_FILE"] = str(Path.home().joinpath(".gurobi", "gurobi.lic"))

import gurobipy as gp
import msgpack
import msgpack_numpy as mnp
import numpy as np
from gurobipy import GRB

# Bump when MB solve/eval semantics change so stale MB caches are recomputed.
MB_FORMULATION_VERSION = 6


def load_instance(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        obj = msgpack.load(f, object_hook=mnp.decode)
    v = np.asarray(obj["valuation_samples_V"], dtype=float)
    c = np.asarray(obj["production_cost_c"], dtype=float)
    return v, c


def build_assortments(n: int) -> np.ndarray:
    return np.array([list(map(int, format(num, f"0{n}b"))) for num in range(2**n)], dtype=int)


def ensure_empty_bundle(assortments: np.ndarray, n_products: int) -> np.ndarray:
    assortments = np.asarray(assortments, dtype=int)
    if assortments.ndim != 2 or assortments.shape[1] != n_products:
        raise ValueError(f"Expected assortments shape (?, {n_products}), got {assortments.shape}")
    empty = np.zeros((1, n_products), dtype=int)
    if assortments.shape[0] == 0:
        return empty
    has_empty = np.any(np.all(assortments == empty[0], axis=1))
    if has_empty:
        return assortments
    return np.vstack([empty, assortments])


def normalize_numeric_keys(d: Dict) -> Dict:
    if not isinstance(d, dict):
        return d
    out = {}
    for key, value in d.items():
        norm_key = key
        if isinstance(key, str):
            try:
                norm_key = int(key)
            except ValueError:
                norm_key = key
        elif isinstance(key, np.integer):
            norm_key = int(key)
        out[norm_key] = value
    return out


def json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def eval_mb_policy(v_eval: np.ndarray, c_n: np.ndarray, bundle_prices: Dict, assortments: np.ndarray) -> float:
    bundle_prices = normalize_numeric_keys(bundle_prices or {})
    assortments = np.asarray(assortments, dtype=int)
    k_count = v_eval.shape[0]
    bundle_cost = assortments @ c_n
    total = 0.0
    eps = 1e-9
    for k in range(k_count):
        # Use the outside option as the baseline, then break equal-surplus ties
        # toward the bundle with higher firm profit.
        best_surplus = 0.0
        best_bundle = None
        best_profit = 0.0
        for bundle_idx in range(assortments.shape[0]):
            price = bundle_prices.get(bundle_idx)
            if price is None:
                continue
            value = float(v_eval[k] @ assortments[bundle_idx])
            surplus = value - float(price)
            if abs(surplus) <= eps:
                surplus = 0.0
            profit = float(price) - float(bundle_cost[bundle_idx])
            if surplus > best_surplus + eps:
                best_surplus = surplus
                best_bundle = bundle_idx
                best_profit = profit
            elif abs(surplus - best_surplus) <= eps and profit > best_profit + eps:
                best_bundle = bundle_idx
                best_profit = profit
        if best_bundle is None:
            continue
        total += float(bundle_prices[best_bundle]) - float(bundle_cost[best_bundle])
    return total / k_count


def extract_mb_policy_info(result: Dict) -> Dict:
    assortments = np.asarray(result.get("assortments")) if result.get("assortments") is not None else None
    bundle_prices_full = normalize_numeric_keys(result.get("bundle_prices_full") or {})
    bundle_prices_selected = normalize_numeric_keys(result.get("bundle_prices_selected") or result.get("bundle_prices") or {})
    if bundle_prices_full:
        policy_scope = "full_bundle_prices"
        active_bundle_prices = bundle_prices_full
    elif bundle_prices_selected:
        policy_scope = "selected_only"
        active_bundle_prices = bundle_prices_selected
    else:
        policy_scope = "missing"
        active_bundle_prices = {}

    bundle_space_size = result.get("bundle_space_size")
    if bundle_space_size is None and assortments is not None:
        bundle_space_size = int(assortments.shape[0])

    return {
        "assortments": assortments,
        "bundle_prices_full": bundle_prices_full,
        "bundle_prices_selected": bundle_prices_selected,
        "active_bundle_prices": active_bundle_prices,
        "policy_scope": policy_scope,
        "bundle_space_size": bundle_space_size,
        "bundle_price_count_full": len(bundle_prices_full),
        "bundle_price_count_selected": len(bundle_prices_selected),
    }


def _disjoint_partition_pairs(bundle_bits: np.ndarray):
    """Standard 2-way disjoint partition subadditivity (Hanson & Martin 1990).

    Yields (mask1, mask2) for every way to split *bundle_bits* into two
    non-empty disjoint subsets whose union equals the bundle.  Only
    partitions with |S1| <= |S2| are generated to avoid duplicates.
    """
    set_inds = np.where(bundle_bits)[0]
    bundle_size = int(bundle_bits.sum())
    if bundle_size < 2:
        return
    n = len(bundle_bits)
    for num in range(1, bundle_size // 2 + 1):
        for inds in combinations(set_inds, num):
            s1 = np.zeros(n, dtype=int)
            s1[list(inds)] = 1
            s2 = bundle_bits - s1
            yield int("".join(map(str, s1.tolist())), 2), int("".join(map(str, s2.tolist())), 2)


def _canonical_partition(blocks):
    return tuple(sorted(tuple(sorted(block)) for block in blocks))


def _set_partitions(items):
    if len(items) == 1:
        yield ((items[0],),)
        return
    first = items[0]
    for partition in _set_partitions(items[1:]):
        yield ((first,),) + partition
        for idx in range(len(partition)):
            updated = list(partition)
            updated[idx] = tuple(sorted(updated[idx] + (first,)))
            yield _canonical_partition(updated)


def _restricted_full_partition_families(bundle_bits: np.ndarray, bundle_to_index: Dict[Tuple[int, ...], int]):
    items = tuple(np.where(bundle_bits == 1)[0].tolist())
    if len(items) <= 1:
        return tuple()
    seen = set()
    families = []
    for partition in _set_partitions(items):
        partition = _canonical_partition(partition)
        if len(partition) <= 1:
            continue
        mapped = []
        valid = True
        for block in partition:
            arr = np.zeros(len(bundle_bits), dtype=int)
            arr[list(block)] = 1
            idx = bundle_to_index.get(tuple(arr.tolist()))
            if idx is None:
                valid = False
                break
            mapped.append(int(idx))
        if not valid:
            continue
        family = tuple(sorted(mapped))
        if family in seen:
            continue
        seen.add(family)
        families.append(family)
    return tuple(sorted(families))


def _restricted_cover_pair_families(bundle_idx: int, assortments: np.ndarray):
    """Heuristic pairwise cover families used by the historical FCP test path.

    For a target bundle k, return all candidate pairs (i, j) such that
    bundle(k) is a strict subset of bundle(i) U bundle(j), matching the
    `test_FCP.py` pruning formulation. Overlap between i and j is allowed.
    """
    active_idx = list(range(1, assortments.shape[0]))
    bundle_sets = {
        idx: set(np.where(assortments[idx] == 1)[0].tolist())
        for idx in active_idx
    }
    target_set = bundle_sets.get(bundle_idx, set())
    if not target_set:
        return tuple()

    families = []
    for pos, i in enumerate(active_idx):
        if i == bundle_idx:
            continue
        i_set = bundle_sets[i]
        for j in active_idx[pos + 1 :]:
            if j == bundle_idx:
                continue
            j_set = bundle_sets[j]
            union_set = i_set.union(j_set)
            if target_set.issubset(union_set) and target_set != i_set and target_set != j_set:
                families.append((i, j))
    return tuple(families)


def eval_bsp_policy(v_eval: np.ndarray, c_n: np.ndarray, size_prices: Dict) -> float:
    size_prices = normalize_numeric_keys(size_prices or {})
    k_count, n_products = v_eval.shape
    total = 0.0
    for k in range(k_count):
        best_surplus = 0.0
        best_size = 0
        best_cost = 0.0
        order = np.argsort(-v_eval[k])
        prefix_val = 0.0
        prefix_cost = 0.0
        for size in range(1, n_products + 1):
            idx = order[size - 1]
            prefix_val += v_eval[k, idx]
            prefix_cost += c_n[idx]
            price = size_prices.get(size)
            if price is None:
                continue
            surplus = prefix_val - float(price)
            if surplus > best_surplus:
                best_surplus = surplus
                best_size = size
                best_cost = prefix_cost
        if best_surplus <= 0.0 or best_size == 0:
            continue
        total += float(size_prices[best_size]) - best_cost
    return total / k_count


def solve_mb(
    v_kn: np.ndarray,
    c_n: np.ndarray,
    time_limit: float = 300.0,
    mip_gap: float = 1e-2,
    output_flag: int = 0,
    threads: int = 0,
):
    assortments = build_assortments(v_kn.shape[1])
    return solve_mb_restricted(
        v_kn=v_kn,
        c_n=c_n,
        assortments=assortments,
        time_limit=time_limit,
        mip_gap=mip_gap,
        output_flag=output_flag,
        threads=threads,
    )


def solve_mb_restricted(
    v_kn: np.ndarray,
    c_n: np.ndarray,
    assortments: np.ndarray,
    time_limit: float = 300.0,
    mip_gap: float = 1e-2,
    output_flag: int = 0,
    threads: int = 0,
    subadditivity_mode: str = "full_partition",
):
    k_count, n_products = v_kn.shape
    assortments = ensure_empty_bundle(assortments, n_products)
    unique_rows = np.unique(assortments.astype(int), axis=0)
    assortments = unique_rows
    bundle_count = assortments.shape[0]
    bundle_cost = (assortments @ c_n).reshape(1, -1)
    costs = np.repeat(bundle_cost, k_count, axis=0)
    revenues = v_kn @ assortments.T
    revenue_ub = float(np.max(revenues))
    weights = np.ones((k_count, 1), dtype=float) / k_count
    bundle_idx = range(bundle_count)
    active_idx = list(range(1, bundle_count))
    segment_idx = range(k_count)

    bundle_to_index = {tuple(row.tolist()): idx for idx, row in enumerate(assortments)}

    model = gp.Model("Bundle_MILP_appendix_c_v2")
    p = model.addVars(active_idx, vtype=GRB.CONTINUOUS, lb=0.0, name="p")
    theta = model.addVars(k_count, active_idx, vtype=GRB.BINARY, name="theta")
    surplus = model.addVars(k_count, vtype=GRB.CONTINUOUS, lb=0.0, name="w")
    s_terms = model.addVars(k_count, active_idx, vtype=GRB.CONTINUOUS, name="S")
    # ZFIX: per-(customer,bundle) profit must be allowed negative. With the default
    # lb=0, Z>=0 forces p[i]>=cost[i] for every chosen bundle, which is NOT part of
    # the mixed-bundling problem and can cut off the true optimum whenever costs are
    # positive (inert at zero cost). See repro README.
    profit = model.addVars(k_count, active_idx, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Z")
    payment = model.addVars(k_count, active_idx, vtype=GRB.CONTINUOUS, lb=0.0, name="q")

    model.setObjective(
        gp.quicksum(weights[k, 0] * profit[k, i] for k in segment_idx for i in active_idx),
        GRB.MAXIMIZE,
    )

    model.addConstrs((surplus[k] >= revenues[k, i] - p[i] for i in active_idx for k in segment_idx), name="surplus_lb")
    if subadditivity_mode == "full_partition":
        for i in active_idx:
            for part_no, family in enumerate(_restricted_full_partition_families(assortments[i], bundle_to_index)):
                model.addConstr(p[i] <= gp.quicksum(p[j] for j in family), name=f"subadd_literal_{i}_{part_no}")
    elif subadditivity_mode == "predicted_cover_pairwise":
        for i in active_idx:
            for pair_no, family in enumerate(_restricted_cover_pair_families(i, assortments)):
                model.addConstr(p[i] <= gp.quicksum(p[j] for j in family), name=f"subadd_cover_{i}_{pair_no}")
    else:
        raise ValueError(f"Unknown restricted subadditivity mode: {subadditivity_mode}")

    model.addConstrs((payment[k, i] >= p[i] - revenue_ub * (1 - theta[k, i]) for i in active_idx for k in segment_idx), name="payment_lb")
    model.addConstrs((payment[k, i] <= p[i] for i in active_idx for k in segment_idx), name="payment_ub")
    model.addConstrs(
        (
            surplus[k] >= gp.quicksum(revenues[k, i] * theta[j, i] - payment[j, i] for i in active_idx)
            for k in segment_idx
            for j in segment_idx
            if j != k
        ),
        name="envy_like",
    )
    model.addConstrs((profit[k, i] == payment[k, i] - costs[k, i] * theta[k, i] for i in active_idx for k in segment_idx), name="profit")
    model.addConstrs((s_terms[k, i] == revenues[k, i] * theta[k, i] - payment[k, i] for i in active_idx for k in segment_idx), name="surplus_term")
    model.addConstrs((surplus[k] == gp.quicksum(s_terms[k, i] for i in active_idx) for k in segment_idx), name="surplus_sum")
    model.addConstrs((gp.quicksum(theta[k, i] for i in active_idx) <= 1 for k in segment_idx), name="at_most_one_bundle")

    model.setParam("OutputFlag", output_flag)
    model.setParam("MIPGap", mip_gap)
    model.setParam("TimeLimit", time_limit)
    if int(threads) > 0:
        model.setParam("Threads", int(threads))
    model.update()

    t0 = time.time()
    model.optimize()
    t1 = time.time()

    result = {
        "solver_status": int(model.Status),
        "feasible": model.SolCount > 0,
        "mb_formulation_version": MB_FORMULATION_VERSION,
        "runtime": model.Runtime,
        "wall_time": t1 - t0,
        "mip_gap": float(model.MIPGap) if model.SolCount > 0 else None,
        "objective": float(model.ObjVal) if model.SolCount > 0 else None,
        "revenue_in_sample": None,
        "model_num_vars": int(model.NumVars),
        "model_num_binvars": int(model.NumBinVars),
        "model_num_constrs": int(model.NumConstrs),
        "bundle_space_size": bundle_count,
        "subadditivity_mode": subadditivity_mode,
        "policy_scope": "full_bundle_prices" if model.SolCount > 0 else "missing",
        "bundle_prices": None,
        "bundle_prices_full": None,
        "bundle_prices_selected": None,
        "chosen_bundle_idx_by_customer": None,
        "chosen_product_matrix": None,
        "assortments": assortments,
    }
    if model.SolCount > 0:
        bundle_prices_full = {0: 0.0}
        bundle_prices_selected = {}
        for i in active_idx:
            bundle_prices_full[i] = float(p[i].X)
            chosen_any = any(theta[k, i].X >= 1 - 1e-2 for k in segment_idx)
            if chosen_any:
                bundle_prices_selected[i] = float(p[i].X)
        chosen_bundle_idx_by_customer = []
        chosen_product_matrix = np.zeros((k_count, n_products), dtype=int)
        for k in segment_idx:
            chosen_bundle = 0
            for i in active_idx:
                if theta[k, i].X >= 1 - 1e-2:
                    chosen_bundle = int(i)
                    break
            chosen_bundle_idx_by_customer.append(int(chosen_bundle))
            chosen_product_matrix[k, :] = assortments[chosen_bundle]
        # Keep the legacy field for compatibility, but make the full table explicit.
        result["bundle_prices"] = bundle_prices_selected
        result["bundle_prices_full"] = bundle_prices_full
        result["bundle_prices_selected"] = bundle_prices_selected
        result["chosen_bundle_idx_by_customer"] = chosen_bundle_idx_by_customer
        result["chosen_product_matrix"] = chosen_product_matrix
        result["revenue_in_sample"] = float(eval_mb_policy(v_kn, c_n, bundle_prices_full, assortments))
    return result


def solve_bsp(
    v_kn: np.ndarray,
    c_n: np.ndarray,
    time_limit: float = 300.0,
    mip_gap: float = 1e-2,
    output_flag: int = 0,
    threads: int = 0,
):
    k_count, n_products = v_kn.shape
    max_size = n_products
    size_idx = range(max_size + 1)
    segment_idx = range(k_count)
    revenue_ub = float(np.max(np.sum(np.sort(v_kn, axis=1)[:, ::-1], axis=1)))
    weights = np.ones((k_count, 1), dtype=float) / k_count

    v_ks = np.zeros((k_count, max_size + 1), dtype=float)
    c_ks = np.zeros((k_count, max_size + 1), dtype=float)
    for k in segment_idx:
        order = np.argsort(-v_kn[k])
        ordered_vals = v_kn[k, order]
        ordered_costs = c_n[order]
        v_ks[k, 1:] = np.cumsum(ordered_vals)
        c_ks[k, 1:] = np.cumsum(ordered_costs)

    model = gp.Model("Bundle_Size_Pricing_MILP_v2")
    p = model.addVars(max_size + 1, vtype=GRB.CONTINUOUS, lb=0.0, name="p")
    theta = model.addVars(k_count, max_size + 1, vtype=GRB.BINARY, name="theta")
    payment = model.addVars(k_count, max_size + 1, vtype=GRB.CONTINUOUS, lb=0.0, name="P")
    surplus_term = model.addVars(k_count, max_size + 1, vtype=GRB.CONTINUOUS, lb=0.0, name="S")
    # ZFIX: BSP per-(customer,size) profit must be allowed negative (same reasoning as
    # the MB solver above); the default lb=0 forces size-price >= prefix-cost.
    profit = model.addVars(k_count, max_size + 1, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Z")
    surplus = model.addVars(k_count, vtype=GRB.CONTINUOUS, name="surplus")

    model.setObjective(
        gp.quicksum(weights[k, 0] * profit[k, size] for k in segment_idx for size in size_idx),
        GRB.MAXIMIZE,
    )

    model.addConstrs((surplus[k] >= v_ks[k, size] - p[size] for k in segment_idx for size in size_idx), name="surplus_lb")
    model.addConstrs((gp.quicksum(theta[k, size] for size in size_idx) == 1 for k in segment_idx), name="one_choice")
    model.addConstrs((payment[k, size] >= p[size] - revenue_ub * (1 - theta[k, size]) for k in segment_idx for size in size_idx), name="payment_lb")
    model.addConstrs((payment[k, size] <= p[size] for k in segment_idx for size in size_idx), name="payment_ub")
    model.addConstrs((surplus_term[k, size] == v_ks[k, size] * theta[k, size] - payment[k, size] for k in segment_idx for size in size_idx), name="surplus_term")
    model.addConstrs((surplus[k] == gp.quicksum(surplus_term[k, size] for size in size_idx) for k in segment_idx), name="surplus_sum")
    model.addConstrs(
        (
            surplus[k] >= gp.quicksum(v_ks[k, size] * theta[j, size] - payment[j, size] for size in size_idx)
            for k in segment_idx for j in segment_idx if j != k
        ),
        name="envy_like",
    )
    model.addConstrs((profit[k, size] == payment[k, size] - c_ks[k, size] * theta[k, size] for k in segment_idx for size in size_idx), name="profit")
    for size1 in size_idx:
        for size2 in size_idx:
            if size1 + size2 <= max_size:
                model.addConstr(p[size1 + size2] <= p[size1] + p[size2], name=f"subadd_{size1}_{size2}")
    for size in range(max_size):
        model.addConstr(p[size + 1] >= p[size], name=f"monotone_{size}")
    model.addConstrs((surplus_term[k, 0] == 0 for k in segment_idx), name="empty_bundle")

    model.setParam("OutputFlag", output_flag)
    model.setParam("MIPGap", mip_gap)
    model.setParam("TimeLimit", time_limit)
    if int(threads) > 0:
        model.setParam("Threads", int(threads))

    t0 = time.time()
    model.optimize()
    t1 = time.time()

    result = {
        "solver_status": int(model.Status),
        "feasible": model.SolCount > 0,
        "runtime": model.Runtime,
        "wall_time": t1 - t0,
        "mip_gap": float(model.MIPGap) if model.SolCount > 0 else None,
        "objective": float(model.ObjVal) if model.SolCount > 0 else None,
        "size_prices": None,
    }
    if model.SolCount > 0:
        size_prices = {}
        for size in size_idx:
            chosen_any = any(theta[k, size].X >= 1 - 1e-2 for k in segment_idx)
            if chosen_any:
                size_prices[size] = float(p[size].X)
        result["size_prices"] = size_prices
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--method", choices=["mb", "bsp"], required=True)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--mip-gap", type=float, default=1e-2)
    ap.add_argument("--output-flag", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save-json", type=str, default="")
    args = ap.parse_args()

    v_kn, c_n = load_instance(Path(args.instance))
    if args.method == "mb":
        res = solve_mb(
            v_kn,
            c_n,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_flag=args.output_flag,
            threads=args.threads,
        )
    else:
        res = solve_bsp(
            v_kn,
            c_n,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_flag=args.output_flag,
            threads=args.threads,
        )

    text = json.dumps(res, ensure_ascii=False, indent=2, default=json_default)
    print(text)
    if args.save_json:
        Path(args.save_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
