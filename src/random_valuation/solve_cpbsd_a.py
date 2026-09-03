import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

# Prefer explicit academic license file when present
if not os.environ.get("GRB_LICENSE_FILE") and Path.home().joinpath('.gurobi', 'gurobi.lic').exists():
    os.environ["GRB_LICENSE_FILE"] = str(Path.home().joinpath('.gurobi', 'gurobi.lic'))

import gurobipy as gp
import msgpack
import msgpack_numpy as mnp
import numpy as np
from gurobipy import GRB


# CPBSD-A: preset preference rankings (approximation)
# Appendix B.7: Preset ranking pi_k, enforce surplus order constraints and solve compact MILP
# Ranking rule used here: sort by potential surplus z_kn = v_kn - c_n (descending)


def load_instance(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        obj = msgpack.load(f, object_hook=mnp.decode)
    v = np.asarray(obj["valuation_samples_V"], dtype=float)
    c = np.asarray(obj["production_cost_c"], dtype=float)
    return v, c


def build_rankings(v_kn: np.ndarray, c_n: np.ndarray) -> np.ndarray:
    """
    pi[k, j] gives product index at rank j (0-based), preset by ranking z=v-c.
    """
    K, N = v_kn.shape
    z = v_kn - c_n[None, :]
    pi = np.zeros((K, N), dtype=int)
    for k in range(K):
        pi[k] = np.argsort(-z[k])
    return pi


def precompute_vc_by_rank(v_kn: np.ndarray, c_n: np.ndarray, pi: np.ndarray):
    """
    vks[k,s], cks[k,s] for s=1..N following preset ranking pi.
    """
    K, N = v_kn.shape
    vks = np.zeros((K, N + 1), dtype=float)
    cks = np.zeros((K, N + 1), dtype=float)
    for k in range(K):
        pv = 0.0
        pc = 0.0
        for s in range(1, N + 1):
            idx = pi[k, s - 1]
            pv += v_kn[k, idx]
            pc += c_n[idx]
            vks[k, s] = pv
            cks[k, s] = pc
    return vks, cks


def solve_cpbsd_a(
    v_kn: np.ndarray,
    c_n: np.ndarray,
    mip_gap=1e-2,
    time_limit=300.0,
    output_flag=0,
    threads=0,
    big_m=None,
    p_ub=None,
    d_ub=None,
):
    K, N = v_kn.shape
    S = list(range(1, N + 1))
    K_idx = range(K)
    N_idx = range(N)

    vmax = float(np.max(v_kn))
    if p_ub is None:
        p_ub = vmax
    if d_ub is None:
        d_ub = p_ub
    if big_m is None:
        # q[k,s] linearizes q = pks * y, so M must upper-bound bundle prices pks.
        big_m = max(0.0, N * p_ub)

    pi = build_rankings(v_kn, c_n)
    vks, cks = precompute_vc_by_rank(v_kn, c_n, pi)

    m = gp.Model("CPBSD_A")

    p = m.addVars(N_idx, lb=0.0, ub=p_ub, vtype=GRB.CONTINUOUS, name="p")
    d = m.addVars(S, lb=0.0, ub=d_ub, vtype=GRB.CONTINUOUS, name="d")

    y = m.addVars(K_idx, S, vtype=GRB.BINARY, name="y")
    q = m.addVars(K_idx, S, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    pks = m.addVars(K_idx, S, lb=0.0, vtype=GRB.CONTINUOUS, name="pks")
    w_s = m.addVars(K_idx, S, lb=0.0, vtype=GRB.CONTINUOUS, name="w_s")
    w = m.addVars(K_idx, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

    # Warm start: p=c+eps, d=0, y selects best positive size (if any)
    eps = 1e-3
    for n in N_idx:
        p[n].Start = float(c_n[n] + eps)
    for s in S:
        d[s].Start = 0.0

    z = v_kn - c_n[None, :] - eps
    for k in K_idx:
        best_s = 0
        best_val = 0.0
        order = pi[k]
        prefix = 0.0
        for s in range(1, N + 1):
            prefix += z[k, order[s-1]]
            if prefix > best_val:
                best_val = prefix
                best_s = s
        for s in S:
            y[k, s].Start = 1.0 if s == best_s and best_s > 0 else 0.0

    m.setObjective(
        (1.0 / K) * gp.quicksum(q[k, s] - cks[k, s] * y[k, s] for k in K_idx for s in S),
        GRB.MAXIMIZE,
    )

    # (B-31) pks = sum_j p_{pi_kj} - s d_s
    m.addConstrs(
        (
            pks[k, s]
            == gp.quicksum(p[pi[k, j]] for j in range(s)) - s * d[s]
            for k in K_idx for s in S
        ),
        name="b31",
    )

    # (B-32) preserve preset ranking by surplus
    m.addConstrs(
        (
            v_kn[k, pi[k, j]] - p[pi[k, j]] >= v_kn[k, pi[k, j + 1]] - p[pi[k, j + 1]]
            for k in K_idx for j in range(N - 1)
        ),
        name="b32",
    )

    # wk >= vks - pks
    m.addConstrs((w[k] >= vks[k, s] - pks[k, s] for k in K_idx for s in S), name="wk_ge")

    # sum_s y_ks <= 1
    m.addConstrs((gp.quicksum(y[k, s] for s in S) <= 1 for k in K_idx), name="y_one")

    # qks = pks * yks via big-M
    m.addConstrs((q[k, s] >= pks[k, s] - big_m * (1 - y[k, s]) for k in K_idx for s in S), name="q_lb")
    m.addConstrs((q[k, s] <= pks[k, s] for k in K_idx for s in S), name="q_ub")

    # wks and wk
    m.addConstrs((w_s[k, s] == vks[k, s] * y[k, s] - q[k, s] for k in K_idx for s in S), name="w_s")
    m.addConstrs((w[k] == gp.quicksum(w_s[k, s] for s in S) for k in K_idx), name="w")

    # discount subadditivity
    m.addConstrs((s * d[s] >= s1 * d[s1] + (s - s1) * d[s - s1] for s in S for s1 in range(1, s)), name="d_sub")
    m.addConstr(d[1] == 0.0, name="d1_zero")

    m.setParam("MIPGap", mip_gap)
    m.setParam("TimeLimit", time_limit)
    m.setParam("OutputFlag", output_flag)
    if int(threads) > 0:
        m.setParam("Threads", int(threads))

    t0 = time.time()
    m.optimize()
    t1 = time.time()

    out = {
        "solver_status": int(m.Status),
        "sol_count": m.SolCount,
        "runtime": m.Runtime,
        "wall_time": t1 - t0,
        "node_count": float(m.NodeCount),
        "K": K,
        "N": N,
        "big_M": big_m,
        "p_ub": p_ub,
        "d_ub": d_ub,
        "warm_start_eps": eps,
        "mip_gap": float(m.MIPGap) if m.SolCount > 0 else None,
        "best_bound": float(m.ObjBound) if m.SolCount > 0 else None,
    }

    if m.SolCount > 0:
        out["objective"] = float(m.ObjVal)
        out["p"] = [p[n].X for n in N_idx]
        out["d"] = [0.0] + [d[s].X for s in S]

    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=str, default="")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260304)
    ap.add_argument("--mip-gap", type=float, default=1e-2)
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--output-flag", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--big-m", type=float, default=-1.0)
    ap.add_argument("--p-ub", type=float, default=-1.0)
    ap.add_argument("--d-ub", type=float, default=-1.0)
    ap.add_argument("--save-json", type=str, default="")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.instance:
        v_kn, c_n = load_instance(Path(args.instance))
    else:
        rng = np.random.default_rng(args.seed)
        v_kn = np.maximum(rng.normal(2.0, 0.8, size=(args.K, args.N)), 0.0)
        c_n = rng.uniform(0.1, 1.2, size=args.N)

    big_m = None if args.big_m <= 0 else args.big_m
    p_ub = None if args.p_ub <= 0 else args.p_ub
    d_ub = None if args.d_ub <= 0 else args.d_ub

    res = solve_cpbsd_a(
        v_kn,
        c_n,
        mip_gap=args.mip_gap,
        time_limit=args.time_limit,
        output_flag=args.output_flag,
        threads=args.threads,
        big_m=big_m,
        p_ub=p_ub,
        d_ub=d_ub,
    )
    text = json.dumps(res, ensure_ascii=False, indent=2)
    print(text)
    if args.save_json:
        p = Path(args.save_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
