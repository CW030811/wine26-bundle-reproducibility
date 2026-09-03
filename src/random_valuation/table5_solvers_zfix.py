#!/usr/bin/env python3
# This file is a direct-run consolidated CPBSD-GCN script.
# It intentionally does not import the original src/data project scripts.
"""Compare FCP-pruned-MB, PBDC, BSP, and CPBSD-A across cost setups.

Default experiment: normal/full/rho=0 with zero, random_ind, random_corr
costs and five seeds. FCP inference uses the K/N-invariant random_ind GCN
checkpoint for all cost setups. Outputs comparison_summary.csv/json.

--graph-variant hanson swaps FCP inference to the Hanson EdgeScoringGCN
(raw features, edge_dim=1); the class/builder are loaded from src/data so
they match the training definitions exactly. --fcp-only skips PBDC/BSP/
CPBSD-A so audited baseline rows can be reused.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

if not os.environ.get("GRB_LICENSE_FILE") and Path.home().joinpath(".gurobi", "gurobi.lic").exists():
    os.environ["GRB_LICENSE_FILE"] = str(Path.home().joinpath(".gurobi", "gurobi.lic"))

import gurobipy as gp
import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
from gurobipy import GRB
from torch_geometric.data import Data

def _safe_float_attr(obj, name: str):
    try:
        return float(getattr(obj, name))
    except Exception:
        return None

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]

plt.rcParams["axes.unicode_minus"] = False

CORE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CORE_DIR.parents[1]
DEFAULT_INVARIANT_RANDOM_IND_MODEL = (
    PACKAGE_ROOT
    / "models"
    / "models_cpbsd_invariant_mb_x_random_ind"
    / "best_state_invariant_cpbsd_mb_x_random_ind_2layer_seed20260514.pt"
)
DEFAULT_INVARIANT_RANDOM_IND_METRICS = (
    PACKAGE_ROOT
    / "models"
    / "models_cpbsd_invariant_mb_x_random_ind"
    / "metrics_invariant_cpbsd_mb_x_random_ind_2layer_seed20260514.json"
)
RANDOM_CORR_FORMULA = "ratio_n ~ Uniform(0, 1); noise_n ~ Normal(0, 0.5); c_n = max(mu_n * ratio_n + noise_n, 0)"


def _load_core02_invariant_gcn():
    core02_path = CORE_DIR / "02_train_cpbsd_gcn_model.py"
    if not core02_path.exists():
        raise FileNotFoundError(f"Missing invariant GCN core script: {core02_path}")
    spec = importlib.util.spec_from_file_location("core02_train_cpbsd_gcn_model", core02_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import invariant GCN core script: {core02_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("core02_train_cpbsd_gcn_model", module)
    spec.loader.exec_module(module)
    return module


# ZFIX repro: this module is imported only for its solvers (solve_mb_restricted /
# solve_bsp / eval_*). The original import-time load of the invariant GCN core is
# not needed here and its dependency (02 script) is not vendored, so guard it.
try:
    _core02 = _load_core02_invariant_gcn()
    DirectionalEdgeScoringGCN = _core02.DirectionalEdgeScoringGCN
    build_invariant_cpbsd_graph = _core02.build_invariant_cpbsd_graph
except Exception:
    _core02 = None
    DirectionalEdgeScoringGCN = None
    build_invariant_cpbsd_graph = None

DEFAULT_HANSON_RANDOM_IND_MODEL = (
    PACKAGE_ROOT
    / "artifacts"
    / "random_valuation"
    / "models"
    / "best_model_edge_cpbsd_mb_x_2layer_seed1000.pt"
)
DEFAULT_HANSON_RANDOM_IND_METRICS = (
    PACKAGE_ROOT
    / "artifacts"
    / "random_valuation"
    / "models"
    / "metrics_edge_cpbsd_mb_x_2layer_seed1000.json"
)

EdgeScoringGCN = None
build_hanson_cpbsd_graph = None


def _load_src_data_module(module_name: str):
    module_path = CORE_DIR / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing src/data module: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


def _ensure_hanson_loaded() -> None:
    """Lazily load the Hanson EdgeScoringGCN class and graph builder.

    Pulled from src/data so inference uses the exact training-time class and
    feature construction; only needed for --graph-variant hanson.
    """
    global EdgeScoringGCN, build_hanson_cpbsd_graph
    if EdgeScoringGCN is not None and build_hanson_cpbsd_graph is not None:
        return
    EdgeScoringGCN = _load_src_data_module("Training_multi_layer_cpbsd_mb_x").EdgeScoringGCN
    build_hanson_cpbsd_graph = _load_src_data_module("cpbsd_hanson_gcn_graph").build_hanson_cpbsd_graph


DIST_FAMILIES = {"exponential", "logit", "lognormal", "normal", "uniform"}

HETEROGENEITY = {"none", "partial", "full"}

COST_SCENARIOS = {"zero", "random_ind", "random_corr"}

@dataclass
class CPBSDSetup:
    n_products: int
    k_samples: int
    dist_family: str
    rho: float
    heterogeneity: str
    cost_scenario: str
    seed: int

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

def _norm_ppf(u: np.ndarray) -> np.ndarray:
    # Inverse standard normal CDF via erfinv: Phi^{-1}(u) = sqrt(2) * erfinv(2u-1)
    t = torch.as_tensor(u, dtype=torch.float64)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * t - 1.0)
    return z.numpy()

def _toeplitz_corr(n: int, rho: float) -> np.ndarray:
    if abs(float(rho)) < 1e-12:
        return np.eye(n)
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])

def valuation_means(n: int, heterogeneity: str) -> np.ndarray:
    if heterogeneity == "none":
        return np.ones(n)

    if heterogeneity == "partial":
        n1 = int(0.4 * n)
        means = np.ones(n)
        means[n1:] = 5.0
        return means

    # full
    if n == 1:
        return np.array([1.0])
    idx = np.arange(n)
    return 1.0 + 9.0 * idx / (n - 1)

def production_costs(n: int, heterogeneity: str, scenario: str, rng=None) -> np.ndarray:
    idx = np.arange(n)

    if scenario == "zero":
        return np.zeros(n)

    if scenario == "random_ind":
        # Independent random costs, same scale as valuation means [1, 10]
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(0.0, 10.0, size=n)

    if scenario == "random_corr":
        # Correlated with valuation means, deterministic from the instance seed.
        if rng is None:
            rng = np.random.default_rng()
        means = valuation_means(n, heterogeneity)
        ratio = rng.uniform(0.0, 1.0, size=n)
        noise = rng.normal(0, 0.5, size=n)
        return np.maximum(means * ratio + noise, 0.0)

    if heterogeneity == "none":
        if scenario == "hvhm":
            return np.full(n, 0.8)
        # hvlm
        return 0.1 + 1.5 * idx / max(1, (n - 1))

    if heterogeneity == "partial":
        n1 = int(0.4 * n)
        c = np.zeros(n)
        if scenario == "hvhm":
            c[:n1] = 0.8
            c[n1:] = 4.4
            return c

        # hvlm (piecewise in Table 2)
        c[:n1] = 0.1 + 1.5 * np.arange(n1) / max(1, (n - 1))
        n2 = n - n1
        c[n1:] = 4.1 + 1.5 * np.arange(n2) / max(1, (n2 - 1))
        return c

    # full heterogeneity
    if scenario == "hvhm":
        return 0.8 + 8.6 * idx / max(1, (n - 1))
    return 0.1 + 10.5 * idx / max(1, (n - 1))

def sample_valuations(k: int, means: np.ndarray, family: str, rho: float, rng: np.random.Generator) -> np.ndarray:
    n = len(means)
    corr = _toeplitz_corr(n, rho)
    z = rng.multivariate_normal(mean=np.zeros(n), cov=corr, size=k)
    u = np.clip(_norm_cdf(z), 1e-10, 1 - 1e-10)

    if family == "exponential":
        v = -means * np.log(1.0 - u)
    elif family == "logit":
        # Gumbel(loc, scale) with scale=0.25 and loc=E[V]-0.577*scale
        scale = 0.25
        loc = means - 0.577 * scale
        v = loc + (-scale * np.log(-np.log(u)))
    elif family == "lognormal":
        sigma = 0.5  # sqrt(0.25)
        mu = np.log(means) - 0.5 * (sigma**2)
        v = np.exp(mu + sigma * _norm_ppf(u))
    elif family == "normal":
        sigma = 0.5
        v = means + sigma * _norm_ppf(u)
    elif family == "uniform":
        v = (means - 1.0) + 2.0 * u
    else:
        raise ValueError(f"Unsupported family: {family}")

    # Keep numerical stability and economic meaning
    return np.maximum(v, 0.0)

def generate_cpbsd_instance(setup: CPBSDSetup) -> Dict:
    rng = np.random.default_rng(setup.seed)

    means = valuation_means(setup.n_products, setup.heterogeneity)
    costs = production_costs(setup.n_products, setup.heterogeneity, setup.cost_scenario, rng=rng)
    valuations = sample_valuations(
        k=setup.k_samples,
        means=means,
        family=setup.dist_family,
        rho=setup.rho,
        rng=rng,
    )

    potential_surplus = valuations - costs[None, :]

    # Minimal formulation sanity check for CPBSD payment rule
    # Pay(S) = sum_{n in S} p_n - |S| d_|S|
    p_demo = rng.uniform(0.5, 2.0, size=setup.n_products)
    d_demo = np.zeros(setup.n_products + 1)
    if setup.n_products >= 1:
        d_demo[1:] = rng.uniform(0.0, 0.3, size=setup.n_products)

    S = rng.integers(0, 2, size=setup.n_products)
    s = int(S.sum())
    pay_by_definition = float((p_demo * S).sum() - s * d_demo[s])

    return {
        "setup": asdict(setup),
        "means_E_V": means,
        "production_cost_c": costs,
        "valuation_samples_V": valuations,
        "potential_surplus_Z": potential_surplus,
        "cpbsd_definition_check": {
            "S_binary": S,
            "bundle_size_s": s,
            "component_price_p": p_demo,
            "size_discount_d": d_demo,
            "payment_sum_p_minus_s_d": pay_by_definition,
        },
    }

def generate_batch(
    out_dir: str,
    n_products: int,
    k_samples: int,
    dist_family: str,
    rho: float,
    heterogeneity: str,
    cost_scenario: str,
    n_instances: int,
    seed: int,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for i in range(n_instances):
        setup = CPBSDSetup(
            n_products=n_products,
            k_samples=k_samples,
            dist_family=dist_family,
            rho=rho,
            heterogeneity=heterogeneity,
            cost_scenario=cost_scenario,
            seed=seed + i,
        )
        data = generate_cpbsd_instance(setup)

        fname = (
            f"cpbsd_instance_{i+1:03d}_N{n_products}_K{k_samples}_"
            f"{dist_family}_rho{rho}_{heterogeneity}_{cost_scenario}.msgpack"
        )
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            msgpack.dump(data, f, default=mnp.encode)
        paths.append(fpath)

    return paths


if not os.environ.get("GRB_LICENSE_FILE") and Path.home().joinpath('.gurobi', 'gurobi.lic').exists():
    os.environ["GRB_LICENSE_FILE"] = str(Path.home().joinpath('.gurobi', 'gurobi.lic'))

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
    mip_gap=1e-3,
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
        "mip_gap": _safe_float_attr(m, "MIPGap") if m.SolCount > 0 else None,
        "best_bound": _safe_float_attr(m, "ObjBound") if m.SolCount > 0 else None,
    }

    if m.SolCount > 0:
        out["objective"] = float(m.ObjVal)
        out["p"] = [p[n].X for n in N_idx]
        out["d"] = [0.0] + [d[s].X for s in S]

    return out


if not os.environ.get("GRB_LICENSE_FILE") and Path.home().joinpath(".gurobi", "gurobi.lic").exists():
    os.environ["GRB_LICENSE_FILE"] = str(Path.home().joinpath(".gurobi", "gurobi.lic"))

MB_FORMULATION_VERSION = 7

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

def _bundle_row_mask(row: np.ndarray) -> int:
    mask = 0
    for idx, bit in enumerate(row.astype(int).tolist()):
        if bit:
            mask |= 1 << idx
    return mask

def _add_hanson_k_way_cover_subadditivity(
    model: gp.Model,
    p,
    assortments: np.ndarray,
    nonempty_idx: List[int],
) -> int:
    """Add Hanson-aligned free-disposal and minimal K-way cover constraints."""
    masks = {idx: _bundle_row_mask(assortments[idx]) for idx in nonempty_idx}
    added_constraints = set()
    subadd_ctr = 0
    item_bits = [1 << bit_idx for bit_idx in range(assortments.shape[1])]

    def add_constraint(target_idx: int, cover: Iterable[int]) -> None:
        nonlocal subadd_ctr
        ordered_cover = tuple(sorted(cover))
        if not ordered_cover:
            return
        key = (target_idx, ordered_cover)
        if key in added_constraints:
            return
        model.addConstr(
            p[target_idx] <= gp.quicksum(p[bundle_idx] for bundle_idx in ordered_cover),
            name=f"hanson_kway_sa_{subadd_ctr}",
        )
        added_constraints.add(key)
        subadd_ctr += 1

    for target_idx in nonempty_idx:
        target_mask = masks[target_idx]
        singleton_covers = []
        multi_cover_candidates = []

        for bundle_idx in nonempty_idx:
            if bundle_idx == target_idx:
                continue
            contribution = masks[bundle_idx] & target_mask
            if contribution == 0:
                continue
            if contribution == target_mask:
                singleton_covers.append(bundle_idx)
            else:
                multi_cover_candidates.append((bundle_idx, contribution))

        for bundle_idx in singleton_covers:
            add_constraint(target_idx, (bundle_idx,))

        if not multi_cover_candidates:
            continue

        item_to_candidate_indices = {bit: [] for bit in item_bits if target_mask & bit}
        for idx, (_, contribution) in enumerate(multi_cover_candidates):
            remaining = contribution
            while remaining:
                lowbit = remaining & -remaining
                item_to_candidate_indices[lowbit].append(idx)
                remaining ^= lowbit

        if any(len(indices) == 0 for indices in item_to_candidate_indices.values()):
            continue

        chosen: List[Tuple[int, int]] = []
        item_cover_count = {bit: 0 for bit in item_to_candidate_indices}

        def dfs(available_indices: Tuple[int, ...], covered_mask: int) -> None:
            if covered_mask == target_mask:
                for _, contribution in chosen:
                    if all(
                        item_cover_count[bit] > 1
                        for bit in item_cover_count
                        if contribution & bit
                    ):
                        return
                add_constraint(target_idx, tuple(bundle_idx for bundle_idx, _ in chosen))
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

            for option_pos, idx in enumerate(branch_options or []):
                bundle_idx, contribution = multi_cover_candidates[idx]
                chosen.append((bundle_idx, contribution))
                updated_items = []
                remaining_contribution = contribution
                while remaining_contribution:
                    lowbit = remaining_contribution & -remaining_contribution
                    item_cover_count[lowbit] += 1
                    updated_items.append(lowbit)
                    remaining_contribution ^= lowbit
                blocked = set(branch_options[: option_pos + 1])
                next_available = tuple(j for j in available_indices if j not in blocked)
                dfs(next_available, covered_mask | contribution)
                for bit in updated_items:
                    item_cover_count[bit] -= 1
                chosen.pop()

        dfs(tuple(range(len(multi_cover_candidates))), 0)

    return subadd_ctr

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

def solve_pbdc(v_kn: np.ndarray, c_n: np.ndarray) -> dict:
    """Pure bundling with disposal for cost from the CPBSD paper.

    Let C=sum_n c_n and t=pPBDC-C.  Customer k buys iff
    H_k=sum_n max(v_kn-c_n, 0) >= t, and each buying customer contributes
    profit t.  We enumerate t in unique({H_k} union {0}) and maximize average
    in-sample profit, breaking exact objective ties toward larger t.
    """
    t0 = time.time()
    v_kn = np.asarray(v_kn, dtype=float)
    c_n = np.asarray(c_n, dtype=float)
    h_k = np.maximum(v_kn - c_n[None, :], 0.0).sum(axis=1)
    candidates = np.unique(np.concatenate([h_k, np.array([0.0], dtype=float)]))
    best_t = 0.0
    best_revenue = -math.inf
    eps = 1e-12
    for t in candidates:
        t_float = float(t)
        revenue = t_float * float(np.mean(h_k >= t_float))
        if revenue > best_revenue + eps or (abs(revenue - best_revenue) <= eps and t_float > best_t):
            best_t = t_float
            best_revenue = revenue
    t1 = time.time()
    return {
        "solver_status": 2,
        "feasible": True,
        "runtime": t1 - t0,
        "wall_time": t1 - t0,
        "K": int(v_kn.shape[0]),
        "N": int(v_kn.shape[1]),
        "objective": float(best_revenue),
        "revenue_in_sample": float(best_revenue),
        "t_star": float(best_t),
        "pbdc_margin_t": float(best_t),
        "total_production_cost": float(np.sum(c_n)),
        "grand_bundle_price": float(np.sum(c_n) + best_t),
        "candidate_t_count": int(len(candidates)),
        "candidate_t_values": [float(x) for x in candidates.tolist()],
        "buyer_count_in_sample": int(np.sum(h_k >= best_t)),
        "buyer_fraction_in_sample": float(np.mean(h_k >= best_t)),
        "tie_break_rule": "maximize average profit, then prefer larger t",
    }

def eval_pbdc_policy(v_eval: np.ndarray, c_n: np.ndarray, t_star: float) -> float:
    v_eval = np.asarray(v_eval, dtype=float)
    c_n = np.asarray(c_n, dtype=float)
    t_star = float(t_star)
    h_k = np.maximum(v_eval - c_n[None, :], 0.0).sum(axis=1)
    return t_star * float(np.mean(h_k >= t_star))

def solve_mb(
    v_kn: np.ndarray,
    c_n: np.ndarray,
    time_limit: float = 300.0,
    mip_gap: float = 1e-3,
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
    mip_gap: float = 1e-3,
    output_flag: int = 0,
    threads: int = 0,
    subadditivity_mode: str = "hanson_empty_kway",
    profit_lb: float = -GRB.INFINITY,  # ZFIX: -inf (correct). Pass 0.0 to reproduce the bug.
):
    k_count, n_products = v_kn.shape
    assortments = ensure_empty_bundle(assortments, n_products)
    assortments = np.unique(assortments.astype(int), axis=0)
    bundle_count = assortments.shape[0]
    empty_matches = np.where(np.all(assortments == 0, axis=1))[0]
    if len(empty_matches) != 1:
        raise ValueError("Expected exactly one empty bundle after normalization.")
    empty_idx = int(empty_matches[0])
    bundle_cost = (assortments @ c_n).reshape(1, -1)
    costs = np.repeat(bundle_cost, k_count, axis=0)
    revenues = v_kn @ assortments.T
    revenue_ub = max(float(np.max(revenues)), 1e-9)
    weights = np.ones((k_count, 1), dtype=float) / k_count
    active_idx = list(range(bundle_count))
    nonempty_idx = [idx for idx in active_idx if idx != empty_idx]
    segment_idx = range(k_count)

    model = gp.Model("Bundle_MILP_hanson_empty_kway")
    p = model.addVars(active_idx, vtype=GRB.CONTINUOUS, lb=0.0, name="p")
    theta = model.addVars(k_count, active_idx, vtype=GRB.BINARY, name="theta")
    surplus = model.addVars(k_count, vtype=GRB.CONTINUOUS, lb=0.0, name="w")
    s_terms = model.addVars(k_count, active_idx, vtype=GRB.CONTINUOUS, name="S")
    profit = model.addVars(k_count, active_idx, vtype=GRB.CONTINUOUS, lb=profit_lb, name="Z")
    payment = model.addVars(k_count, active_idx, vtype=GRB.CONTINUOUS, lb=0.0, name="q")

    model.setObjective(
        gp.quicksum(weights[k, 0] * profit[k, i] for k in segment_idx for i in active_idx),
        GRB.MAXIMIZE,
    )

    model.addConstrs((surplus[k] >= revenues[k, i] - p[i] for i in active_idx for k in segment_idx), name="surplus_lb")
    subadd_constraint_count = _add_hanson_k_way_cover_subadditivity(model, p, assortments, nonempty_idx)
    model.addConstr(p[empty_idx] == 0.0, name="empty_price")
    model.addConstrs((s_terms[k, empty_idx] == 0.0 for k in segment_idx), name="empty_surplus_term")

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
    model.addConstrs((gp.quicksum(theta[k, i] for i in active_idx) == 1 for k in segment_idx), name="one_bundle_with_empty")

    model.setParam("OutputFlag", output_flag)
    model.setParam("MIPGap", mip_gap)
    model.setParam("TimeLimit", time_limit)
    model.setParam("DualReductions", 0)
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
        "mip_gap": _safe_float_attr(model, "MIPGap") if model.SolCount > 0 else None,
        "objective": float(model.ObjVal) if model.SolCount > 0 else None,
        "revenue_in_sample": None,
        "model_num_vars": int(model.NumVars),
        "model_num_binvars": int(model.NumBinVars),
        "model_num_constrs": int(model.NumConstrs),
        "bundle_space_size": bundle_count,
        "subadditivity_mode": "hanson_empty_kway",
        "requested_subadditivity_mode": subadditivity_mode,
        "subadd_constraint_count": int(subadd_constraint_count),
        "empty_bundle_idx": empty_idx,
        "outside_option_mode": "explicit_empty_exactly_one",
        "policy_scope": "full_bundle_prices" if model.SolCount > 0 else "missing",
        "bundle_prices": None,
        "bundle_prices_full": None,
        "bundle_prices_selected": None,
        "chosen_bundle_idx_by_customer": None,
        "chosen_product_matrix": None,
        "assortments": assortments,
    }
    if model.SolCount > 0:
        bundle_prices_full = {}
        bundle_prices_selected = {}
        for i in active_idx:
            bundle_prices_full[i] = float(p[i].X)
            chosen_any = any(theta[k, i].X >= 1 - 1e-2 for k in segment_idx)
            if chosen_any:
                bundle_prices_selected[i] = float(p[i].X)
        chosen_bundle_idx_by_customer = []
        chosen_product_matrix = np.zeros((k_count, n_products), dtype=int)
        for k in segment_idx:
            chosen_bundle = empty_idx
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
    mip_gap: float = 1e-3,
    output_flag: int = 0,
    threads: int = 0,
    profit_lb: float = -GRB.INFINITY,  # ZFIX: -inf (correct). Pass 0.0 to reproduce the bug.
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
    profit = model.addVars(k_count, max_size + 1, vtype=GRB.CONTINUOUS, lb=profit_lb, name="Z")
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
        "mip_gap": _safe_float_attr(model, "MIPGap") if model.SolCount > 0 else None,
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def load_instance(path: Path) -> Tuple[dict, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        obj = msgpack.load(f, object_hook=mnp.decode, strict_map_key=False)
    return obj, np.asarray(obj["valuation_samples_V"], dtype=float), np.asarray(obj["production_cost_c"], dtype=float)

def build_graph(v_kn: np.ndarray, c_n: np.ndarray, variant: str = "invariant") -> Data:
    if variant == "hanson":
        _ensure_hanson_loaded()
        data, _ = build_hanson_cpbsd_graph(v_kn=v_kn, c_n=c_n)
    else:
        data, _ = build_invariant_cpbsd_graph(v_kn=v_kn, c_n=c_n)
    return data

def resolve_torch_device(device_name: str) -> torch.device:
    requested = device_name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)

def infer_probabilities(model, graph_data: Data, device: torch.device) -> np.ndarray:
    graph_data = graph_data.to(device)
    with torch.no_grad():
        raw_out = model(graph_data)
    if isinstance(raw_out, dict):
        if "logit_matrix" in raw_out:
            logits_nm = raw_out["logit_matrix"].detach().cpu().numpy()
        elif "edge_logits" in raw_out:
            n = int(graph_data.product_num)
            m = int(graph_data.segment_num)
            logits_nm = raw_out["edge_logits"].detach().cpu().numpy().reshape(n, m)
        else:
            raise ValueError(f"Unexpected output keys: {list(raw_out.keys())}")
    else:
        raise TypeError(f"Unexpected model output type: {type(raw_out)}")
    return torch.sigmoid(torch.tensor(logits_nm)).numpy().T

def build_fcp_candidate_bundles(prob: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, int]:
    customer_bundles = []
    for k in range(prob.shape[0]):
        bundle = (prob[k] >= threshold).astype(int)
        customer_bundles.append(bundle.tolist())

    unique_candidates = np.array(sorted({tuple(bundle) for bundle in customer_bundles}), dtype=int)
    if unique_candidates.size == 0:
        unique_candidates = np.zeros((0, prob.shape[1]), dtype=int)
    return unique_candidates, len(customer_bundles)

def evaluate_revenue(v_kn: np.ndarray, c_n: np.ndarray, p: np.ndarray, d: np.ndarray) -> float:
    k_count, n_products = v_kn.shape
    total = 0.0
    for k in range(k_count):
        best_surplus = 0.0
        best_idx = None
        best_s = 0
        for s in range(1, n_products + 1):
            util = v_kn[k] - p + d[s]
            idx = np.argpartition(util, -s)[-s:]
            surplus = float(util[idx].sum())
            if surplus > best_surplus:
                best_surplus = surplus
                best_idx = idx
                best_s = s
        if best_surplus <= 0 or best_idx is None:
            continue
        profit = float((p[best_idx] - c_n[best_idx]).sum() - best_s * d[best_s])
        total += profit
    return total / k_count

def out_of_sample_revenue(setup: dict, c_n: np.ndarray, p: np.ndarray, d: np.ndarray, out_k: int = 5000) -> float:
    rng = np.random.default_rng(int(setup["seed"]) + 99991)
    means = valuation_means(int(setup["n_products"]), setup["heterogeneity"])
    v_out = sample_valuations(
        k=out_k,
        means=means,
        family=setup["dist_family"],
        rho=float(setup["rho"]),
        rng=rng,
    )
    return evaluate_revenue(v_out, c_n, p, d)

def result_to_row(
    *,
    instance_id: str,
    seed: int,
    n_products: int,
    k_samples: int,
    method: str,
    result_path: Path,
    in_sample_revenue: float | None,
    out_sample_revenue: float | None,
    runtime: float | None,
    status_code: int,
    extra: Dict[str, object],
) -> Dict[str, object]:
    row = {
        "instance_id": instance_id,
        "seed": seed,
        "n": n_products,
        "k": k_samples,
        "method": method,
        "revenue_in_sample": in_sample_revenue,
        "revenue_out_sample": out_sample_revenue,
        "solver_runtime": runtime,
        "status_code": status_code,
        "result_path": str(result_path),
    }
    row.update(extra)
    return row


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_for_cost(root: Path, cost: str) -> Path:
    if cost not in COST_SCENARIOS:
        raise ValueError(f"Unsupported cost setup for this experiment: {cost}")
    return DEFAULT_INVARIANT_RANDOM_IND_MODEL


def load_model(model_path: Path, device: torch.device, variant: str = "invariant"):
    setattr(sys.modules["__main__"], "DirectionalEdgeScoringGCN", DirectionalEdgeScoringGCN)
    if variant == "hanson":
        # Hanson checkpoints are full-module pickles referencing __main__.EdgeScoringGCN,
        # so the real class must be registered before torch.load.
        _ensure_hanson_loaded()
        setattr(sys.modules["__main__"], "EdgeScoringGCN", EdgeScoringGCN)
    else:
        setattr(sys.modules["__main__"], "EdgeScoringGCN", DirectionalEdgeScoringGCN)
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        model = loaded
    elif isinstance(loaded, dict) and "model_config" in loaded and "state_dict" in loaded:
        model = DirectionalEdgeScoringGCN(**loaded["model_config"])
        model.load_state_dict(loaded["state_dict"], strict=True)
    elif isinstance(loaded, dict):
        raise TypeError("Raw legacy state_dict checkpoints are not supported; pass an invariant checkpoint saved as {model_config, state_dict}.")
    else:
        raise TypeError(f'Unsupported checkpoint type: {type(loaded)}')
    is_directional = model.__class__.__name__ == "DirectionalEdgeScoringGCN"
    if variant == "hanson" and is_directional:
        raise SystemExit(
            "--graph-variant hanson needs a Hanson EdgeScoringGCN checkpoint, "
            "but the model is DirectionalEdgeScoringGCN. Pass a Hanson --model-path."
        )
    if variant == "invariant" and not is_directional:
        raise SystemExit(
            "--graph-variant invariant needs a DirectionalEdgeScoringGCN checkpoint, "
            f"but the model is {model.__class__.__name__}. Pass --graph-variant hanson."
        )
    model.to(device)
    model.eval()
    return model


def run_one_setting(args: argparse.Namespace, *, cost: str, seed: int, model_path: Path) -> List[dict]:
    setting_root = args.out_root / f'normal_rho0_full_{cost}' / f'seed_{seed}'
    inst_dir = setting_root / 'instances'
    result_dir = setting_root / 'results'
    ensure_dir(inst_dir)
    ensure_dir(result_dir)
    if len(list(inst_dir.glob('*.msgpack'))) < args.instances:
        generate_batch(str(inst_dir), args.N, args.K, 'normal', 0.0, 'full', cost, args.instances, seed)
    device = resolve_torch_device(args.device)
    model = load_model(model_path, device, variant=args.graph_variant)
    rows: List[Dict[str, object]] = []
    for instance_path in sorted(inst_dir.glob('*.msgpack'))[:args.instances]:
        obj, v_kn, c_n = load_instance(instance_path)
        setup = obj['setup']
        instance_id = instance_path.stem
        graph_data = build_graph(v_kn, c_n, variant=args.graph_variant)
        infer_t0 = time.time()
        prob = infer_probabilities(model, graph_data, device)
        infer_t1 = time.time()
        cand_t0 = time.time()
        candidate_assortments, raw_customer_bundle_count = build_fcp_candidate_bundles(prob, threshold=args.threshold)
        cand_t1 = time.time()
        fcp_t0 = time.time()
        fcp_res = solve_mb_restricted(v_kn, c_n, candidate_assortments, time_limit=args.time_limit_fcp_mb, mip_gap=args.mip_gap, output_flag=args.output_flag, subadditivity_mode='hanson_empty_kway')
        fcp_t1 = time.time()
        fcp_json = result_dir / f'{instance_id}__fcp_pruned_mb.json'
        fcp_json.write_text(json.dumps(fcp_res, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')
        v_out = sample_valuations(args.out_samples, valuation_means(args.N, 'full'), 'normal', 0.0, np.random.default_rng(int(setup['seed']) + 99991))
        fcp_policy = fcp_res.get('bundle_prices_full') or fcp_res.get('bundle_prices') or {}
        fcp_out = eval_mb_policy(v_out, c_n, fcp_policy, np.asarray(fcp_res['assortments'], dtype=int)) if fcp_res.get('feasible') and fcp_policy else None
        rows.append(result_to_row(instance_id=instance_id, seed=int(setup['seed']), n_products=args.N, k_samples=args.K, method='FCP-pruned-MB', result_path=fcp_json, in_sample_revenue=fcp_res.get('objective'), out_sample_revenue=fcp_out, runtime=fcp_t1 - fcp_t0, status_code=int(fcp_res.get('solver_status', -1)), extra={'cost': cost, 'gcn_inference_time': infer_t1 - infer_t0, 'candidate_generation_time': cand_t1 - cand_t0, 'bundle_space_size': int(fcp_res.get('bundle_space_size', len(candidate_assortments))), 'bundle_space_fraction': float(fcp_res.get('bundle_space_size', len(candidate_assortments))) / float(2 ** args.N), 'full_bundle_space_size': int(2 ** args.N), 'raw_customer_bundle_count': raw_customer_bundle_count, 'unique_threshold_bundle_count': int(candidate_assortments.shape[0]), 'threshold': args.threshold, 'out_samples': args.out_samples, 'mip_gap_requested': args.mip_gap, 'model_path': str(model_path), 'model_metrics_path': str(args.model_metrics_path), 'model_class': model.__class__.__name__, 'graph_builder': 'build_hanson_cpbsd_graph' if args.graph_variant == 'hanson' else 'build_invariant_cpbsd_graph', 'graph_variant': args.graph_variant, 'random_corr_formula': RANDOM_CORR_FORMULA if cost == 'random_corr' else ''}))
        if args.fcp_only:
            continue
        pbdc_res = solve_pbdc(v_kn, c_n)
        pbdc_json = result_dir / f'{instance_id}__pbdc.json'
        pbdc_out = eval_pbdc_policy(v_out, c_n, pbdc_res['t_star'])
        pbdc_res['revenue_out_sample'] = pbdc_out
        pbdc_res['out_samples'] = args.out_samples
        pbdc_json.write_text(json.dumps(pbdc_res, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')
        rows.append(result_to_row(instance_id=instance_id, seed=int(setup['seed']), n_products=args.N, k_samples=args.K, method='PBDC', result_path=pbdc_json, in_sample_revenue=pbdc_res.get('objective'), out_sample_revenue=pbdc_out, runtime=pbdc_res.get('runtime'), status_code=int(pbdc_res.get('solver_status', 2)), extra={'cost': cost, 'full_bundle_space_size': int(2 ** args.N), 'out_samples': args.out_samples, 'mip_gap_requested': args.mip_gap, 'pbdc_t_star': pbdc_res['t_star'], 'grand_bundle_price': pbdc_res['grand_bundle_price'], 'pbdc_buyer_fraction_in_sample': pbdc_res['buyer_fraction_in_sample'], 'pbdc_tie_break_rule': pbdc_res['tie_break_rule'], 'random_corr_formula': RANDOM_CORR_FORMULA if cost == 'random_corr' else ''}))
        bsp_t0 = time.time()
        bsp_res = solve_bsp(v_kn, c_n, time_limit=args.time_limit_bsp, mip_gap=args.mip_gap, output_flag=args.output_flag)
        bsp_t1 = time.time()
        bsp_json = result_dir / f'{instance_id}__bsp.json'
        bsp_json.write_text(json.dumps(bsp_res, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')
        bsp_out = eval_bsp_policy(v_out, c_n, bsp_res.get('size_prices') or {}) if bsp_res.get('feasible') and bsp_res.get('size_prices') else None
        rows.append(result_to_row(instance_id=instance_id, seed=int(setup['seed']), n_products=args.N, k_samples=args.K, method='BSP', result_path=bsp_json, in_sample_revenue=bsp_res.get('objective'), out_sample_revenue=bsp_out, runtime=bsp_t1 - bsp_t0, status_code=int(bsp_res.get('solver_status', 2 if bsp_res.get('feasible') else 3)), extra={'cost': cost, 'full_bundle_space_size': int(2 ** args.N), 'out_samples': args.out_samples, 'mip_gap_requested': args.mip_gap}))
        cpbsd_t0 = time.time()
        cpbsd_a_res = solve_cpbsd_a(v_kn, c_n, mip_gap=args.mip_gap, time_limit=args.time_limit_cpbsd_a, output_flag=args.output_flag)
        cpbsd_t1 = time.time()
        cpbsd_json = result_dir / f'{instance_id}__cpbsd_a.json'
        cpbsd_json.write_text(json.dumps(cpbsd_a_res, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')
        cpbsd_out = None
        if cpbsd_a_res.get('sol_count', 0) > 0:
            cpbsd_out = out_of_sample_revenue(setup, c_n, np.asarray(cpbsd_a_res.get('p', []), dtype=float), np.asarray(cpbsd_a_res.get('d', []), dtype=float), out_k=args.out_samples)
        rows.append(result_to_row(instance_id=instance_id, seed=int(setup['seed']), n_products=args.N, k_samples=args.K, method='CPBSD-A', result_path=cpbsd_json, in_sample_revenue=cpbsd_a_res.get('objective'), out_sample_revenue=cpbsd_out, runtime=cpbsd_t1 - cpbsd_t0, status_code=int(cpbsd_a_res.get('solver_status', -1)), extra={'cost': cost, 'full_bundle_space_size': int(2 ** args.N), 'out_samples': args.out_samples, 'mip_gap_requested': args.mip_gap}))
    return rows


def main() -> None:
    root = package_root()
    parser = argparse.ArgumentParser(description='Compare FCP-pruned-MB, PBDC, BSP, and CPBSD-A on normal/full/rho=0 zero/random_ind/random_corr setups with 5 seeds using the invariant random_ind GCN.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out-root', type=Path, default=root / 'work/core04_invariant_randind_three_costs_20260519', help='Fresh output root.')
    parser.add_argument('--N', type=int, default=10, help='Number of products.')
    parser.add_argument('--K', type=int, default=50, help='Number of in-sample customers.')
    parser.add_argument('--instances', type=int, default=1, help='Instances per seed/cost.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[20260413, 20260414, 20260415, 20260416, 20260417], help='Experiment seeds.')
    parser.add_argument('--costs', nargs='+', default=['zero', 'random_ind', 'random_corr'], choices=['zero', 'random_ind', 'random_corr'], help='Cost setups; hvhm is intentionally excluded.')
    parser.add_argument('--model-path', type=Path, default=DEFAULT_INVARIANT_RANDOM_IND_MODEL, help='Invariant random_ind checkpoint used for every cost setup.')
    parser.add_argument('--model-metrics-path', type=Path, default=DEFAULT_INVARIANT_RANDOM_IND_METRICS, help='Metrics file matching the invariant random_ind checkpoint.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help='Torch device.')
    parser.add_argument('--threshold', type=float, default=0.5, help='GCN edge probability threshold.')
    parser.add_argument('--out-samples', type=int, default=5000, help='Out-of-sample valuation draws.')
    parser.add_argument('--time-limit-fcp-mb', type=float, default=300.0, help='FCP-pruned-MB solver time limit in seconds.')
    parser.add_argument('--time-limit-bsp', type=float, default=300.0, help='BSP solver time limit in seconds.')
    parser.add_argument('--time-limit-cpbsd-a', type=float, default=300.0, help='CPBSD-A solver time limit in seconds.')
    parser.add_argument('--mip-gap', type=float, default=1e-3, help='Gurobi MIPGap for all solved models.')
    parser.add_argument('--output-flag', type=int, default=0, help='Gurobi OutputFlag.')
    parser.add_argument('--graph-variant', choices=['invariant', 'hanson'], default='invariant', help='FCP model family + graph builder.')
    parser.add_argument('--fcp-only', action='store_true', help='Run only FCP-pruned-MB; skip PBDC/BSP/CPBSD-A (reuse audited baselines).')
    args = parser.parse_args()
    if args.graph_variant == 'hanson':
        # Swap untouched invariant defaults to the Hanson random_ind artifacts.
        if args.model_path == DEFAULT_INVARIANT_RANDOM_IND_MODEL:
            args.model_path = DEFAULT_HANSON_RANDOM_IND_MODEL
        if args.model_metrics_path == DEFAULT_INVARIANT_RANDOM_IND_METRICS:
            args.model_metrics_path = DEFAULT_HANSON_RANDOM_IND_METRICS
    if not args.model_path.exists():
        raise FileNotFoundError(f"random_ind checkpoint not found: {args.model_path}")
    if not args.model_metrics_path.exists():
        raise FileNotFoundError(f"random_ind metrics file not found: {args.model_metrics_path}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cost in args.costs:
        mpath = args.model_path
        for seed in args.seeds:
            print(f'Running cost={cost} seed={seed} model={mpath}', flush=True)
            all_rows.extend(run_one_setting(args, cost=cost, seed=seed, model_path=mpath))
    by_instance: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in all_rows:
        key = f"{row['cost']}/{row['seed']}/{row['instance_id']}"
        by_instance.setdefault(key, {})[str(row['method'])] = row
    for methods in by_instance.values():
        bsp_rev = methods.get('BSP', {}).get('revenue_in_sample')
        cpbsd_rev = methods.get('CPBSD-A', {}).get('revenue_in_sample')
        for row in methods.values():
            row['ratio_to_bsp'] = row['revenue_in_sample'] / bsp_rev if bsp_rev not in (None, 0) and row.get('revenue_in_sample') is not None else None
            row['ratio_to_cpbsd_a'] = row['revenue_in_sample'] / cpbsd_rev if cpbsd_rev not in (None, 0) and row.get('revenue_in_sample') is not None else None
    json_path = args.out_root / 'comparison_summary.json'
    csv_path = args.out_root / 'comparison_summary.csv'
    json_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')
    fieldnames = sorted({key for row in all_rows for key in row.keys()}) if all_rows else []
    if fieldnames:
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    print(json.dumps({'json': str(json_path), 'csv': str(csv_path), 'rows': len(all_rows)}, indent=2))


if __name__ == '__main__':
    main()
