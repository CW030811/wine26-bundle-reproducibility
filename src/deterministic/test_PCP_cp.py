"""
test_PCP_cp.py  -  PCP evaluation with cutting-plane subadditivity.

Key difference from test_PCP.py:
  - Multi-bundle price-subadditivity cuts are generated ON DEMAND via
    Gurobi lazy constraints instead of being pre-enumerated upfront.
  - Separation is exact and specialized to PCP progressive-chain structure:
    for each target bundle k, the callback first uses a segment-chain sparse DP
    to search for the cheapest violated cover, rather than enumerating all
    minimal covers ahead of time.
  - Within-segment chain monotonicity is still added upfront, and singleton
    monotonicity cuts are also added upfront, except those already implied by
    the chain constraints.
  - Bundle-matrix construction and progressive-bundle generation use bitmask
    operations throughout to reduce Python overhead during model building.
  - Diagnostic timing / callback statistics are printed so we can see whether
    runtime is dominated by preprocessing, model construction, separation, or
    the solver itself.
  - Mathematically, this remains exact for the same PCP cut family used here:
    the solve terminates when no violated lazy subadditivity cut exists at the
    incumbent solutions encountered by the solver.
"""

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
import math

from torch_geometric.nn import GENConv
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import gurobipy as gp
from gurobipy import GRB

from test_FCP import revenue_ratio as fcp_revenue_ratio


# ---------------------------------------------------------------------------
# GCN model
# ---------------------------------------------------------------------------

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
        out = {'edge_logits': logits}

        if hasattr(data, 'product_num') and hasattr(data, 'segment_num'):
            try:
                n = int(data.product_num)
                m = int(data.segment_num)
                if logits.numel() == n * m:
                    out['logit_matrix'] = logits.view(n, m)
            except Exception:
                pass

        return out


if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([EdgeScoringGCN])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def process_data(file_path):
    """
    Process data file and return graph data and related parameters.
    Supports both old (stored cs/Rs) and new (on-demand) formats.
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

    if 'cs' in data and 'Rs' in data:
        cs = data['cs']
        Rs = data['Rs']
    else:
        cs = None
        Rs = None

    node_num = product_num + segment_num
    feature = np.zeros((node_num, 4), dtype=float)

    if isinstance(unit_cs, np.ndarray):
        uc = unit_cs[0] if unit_cs.ndim == 2 else unit_cs
        uc = np.asarray(uc).reshape(-1)[:product_num]
    else:
        uc = np.asarray(unit_cs).reshape(-1)[:product_num]
    feature[:product_num, 0] = uc

    uu_avg = np.average(unit_us, axis=0)
    feature[:product_num, 1] = np.asarray(uu_avg).reshape(-1)[:product_num]

    if isinstance(Ns, np.ndarray) and Ns.ndim == 2:
        ns_vec = Ns[:, 0]
    else:
        ns_vec = np.asarray(Ns).reshape(-1)
    ns_vec = ns_vec[:segment_num]
    feature[product_num:, 2] = ns_vec

    if isinstance(ship_cs, np.ndarray) and ship_cs.ndim == 2:
        sc_vec = ship_cs[:segment_num, 0]
    else:
        sc_vec = np.asarray(ship_cs).reshape(-1)[:segment_num]
    feature[product_num:, 3] = sc_vec

    x = torch.tensor(feature, dtype=torch.float)

    prods, custs, edge_weights = [], [], []
    for i in range(product_num):
        for j in range(segment_num):
            prods.append(i)
            custs.append(j + product_num)
            edge_weights.append([float(unit_us[j, i])])

    edge_index = torch.tensor([prods, custs], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    side_ind = torch.tensor([1] * product_num + [0] * segment_num, dtype=torch.long).view(-1, 1)

    prod_labels = np.array(opt_bundles).T
    seg_labels = -np.ones((segment_num, segment_num), dtype=int)
    y = np.append(prod_labels, seg_labels, axis=0)
    y = torch.tensor(y, dtype=torch.long)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        side_ind=side_ind,
        y=y,
        product_num=product_num,
        segment_num=segment_num,
    )
    miscellaneous = (
        product_num, segment_num, unit_cs, ship_cs, unit_us,
        Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, cs, Rs,
    )
    return graph, miscellaneous


def _build_segment_chain_data(selected_products, n):
    """Build progressive chain metadata once for PCP-style feasible bundles."""
    segment_chains = []
    bundle_to_segs = {}
    for seg_idx, selected_list in enumerate(selected_products):
        chain = []
        mask = 0
        for product_idx in selected_list:
            mask |= (1 << (n - 1 - int(product_idx)))
            b = int(mask)
            chain.append(b)
            bundle_to_segs.setdefault(b, set()).add(seg_idx)
        segment_chains.append(chain)

    bundle_primary_seg = {}
    for seg_idx, chain in enumerate(segment_chains):
        for b in chain:
            if b not in bundle_primary_seg:
                bundle_primary_seg[b] = seg_idx

    return segment_chains, bundle_to_segs, bundle_primary_seg


# ---------------------------------------------------------------------------
# Bitmask separation helpers
# ---------------------------------------------------------------------------

def cheapest_cover_dp(k, candidate_bundles, p_val, n):
    """
    Find the minimum-price cover of bundle k from candidate_bundles using
    bitmask DP.

    A cover C satisfies: union of products(b) for b in C  ⊇  products(k).
    We minimise sum_{b in C} p_val[b].

    Complexity: O(|candidates| * 2^t) where t = popcount(k).

    Returns:
        (min_cost, cover_tuple)  or  (INF, None) if no cover exists.
    """
    INF = 1e100
    pk = p_val.get(k, 0.0)

    # Bit positions that are set in k (product indices inside the bundle).
    # Bundle encoding: product i  <->  bit (n-1-i), but we only need the
    # relative bit positions within k for the DP state.
    positions = [pos for pos in range(n) if (k >> pos) & 1]
    t = len(positions)
    if t == 0:
        return 0.0, ()

    full = (1 << t) - 1

    # Map a full-width bitmask to a t-bit compressed mask restricted to k.
    def compress(mask):
        out = 0
        for j, pos in enumerate(positions):
            if (mask >> pos) & 1:
                out |= (1 << j)
        return out

    # Build candidate list with compressed coverage contribution.
    # Prune: if p_val[b] >= pk - eps, it can never be part of a violated cut
    # (all prices are non-negative, so any cover containing b has total cost
    #  >= p_val[b] >= pk - eps, meaning the cut would not be violated).
    cand = []
    for b in candidate_bundles:
        if b == k:
            continue
        pb = p_val.get(b, 0.0)
        if pb >= pk - 1e-8:
            continue
        intersection = b & k
        if intersection == 0:
            continue
        cov = compress(intersection)
        cand.append((b, cov, pb))

    if not cand:
        return INF, None

    # 0-1 knapsack DP: dp[mask] = min cost to cover the products in `mask`
    # (expressed in compressed t-bit coordinates).
    dp = [INF] * (1 << t)
    prev = [None] * (1 << t)
    dp[0] = 0.0

    for b, cov, cost in cand:
        new_dp = dp[:]
        new_prev = prev[:]
        for mask in range(1 << t):
            if dp[mask] >= INF / 2:
                continue
            nm = mask | cov
            val = dp[mask] + cost
            if val < new_dp[nm] - 1e-12:
                new_dp[nm] = val
                new_prev[nm] = (mask, b)
        dp, prev = new_dp, new_prev

    if dp[full] >= INF / 2:
        return INF, None

    # Trace back to recover the cover.
    cover = []
    cur = full
    while cur != 0:
        entry = prev[cur]
        if entry is None:
            break
        pmask, b = entry
        cover.append(b)
        cur = pmask
    cover.reverse()

    return dp[full], tuple(cover)

def cheapest_cover_dp_pruned(k, candidate_bundles, p_val, n, viol_tol=1e-8):
    """
    Exact cheapest-cover DP for bundle k, with safe pruning/compression.

    Finds min sum p[b] over bundles b in candidate_bundles such that
    union(products(b)) covers products(k).

    Returns:
        (best_cost, cover_tuple) or (INF, None).
    """
    INF = 1e100
    pk = p_val.get(k, 0.0)

    # local bit positions of k
    positions = [pos for pos in range(n) if (k >> pos) & 1]
    t = len(positions)
    if t == 0:
        return 0.0, ()

    full = (1 << t) - 1

    def compress(mask):
        out = 0
        for j, pos in enumerate(positions):
            if (mask >> pos) & 1:
                out |= (1 << j)
        return out

    # Step 1: build candidates that could possibly participate in a violated cut
    # If p[b] >= p[k], then any cover containing b cannot violate p[k] <= sum p[b]
    # because prices are nonnegative.
    best_by_cov = {}  # cov_mask -> (bundle_id, cost)
    for b in candidate_bundles:
        if b == k:
            continue

        pb = p_val.get(b, 0.0)
        if pb >= pk - viol_tol:
            continue

        inter = b & k
        if inter == 0:
            continue

        cov = compress(inter)
        prev = best_by_cov.get(cov)
        if prev is None or pb < prev[1] - 1e-12:
            best_by_cov[cov] = (b, pb)

    if not best_by_cov:
        return INF, None

    cand = [(b, cov, cost) for cov, (b, cost) in best_by_cov.items()]

    # Step 2: safe dominance pruning
    # If candidate 2 covers a superset of candidate 1's local mask and is no more
    # expensive, candidate 1 is never needed in an optimal cover.
    keep = [True] * len(cand)
    for i in range(len(cand)):
        bi, cov_i, cost_i = cand[i]
        for j in range(len(cand)):
            if i == j:
                continue
            bj, cov_j, cost_j = cand[j]
            if ((cov_i | cov_j) == cov_j) and (cost_j <= cost_i + 1e-12):
                # j dominates i
                keep[i] = False
                break

    cand = [cand[i] for i in range(len(cand)) if keep[i]]
    if not cand:
        return INF, None

    # Step 3: exact 0-1 DP
    dp = [INF] * (1 << t)
    prev = [None] * (1 << t)
    dp[0] = 0.0

    for b, cov, cost in cand:
        new_dp = dp[:]
        new_prev = prev[:]
        for mask in range(1 << t):
            if dp[mask] >= INF / 2:
                continue
            nm = mask | cov
            val = dp[mask] + cost
            if val < new_dp[nm] - 1e-12:
                new_dp[nm] = val
                new_prev[nm] = (mask, b)
        dp, prev = new_dp, new_prev

    if dp[full] >= INF / 2:
        return INF, None

    cover = []
    cur = full
    while cur != 0:
        entry = prev[cur]
        if entry is None:
            return INF, None
        pmask, b = entry
        cover.append(int(b))
        cur = pmask

    cover.sort()  # canonicalize for duplicate detection
    return dp[full], tuple(cover)


def _pareto_prune_states(state_map, cutoff):
    """
    Remove states that are clearly useless:
      - cost already too large to yield a violated cut
      - dominated by a cheaper state with a superset mask
    """
    items = [
        (mask, data)
        for mask, data in state_map.items()
        if data[0] < cutoff - 1e-12
    ]
    if not items:
        return {}

    keep = [True] * len(items)
    for i in range(len(items)):
        if not keep[i]:
            continue
        mask_i, data_i = items[i]
        cost_i = data_i[0]
        for j in range(len(items)):
            if i == j or not keep[j]:
                continue
            mask_j, data_j = items[j]
            cost_j = data_j[0]
            if ((mask_i | mask_j) == mask_j) and (cost_j <= cost_i + 1e-12):
                keep[i] = False
                break

    return {items[i][0]: items[i][1] for i in range(len(items)) if keep[i]}


def cheapest_cover_segments_dp(k, per_k_segment_options, p_val, viol_tol=1e-8):
    """
    Exact cheapest-cover separation specialized to PCP progressive chains.

    Because bundles come from progressive chains and chain monotonicity is
    enforced, an optimal cover uses at most one bundle per segment.  We exploit
    that structure with a sparse DP over segment layers instead of a 2^|k| DP.
    """
    INF = 1e100
    pk = p_val.get(k, 0.0)
    cutoff = pk - viol_tol
    if cutoff <= 0.0:
        return INF, None

    segment_options, suffix_union = per_k_segment_options
    full_mask = k

    states = {0: (0.0, None, None)}
    layers = [states]
    for seg_idx, options in enumerate(segment_options):
        next_states = {
            covered_mask: (cost, covered_mask, None)
            for covered_mask, (cost, _, _) in states.items()
        }

        for covered_mask, (cost, _, _) in states.items():
            if cost >= cutoff - 1e-12:
                continue
            if (covered_mask | suffix_union[seg_idx]) != full_mask:
                continue

            for overlap_mask, bundle_id in options:
                bundle_cost = p_val.get(bundle_id, 0.0)
                new_cost = cost + bundle_cost
                if new_cost >= cutoff - 1e-12:
                    continue
                new_mask = covered_mask | overlap_mask
                prev = next_states.get(new_mask)
                if prev is None or new_cost < prev[0] - 1e-12:
                    next_states[new_mask] = (new_cost, covered_mask, int(bundle_id))

        states = _pareto_prune_states(next_states, cutoff)
        if not states:
            return INF, None
        layers.append(states)

    best = states.get(full_mask)
    if best is None:
        return INF, None

    best_cost = best[0]
    cover = []
    cur_mask = full_mask
    for layer_idx in range(len(segment_options), 0, -1):
        entry = layers[layer_idx].get(cur_mask)
        if entry is None:
            return INF, None
        _, prev_mask, chosen_bundle = entry
        if chosen_bundle is not None:
            cover.append(chosen_bundle)
        cur_mask = prev_mask if prev_mask is not None else cur_mask

    if cur_mask != 0:
        return INF, None

    cover.sort()
    return best_cost, tuple(cover)


def _two_segment_lb_exceeds_target(segment_options, p_sol, target_price, viol_tol):
    """
    Cheap exact screen before running the segment DP.

    For PCP chain options, each segment's cheapest usable bundle for target k is
    the first option in that segment list because chain monotonicity enforces
    nondecreasing prices along the progressive chain. Any multi-segment cover
    needs bundles from at least two segments, so the sum of the two cheapest
    segment-entry prices is a valid lower bound on every admissible cover cost.
    """
    if len(segment_options) < 2:
        return True

    best_1 = math.inf
    best_2 = math.inf
    for opts in segment_options:
        entry_price = p_sol[opts[0][1]]
        if entry_price < best_1:
            best_2 = best_1
            best_1 = entry_price
        elif entry_price < best_2:
            best_2 = entry_price

    return (best_1 + best_2) >= target_price - viol_tol

# ---------------------------------------------------------------------------
# MILP with cutting-plane subadditivity
# ---------------------------------------------------------------------------

def _add_progressive_chain_constraints(model, p, selected_products, n, feasible_bundles_list):
    """
    Within-segment monotonicity: p[b_1] <= p[b_2] <= ... for the progressive
    chain of each segment.
    """
    feasible_bundle_set = set(feasible_bundles_list)

    for seg_idx, selected_list in enumerate(selected_products):
        if len(selected_list) <= 1:
            continue

        progressive = []
        bid = 0
        for prod_idx in selected_list:
            bid |= (1 << (n - 1 - int(prod_idx)))
            if bid in feasible_bundle_set:
                progressive.append(int(bid))

        for i in range(len(progressive) - 1):
            model.addConstr(
                p[progressive[i]] <= p[progressive[i + 1]],
                name=f"chain_s{seg_idx}_b{i}",
            )


def _add_all_singleton_monotonicity_constraints(
    model, p, feasible_bundles_list, bundle_to_segs=None
):
    """
    Add all singleton cover cuts upfront:
        p[k] <= p[b] whenever products(k) ⊆ products(b), b != k.

    This is exact and usually cheap enough (O(|B|^2) bit checks). It tightens
    the master model a lot and leaves only multi-bundle covers to the callback.
    """
    nonempty = [int(b) for b in feasible_bundles_list if b != 0]

    for k in nonempty:
        k_segs = bundle_to_segs.get(k, set()) if bundle_to_segs is not None else None
        for b in nonempty:
            if b == k:
                continue
            # bitwise inclusion: k ⊆ b
            if (b & k) == k:
                if bundle_to_segs is not None:
                    b_segs = bundle_to_segs.get(b, set())
                    if k_segs & b_segs:
                        continue
                model.addConstr(p[k] <= p[b], name=f"mono_{k}_{b}")


def revenue_ratio(
    n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev,
    feasible_bundles, selected_products=None,
    stored_cs=None, stored_Rs=None,
    fcp_fallback_time_limit=60.0,
):
    """
    Solve the PCP MILP using lazy-constraint callback separation.

    Exact model:
      - within-segment chain monotonicity added upfront
      - all singleton cover cuts p[k] <= p[b] (k ⊆ b) added upfront
      - multi-bundle cover cuts added lazily at incumbent solutions

    This is mathematically equivalent to the full pre-enumerated PCP model,
    up to Gurobi's MIP tolerances / time limit behavior.
    """
    TOTAL_TIME_LIMIT = 600.0
    VIOL_TOL = 1e-6
    # One violated lazy cut is enough to reject the incumbent, so stopping after
    # the first cut often reduces callback time substantially while preserving
    # exactness.
    MAX_LAZY_PER_CALLBACK = 1

    segment_ind = np.arange(m)
    feasible_bundles_list = [int(b) for b in feasible_bundles]

    # --- Rs and cost matrices ---
    t_data_start = time.time()
    if stored_cs is not None and stored_Rs is not None:
        costs = stored_cs
        Rs = stored_Rs
        Rbar = np.max(Rs)
        bundle_to_idx = {b: b for b in feasible_bundles_list}
    else:
        assortments = np.zeros((len(feasible_bundles_list), n), dtype=int)
        for row_idx, bundle_id in enumerate(feasible_bundles_list):
            remaining = int(bundle_id)
            while remaining:
                lowbit = remaining & -remaining
                bit_pos = lowbit.bit_length() - 1
                assortments[row_idx, n - 1 - bit_pos] = 1
                remaining ^= lowbit
        Rs_f = np.sqrt(unit_us.dot(assortments.T))
        cs_base = np.sum(assortments * unit_cs, axis=1)
        cs_f = (cs_base + ship_cs) * 0.2
        bundle_to_idx = {b: idx for idx, b in enumerate(feasible_bundles_list)}
        costs = cs_f
        Rs = Rs_f
        Rbar = np.max(Rs_f)
    t_data_done = time.time()

    # --- Build model ---
    t_model_start = time.time()
    model = gp.Model('PCP_lazy')
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-3
    model.Params.TimeLimit = TOTAL_TIME_LIMIT
    model.Params.LazyConstraints = 1
    model.Params.MIPFocus = 1
    model.Params.Cuts = 2
    p = model.addVars(
        feasible_bundles_list,
        lb=0.0,
        ub=float(Rbar),
        vtype=GRB.CONTINUOUS,
        name='p',
    )
    theta = model.addVars(m, feasible_bundles_list, vtype=GRB.BINARY, name='theta')
    s = model.addVars(m, vtype=GRB.CONTINUOUS, name='s')
    S = model.addVars(m, feasible_bundles_list, lb=0.0, vtype=GRB.CONTINUOUS, name='S')
    Z = model.addVars(m, feasible_bundles_list, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Z')
    P = model.addVars(m, feasible_bundles_list, lb=0.0, vtype=GRB.CONTINUOUS, name='P')

    model.setObjective(
        gp.quicksum(Ns[k, 0] * Z[k, i] for k in segment_ind for i in feasible_bundles_list),
        GRB.MAXIMIZE,
    )

    model.addConstrs((s[k] >= Rs[k, bundle_to_idx[i]] - p[i]
                      for i in feasible_bundles_list for k in segment_ind))
    model.addConstrs((P[k, i] >= p[i] - Rbar * (1 - theta[k, i])
                      for i in feasible_bundles_list for k in segment_ind))
    model.addConstrs((P[k, i] <= p[i]
                      for i in feasible_bundles_list for k in segment_ind))
    model.addConstrs((s[k] >= gp.quicksum(
                          Rs[k, bundle_to_idx[i]] * theta[j, i] - P[j, i]
                          for i in feasible_bundles_list)
                      for k in segment_ind for j in segment_ind))
    model.addConstrs((Z[k, i] == P[k, i] - costs[k, bundle_to_idx[i]] * theta[k, i]
                      for i in feasible_bundles_list for k in segment_ind))
    model.addConstrs((S[k, i] == Rs[k, bundle_to_idx[i]] * theta[k, i] - P[k, i]
                      for i in feasible_bundles_list for k in segment_ind))
    model.addConstrs((s[k] == gp.quicksum(S[k, i] for i in feasible_bundles_list)
                      for k in segment_ind))
    model.addConstrs((gp.quicksum(theta[k, i] for i in feasible_bundles_list) == 1
                      for k in segment_ind))

    segment_chain_data = None
    per_k_segment_options = None

    # within-segment chain monotonicity
    if selected_products is not None:
        _add_progressive_chain_constraints(model, p, selected_products, n, feasible_bundles_list)
        segment_chains, bundle_to_segs, bundle_primary_seg = _build_segment_chain_data(selected_products, n)
        segment_chain_data = (segment_chains, bundle_to_segs, bundle_primary_seg)

    # empty bundle
    if 0 in feasible_bundles_list:
        model.addConstr(p[0] == 0, name='p0_zero')
        model.addConstrs((S[k, 0] == 0 for k in segment_ind))

    nonempty_bundles = [b for b in feasible_bundles_list if b != 0]
    lazy_check_bundles = nonempty_bundles

    if segment_chain_data is not None:
        _, bundle_to_segs, bundle_primary_seg = segment_chain_data
        # Precompute exact per-target segment options for fast callback separation.
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

        # Skip singleton cuts already implied by within-segment chain constraints.
        _add_all_singleton_monotonicity_constraints(
            model, p, feasible_bundles_list, bundle_to_segs=bundle_to_segs
        )
    else:
        _add_all_singleton_monotonicity_constraints(model, p, feasible_bundles_list)

    t_model_done = time.time()

    # print(
    #     f"  [diag] n={n}, m={m}, #feasible_bundles={len(feasible_bundles_list)}, "
    #     f"#nonempty={len(nonempty_bundles)}"
    # )
    # print(
    #     f"  [diag] prep: data={t_data_done - t_data_start:.3f}s, "
    #     f"model_build={t_model_done - t_model_start:.3f}s"
    # )

    # dedup cache for lazy cuts already added
    added_lazy = set()
    cb_stats = {
        'calls': 0,
        'bundle_checks': 0,
        'cuts_added': 0,
        'sep_time': 0.0,
        'max_cuts_in_callback': 0,
        'callbacks_with_cuts': 0,
    }

    # Store references on model for callback access if you want, but closure is enough.
    def _lazy_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        cb_stats['calls'] += 1
        cb_start = time.time()

        # incumbent prices
        p_values = cb_model.cbGetSolution([p[b] for b in feasible_bundles_list])
        p_sol = dict(zip(feasible_bundles_list, p_values))

        # Check bundles in descending current price; once we cut this incumbent,
        # there is no need to find every violated cut in this callback.
        bundles_by_price = sorted(
            (b for b in lazy_check_bundles if p_sol[b] > 1e-8),
            key=lambda b: p_sol[b],
            reverse=True
        )

        cuts_added = 0
        for k in bundles_by_price:
            cb_stats['bundle_checks'] += 1
            if per_k_segment_options is not None:
                segment_options, _ = per_k_segment_options[k]
                if _two_segment_lb_exceeds_target(
                    segment_options, p_sol, p_sol[k], VIOL_TOL
                ):
                    continue
                best_cost, best_cover = cheapest_cover_segments_dp(
                    k, per_k_segment_options[k], p_sol, viol_tol=VIOL_TOL
                )
            else:
                best_cost, best_cover = cheapest_cover_dp_pruned(
                    k, nonempty_bundles, p_sol, n, viol_tol=VIOL_TOL
                )

            if best_cover is None:
                continue

            # singleton covers are already enforced upfront
            if len(best_cover) <= 1:
                continue

            if best_cost < p_sol[k] - VIOL_TOL:
                key = (int(k), tuple(best_cover))

                if key in added_lazy:
                    continue

                cb_model.cbLazy(
                    p[k] <= gp.quicksum(p[b] for b in best_cover)
                )
                added_lazy.add(key)
                cb_stats['cuts_added'] += 1

  
                cuts_added += 1

                if cuts_added >= MAX_LAZY_PER_CALLBACK:
                    break

        if cuts_added > 0:
            cb_stats['callbacks_with_cuts'] += 1
        cb_stats['max_cuts_in_callback'] = max(cb_stats['max_cuts_in_callback'], cuts_added)
        cb_stats['sep_time'] += time.time() - cb_start

    t_opt_start = time.time()
    model.optimize(_lazy_callback)
    t_opt_done = time.time()

    gurobi_runtime = float(model.Runtime)
    solve_time = gurobi_runtime
    solve_wall = t_opt_done - t_model_done

    # print(
    #     f"  [diag] optimize: wall={solve_wall:.3f}s, gurobi_runtime={model.Runtime:.3f}s, "
    #     f"status={model.status}"
    # )
    # print(
    #     f"  [diag] callback: calls={cb_stats['calls']}, "
    #     f"bundle_checks={cb_stats['bundle_checks']}, cuts_added={cb_stats['cuts_added']}, "
    #     f"callbacks_with_cuts={cb_stats['callbacks_with_cuts']}, "
    #     f"max_cuts_per_callback={cb_stats['max_cuts_in_callback']}, "
    #     f"sep_time={cb_stats['sep_time']:.3f}s"
    # )
    # if cb_stats['calls'] > 0:
    #     print(
    #         f"  [diag] callback avg: "
    #         f"checks/call={cb_stats['bundle_checks'] / cb_stats['calls']:.1f}, "
    #         f"sep_time/call={cb_stats['sep_time'] / cb_stats['calls']:.4f}s"
    #     )

    if model.status == GRB.OPTIMAL:
        return model.ObjVal / opt_rev, solve_time
    elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        return model.ObjVal / opt_rev, solve_time
    elif model.status == GRB.TIME_LIMIT and model.SolCount == 0:
        print("Time limit reached with no feasible lazy-PCP solution. Falling back to FCP.")
        if selected_products is None:
            print("FCP fallback skipped: selected_products is unavailable.")
            return 0.0, solve_time

        pred_assort = np.zeros((m, n), dtype=int)
        for seg_idx, selected_list in enumerate(selected_products):
            for product_idx in selected_list:
                pred_assort[seg_idx, int(product_idx)] = 1

        try:
            fcp_ratio, fallback_solve_time = fcp_revenue_ratio(
                n,
                m,
                unit_cs,
                ship_cs,
                unit_us,
                Ns,
                opt_rev,
                pred_assort,
                time_limit=fcp_fallback_time_limit,
            )
            fallback_total_time = solve_time + fallback_solve_time
            print(
                f"  [diag] FCP fallback succeeded with time_limit={fcp_fallback_time_limit}s, "
                f"ratio={fcp_ratio:.4f}"
            )
            return fcp_ratio, fallback_total_time
        except Exception as fallback_error:
            print(f"FCP fallback failed: {fallback_error}")
            return 0.0, solve_time
    else:
        print(f"Optimization failed with final status: {model.status}")
        return 0.0, solve_time
    
# ---------------------------------------------------------------------------
# Bundle generation helpers
# ---------------------------------------------------------------------------

def top_m_selection(output, m, threshold=0.5):
    """
    For each segment (column of output), select products with probability >= threshold.
    output shape: (n_products, n_segments).
    Returns list of lists of selected product indices (desc probability order).
    """
    n, m_segments = output.shape
    selected_products = []
    for j in range(m_segments):
        sorted_indices = np.argsort(output[:, j])[::-1]
        top_m_indices = sorted_indices[:n]
        filtered = [idx for idx in top_m_indices if output[idx, j] >= threshold]
        selected_products.append(filtered)
    return selected_products


def generate_progressive_bundles(selected_products, n):
    """
    For each segment, build progressive bundles {p1}, {p1,p2}, ..., {p1,...,pM}
    where products are ordered by descending probability.
    Returns set of feasible bundle IDs (Python int).
    """
    feasible_bundles = {0}
    for selected_list in selected_products:
        bundle_idx = 0
        for product_idx in selected_list:
            bundle_idx |= (1 << (n - 1 - int(product_idx)))
            feasible_bundles.add(int(bundle_idx))
    return feasible_bundles


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_path = os.path.join(dir_path, "models", "main_base_4layer_correct_lr_3", "best_model_edge_4layer_seed1.pt")
    test_data_path = os.path.join(dir_path, "data", "deterministic", "test_BSP_m20n40_correct_1e_3") + os.sep
    result_path = 'test_result_PCP_cp_m10n20.csv'
    fcp_fallback_time_limit = 60.0

    print(f"Directory path: {dir_path}")
    print(f"Model path: {model_path}")
    print(f"Test data path: {test_data_path}")
    print(f"FCP fallback time limit: {fcp_fallback_time_limit}s")

    print('Loading trained model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setattr(sys.modules["__main__"], "EdgeScoringGCN", EdgeScoringGCN)
    gcn_model = torch.load(model_path, map_location=device, weights_only=False)
    gcn_model.to(device)
    gcn_model.eval()

    print('\nBegin model evaluation...')
    dir_list = os.listdir(test_data_path)
    test_dataset = []
    miscellaneous_dataset = []
    print('Start reading the dataset...')

    for fname in dir_list:
        if fname == '.DS_Store':
            continue
        try:
            dat, misc = process_data(test_data_path + fname)
            test_dataset.append(dat)
            miscellaneous_dataset.append(misc)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    sample_num = len(test_dataset)
    print(f'Successfully loaded {sample_num} test samples.')

    results = []

    with torch.no_grad():
        for i in tqdm(range(sample_num), desc="Evaluating"):
            try:
                dat = test_dataset[i].to(device)
                misc = miscellaneous_dataset[i]
                n, segment_num, unit_cs, ship_cs, unit_us, Ns, \
                    opt_bundles, opt_prices, opt_rev, running_time, gap, \
                    stored_cs, stored_Rs = misc

                gcn_start_time = time.time()
                raw_out = gcn_model(dat)
                gcn_inference_time = time.time() - gcn_start_time

                if isinstance(raw_out, dict):
                    if 'logit_matrix' in raw_out:
                        logits_nm = raw_out['logit_matrix'].detach().cpu().numpy()
                    elif 'edge_logits' in raw_out:
                        logits_nm = raw_out['edge_logits'].detach().cpu().numpy().reshape(n, segment_num)
                    else:
                        raise ValueError('Unexpected model output keys: ' + ','.join(raw_out.keys()))
                    sigmoid_output = 1.0 / (1.0 + np.exp(-logits_nm))
                else:
                    output = raw_out[:n, :].detach().cpu().numpy()
                    sigmoid_output = np.exp(output) / (np.exp(output) + np.exp(1))

                selected_products = top_m_selection(sigmoid_output, m=n, threshold=0.5)
                feasible_bundles = generate_progressive_bundles(selected_products, n)

                ratio, milp_time = revenue_ratio(
                    n, segment_num, unit_cs, ship_cs, unit_us, Ns, opt_rev,
                    feasible_bundles, selected_products, stored_cs, stored_Rs,
                    fcp_fallback_time_limit=fcp_fallback_time_limit,
                )

                total_time = gcn_inference_time + milp_time
                runtime_ratio = total_time / running_time if running_time > 0 else float('inf')

                if ratio < 0 or np.isinf(opt_rev) or opt_rev == 0:
                    print(f"\nProblematic sample {i}: ratio={ratio}, opt_rev={opt_rev}")
                    continue

                results.append([n, ratio, runtime_ratio, total_time, milp_time])

            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                continue

    results = np.array(results)
    header = 'n_products,revenue_ratio,runtime_ratio,total_time,milp_time'
    # np.savetxt(result_path, results, delimiter=',', header=header, comments='')

    print(f'\nEvaluation completed!')
    print(f'Method: PCP with cutting-plane subadditivity (bitmask DP separation)')
    print(f'Results saved to: {result_path}')
    print(f'Number of samples evaluated: {len(results)}')

    if len(results) > 0:
        print(f'\n  Revenue Ratio Statistics:')
        print(f'  Mean:  {np.mean(results[:, 1]):.4f}')
        print(f'  Std:   {np.std(results[:, 1]):.4f}')
        print(f'  Min:   {np.min(results[:, 1]):.4f}')
        print(f'  Max:   {np.max(results[:, 1]):.4f}')
        print(f'\n  Runtime Statistics:')
        print(f'  Mean total time:   {np.mean(results[:, 3]):.4f}s')
        print(f'  Mean MILP time:    {np.mean(results[:, 4]):.4f}s')
        print(f'  Mean GCN time:     {np.mean(results[:, 3] - results[:, 4]):.4f}s')
        print(f'  Mean runtime ratio (vs default): {np.mean(results[:, 2]):.4f}')


if __name__ == "__main__":
    main()
