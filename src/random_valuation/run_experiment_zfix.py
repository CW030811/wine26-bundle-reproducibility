"""Random-valuation main experiment, z-fixed, with a clean before/after on the
solver bug.

Uses the Table-5 solver (hanson_empty_kway subadditivity, scalable to N=30).
For each cell (scale N x cost x seed) it generates the Table-5 instance
deterministically, then runs FCP-pruned-MB and BSP TWICE on the same retrained
Hanson model and same instance:
  - "fixed": Z profit var lb=-inf (correct)
  - "buggy": Z profit var lb=0   (the bug)
isolating the experiment-side solver fix. CPBSD-A is unaffected by the fix and
is reused from the published Table 5 in the aggregation step.
"""
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from table5_solvers_zfix import solve_mb_restricted, solve_bsp, eval_mb_policy, eval_bsp_policy
# AUDIT FIX: generate instances with the Table-5/04 generator (random_corr ratio
# ~ Uniform(0,1), matching the paper appendix), NOT generate_data_CPBSD.py whose
# canonical copy uses Uniform(0.3,0.9) and would not match Table 5 for random_corr.
from table5_solvers_zfix import CPBSDSetup, generate_cpbsd_instance, valuation_means, sample_valuations
from cpbsd_hanson_gcn_graph import build_hanson_cpbsd_graph
from Training_multi_layer_cpbsd_mb_x import EdgeScoringGCN

VARIANTS = {"fixed": {}, "buggy": {"profit_lb": 0.0}}  # fixed uses the solver default lb=-inf


def load_model(path, device):
    setattr(sys.modules["__main__"], "EdgeScoringGCN", EdgeScoringGCN)
    m = torch.load(path, map_location=device, weights_only=False)
    m.to(device); m.eval(); return m


def infer_prob(model, v_kn, c_n, device):
    data, _ = build_hanson_cpbsd_graph(v_kn=v_kn, c_n=c_n)
    data = data.to(device)
    with torch.no_grad():
        out = model(data)
    n, m = int(data.product_num), int(data.segment_num)
    logits = out["logit_matrix"].cpu().numpy() if "logit_matrix" in out else out["edge_logits"].cpu().numpy().reshape(n, m)
    return torch.sigmoid(torch.tensor(logits)).numpy().T  # [m, n]


def fcp_candidates(prob, threshold=0.5):
    bundles = [tuple((prob[k] >= threshold).astype(int).tolist()) for k in range(prob.shape[0])]
    uniq = np.array(sorted(set(bundles)), dtype=int)
    return uniq if uniq.size else np.zeros((0, prob.shape[1]), dtype=int)


def oos_draw(N, seed, k=5000):
    rng = np.random.default_rng(int(seed) + 99991)
    return sample_valuations(k=k, means=valuation_means(N, "full"), family="normal", rho=0.0, rng=rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--scales", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260413, 20260414, 20260415, 20260416, 20260417])
    ap.add_argument("--costs", nargs="+", default=["zero", "random_ind", "random_corr"])
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-samples", type=int, default=5000)
    ap.add_argument("--mip-gap", type=float, default=1e-3)
    ap.add_argument("--time-limit", type=float, default=300.0)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(args.model_path, device)
    rows = []
    for N in args.scales:
        for cost in args.costs:
            for seed in args.seeds:
                setup = CPBSDSetup(N, args.K, "normal", 0.0, "full", cost, seed)
                inst = generate_cpbsd_instance(setup)
                v_kn = np.asarray(inst["valuation_samples_V"], float)
                c_n = np.asarray(inst["production_cost_c"], float)
                prob = infer_prob(model, v_kn, c_n, device)
                cands = fcp_candidates(prob, args.threshold)
                v_out = oos_draw(N, seed, args.out_samples)
                for variant, kw in VARIANTS.items():
                    t0 = time.time()
                    f = solve_mb_restricted(v_kn, c_n, cands, time_limit=args.time_limit, mip_gap=args.mip_gap,
                                            output_flag=0, subadditivity_mode="hanson_empty_kway", **kw)
                    f_t = time.time() - t0
                    f_pol = f.get("bundle_prices_full") or {}
                    f_oos = eval_mb_policy(v_out, c_n, f_pol, np.asarray(f["assortments"], int)) if f.get("feasible") and f_pol else None
                    rows.append(dict(scale=f"N{N}_K{args.K}", cost=cost, seed=seed, method="FCP", variant=variant,
                                     ins=f.get("objective"), oos=f_oos, runtime=f_t,
                                     n_candidates=int(cands.shape[0]), status=int(f.get("solver_status", -1))))
                    t0 = time.time()
                    b = solve_bsp(v_kn, c_n, time_limit=args.time_limit, mip_gap=args.mip_gap, output_flag=0, **kw)
                    b_t = time.time() - t0
                    b_pol = b.get("size_prices") or {}
                    b_oos = eval_bsp_policy(v_out, c_n, b_pol) if b.get("feasible") and b_pol else None
                    rows.append(dict(scale=f"N{N}_K{args.K}", cost=cost, seed=seed, method="BSP", variant=variant,
                                     ins=b.get("objective"), oos=b_oos, runtime=b_t,
                                     n_candidates=int(cands.shape[0]), status=int(b.get("solver_status", -1))))
                print(f"done N={N} cost={cost} seed={seed} cands={cands.shape[0]}", flush=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"WROTE {args.out_csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
