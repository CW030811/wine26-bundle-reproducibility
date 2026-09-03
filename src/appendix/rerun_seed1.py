from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import types
import __main__
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = PACKAGE_ROOT / "src" / "appendix" / "legacy_runtime"
OUT_ROOT = PACKAGE_ROOT / "results" / "appendix_cd_seed1"
DATASET_DIR = PACKAGE_ROOT / "data" / "deterministic"
MODEL_PATH = PACKAGE_ROOT / "models" / "appendix_seed1" / "model_edge_4layer_seed1.pt"
MB_DATASETS = ["test_m10n10_1e_3", "test_m20n10_1e_3", "test_m30n10_1e_3"]
BSP_DATASETS = ["test_BSP_m10n20_1e_3", "test_BSP_m20n20_1e_3"]
CUTOFFS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K_STRATEGIES = ["original", "K_m", "K_const_5", "K_const_10", "K_sqrt_mn", "K_sqrt_m", "K_2sqrt_m"]
K_LABELS = {
    "original": "2m",
    "K_m": "K=m",
    "K_const_5": "K=5",
    "K_const_10": "K=10",
    "K_sqrt_mn": "K=sqrt(mn)",
    "K_sqrt_m": "K=sqrt(m)",
    "K_2sqrt_m": "K=2*sqrt(m)",
}
K_COLORS = {
    "original": "#4C9BD6",
    "K_m": "#4C9BD6",
    "K_const_5": "#4DAF4A",
    "K_const_10": "#E15759",
    "K_sqrt_mn": "#9C755F",
    "K_sqrt_m": "#8E6BBE",
    "K_2sqrt_m": "#C9BE2D",
}
K_MARKERS = {
    "original": "o",
    "K_m": "P",
    "K_const_5": "s",
    "K_const_10": "^",
    "K_sqrt_mn": "v",
    "K_sqrt_m": "D",
    "K_2sqrt_m": "X",
}
DATASET_COMPACT_LABELS = {
    "test_m10n10_1e_3": "m10n10",
    "test_m20n10_1e_3": "m20n10",
    "test_m30n10_1e_3": "m30n10",
}


def load_module_with_future(name: str, path: Path, time_limit: int | None = None):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    src = path.read_text(encoding="utf-8")
    if time_limit is not None:
        src = src.replace("model.Params.TimeLimit = 600", f"model.Params.TimeLimit = {float(time_limit)!r}")
    code = compile("from __future__ import annotations\n" + src, str(path), "exec")
    exec(code, mod.__dict__)
    return mod


def load_modules(time_limit: int | None = None):
    test_fcp = load_module_with_future("test_FCP", TEST_DIR / "test_FCP.py", time_limit=time_limit)
    test_pcp = load_module_with_future("test_PCP", TEST_DIR / "test_PCP.py", time_limit=time_limit)
    score = load_module_with_future("test_FCPLS_score", TEST_DIR / "test_FCPLS_score.py", time_limit=time_limit)
    return test_fcp, test_pcp, score


def load_seed1_model(score_mod, device):
    __main__.EdgeScoringGCN = score_mod.EdgeScoringGCN
    loaded = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if getattr(loaded, "convs", None) is not None:
        state = loaded.state_dict()
        model = score_mod.EdgeScoringGCN(
            in_channels=4,
            hidden_channels=128,
            num_layers=4,
            edge_dim=1,
            dropout=0.5,
        )
        model.load_state_dict(state, strict=True)
    else:
        model = loaded
    model.to(device)
    model.eval()
    return model


def selected_files(subdir: str, sample_count: int):
    files = sorted(
        f for f in os.listdir(DATASET_DIR / subdir)
        if f.endswith(".msgpack") and f != ".DS_Store"
    )
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(files), size=min(sample_count, len(files)), replace=False)
    return [files[i] for i in idxs]


def logit_threshold(cutoff: float) -> float:
    return float(np.log(cutoff / (1.0 - cutoff)))


def bundle_metrics(pred_assort, opt_bundles, n: int, m: int):
    opt_raw = np.array(opt_bundles)
    if opt_raw.shape == (m, n):
        opt = opt_raw
    elif opt_raw.shape == (n, m):
        opt = opt_raw.T
    else:
        opt = opt_raw.T if opt_raw.shape[0] == n else opt_raw
    pred = np.asarray(pred_assort).astype(int)
    opt = np.asarray(opt).astype(int)
    tp = np.sum((pred == 1) & (opt == 1))
    fn = np.sum((pred == 0) & (opt == 1))
    fp = np.sum((pred == 1) & (opt == 0))
    tn = np.sum((pred == 0) & (opt == 0))
    total = tp + fn + fp + tn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "tpr": tp / (tp + fn) if (tp + fn) else 0.0,
        "tnr": tn / (tn + fp) if (tn + fp) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "error_rate": (fp + fn) / total if total else 0.0,
    }


def infer_logits(model, dat, n: int, m: int, device):
    with torch.no_grad():
        raw = model(dat.to(device))
    if "logit_matrix" in raw:
        logits_nm = raw["logit_matrix"].detach().cpu().numpy()
    else:
        logits_nm = raw["edge_logits"].detach().cpu().numpy().reshape(n, m)
    return logits_nm


def _load_cutoff_samples(test_fcp, model, device, sample_count: int):
    samples = []
    for ds in MB_DATASETS:
        for fname in selected_files(ds, sample_count):
            fpath = DATASET_DIR / ds / fname
            dat, misc = test_fcp.process_data(str(fpath))
            n, m = misc[0], misc[1]
            t0 = time.perf_counter()
            logits_nm = infer_logits(model, dat, n, m, device)
            gcn_time = time.perf_counter() - t0
            samples.append((ds, fname, dat, misc, logits_nm, gcn_time))
    return samples


def run_cutoff(fcp_sample_count: int, pcp_sample_count: int, time_limit: int | None, resume: bool):
    test_fcp, test_pcp, score = load_modules(time_limit=time_limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_seed1_model(score, device)
    outdir = OUT_ROOT / "cutoff"
    outdir.mkdir(parents=True, exist_ok=True)

    # The final paper artifact intentionally used the legacy sampling protocol:
    # 30 samples per dataset for FCP and a separately sampled set of 10 for PCP.
    fcp_samples = _load_cutoff_samples(test_fcp, model, device, fcp_sample_count)
    pcp_samples = _load_cutoff_samples(test_fcp, model, device, pcp_sample_count)

    csv_path = outdir / "cutoff_sensitivity_FCP_PCP_combined.csv"
    fieldnames = [
        "dataset", "sample_id", "cutoff", "strategy", "revenue_ratio", "time_ratio",
        "accuracy", "precision", "recall", "tpr", "tnr", "fpr", "fnr", "error_rate",
    ]
    done = set()
    if resume and csv_path.exists():
        old = pd.read_csv(csv_path)
        for r in old[["dataset", "sample_id", "cutoff", "strategy"]].itertuples(index=False):
            done.add((r.dataset, r.sample_id, float(r.cutoff), r.strategy))
    write_header = not csv_path.exists() or not resume
    fh = csv_path.open("a" if resume else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    for cutoff in tqdm(CUTOFFS, desc="cutoff sweep"):
        thresh = logit_threshold(cutoff)
        for ds, fname, _dat, misc, logits_nm, gcn_time in fcp_samples:
            n, m, unit_cs, ship_cs, unit_us, Ns, opt_bundles, _opt_prices, opt_rev, running_time, _gap = misc
            pred_fcp = (logits_nm.T >= thresh).astype(int)
            if (ds, fname, float(cutoff), "FCP") not in done:
                try:
                    rev, milp_time = test_fcp.revenue_ratio(n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, pred_fcp)
                except Exception:
                    rev, milp_time = np.nan, np.nan
                metrics = bundle_metrics(pred_fcp, opt_bundles, n, m)
                writer.writerow({
                    "dataset": ds, "sample_id": fname, "cutoff": cutoff, "strategy": "FCP",
                    "revenue_ratio": rev,
                    "time_ratio": (gcn_time + (milp_time if np.isfinite(milp_time) else 0.0)) / running_time if running_time > 0 else np.nan,
                    **metrics,
                })
                fh.flush()

        for ds, fname, _dat, misc, logits_nm, gcn_time in pcp_samples:
            n, m, unit_cs, ship_cs, unit_us, Ns, opt_bundles, _opt_prices, opt_rev, running_time, _gap = misc
            prob_nm = 1.0 / (1.0 + np.exp(-logits_nm))
            selected_products = test_pcp.top_m_selection(prob_nm, m=n, threshold=cutoff)
            feasible_bundles = test_pcp.generate_progressive_bundles(selected_products, n)
            if (ds, fname, float(cutoff), "PCP") not in done:
                try:
                    rev, milp_time = test_pcp.revenue_ratio(
                        n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev,
                        feasible_bundles, selected_products=selected_products,
                    )
                except Exception:
                    rev, milp_time = np.nan, np.nan
                # For PCP classification metrics, compare the union of progressive selected products per segment.
                pred_pcp = np.zeros((m, n), dtype=int)
                for seg, plist in enumerate(selected_products):
                    pred_pcp[seg, plist] = 1
                metrics = bundle_metrics(pred_pcp, opt_bundles, n, m)
                writer.writerow({
                    "dataset": ds, "sample_id": fname, "cutoff": cutoff, "strategy": "PCP",
                    "revenue_ratio": rev,
                    "time_ratio": (gcn_time + (milp_time if np.isfinite(milp_time) else 0.0)) / running_time if running_time > 0 else np.nan,
                    **metrics,
                })
                fh.flush()
    fh.close()
    df = pd.read_csv(csv_path)
    plot_cutoff(df, outdir)
    return outdir


def k_for_strategy(strategy: str, n: int, m: int) -> int:
    if strategy in {"original", "K_m"}:
        return int(m)
    if strategy == "K_const_5":
        return 5
    if strategy == "K_const_10":
        return 10
    if strategy == "K_sqrt_mn":
        return int(math.ceil(math.sqrt(m * n)))
    if strategy == "K_sqrt_m":
        return int(math.ceil(math.sqrt(m)))
    if strategy == "K_2sqrt_m":
        return int(math.ceil(2 * math.sqrt(m)))
    raise ValueError(f"Unknown K strategy: {strategy}")


def generate_neighbors_for_k(score_mod, current_assignment, prob, n: int, m: int, strategy: str):
    start = time.time()
    current_pred = score_mod.assignment_to_pred_assort(current_assignment, n, m)
    timing = {
        "add_candidate_time": 0.0,
        "drop_candidate_time": 0.0,
        "neighbor_generation_time": 0.0,
        "convert_time": time.time() - start,
    }
    neighbors = []

    if strategy == "original":
        add_start = time.time()
        for k in range(m):
            pk = prob[k]
            zero_idx = np.where(current_pred[k] == 0)[0]
            if zero_idx.size > 0:
                add_j = zero_idx[np.argmax(pk[zero_idx])]
                neighbor_pred = current_pred.copy()
                neighbor_pred[k, add_j] = 1
                neighbors.append(score_mod.convert_pred_assort_to_assignment(neighbor_pred))
        timing["add_candidate_time"] = time.time() - add_start

        drop_start = time.time()
        for k in range(m):
            pk = prob[k]
            one_idx = np.where((current_pred[k] == 1) & (pk >= 0.5))[0]
            if one_idx.size > 0:
                rm_j = one_idx[np.argmin(pk[one_idx])]
                neighbor_pred = current_pred.copy()
                neighbor_pred[k, rm_j] = 0
                neighbors.append(score_mod.convert_pred_assort_to_assignment(neighbor_pred))
        timing["drop_candidate_time"] = time.time() - drop_start
        return neighbors, timing

    K = k_for_strategy(strategy, n, m)
    add_start = time.time()
    add_candidates = []
    for k in range(m):
        for j in range(n):
            if current_pred[k, j] == 0:
                add_candidates.append((k, j, float(prob[k, j]), "add"))
    add_candidates.sort(key=lambda x: x[2], reverse=True)
    add_list = add_candidates[:K]
    timing["add_candidate_time"] = time.time() - add_start

    drop_start = time.time()
    drop_candidates = []
    for k in range(m):
        for j in range(n):
            if current_pred[k, j] == 1 and prob[k, j] >= 0.5:
                drop_candidates.append((k, j, float(1.0 - prob[k, j]), "drop"))
    drop_candidates.sort(key=lambda x: x[2], reverse=True)
    drop_list = drop_candidates[:K]
    timing["drop_candidate_time"] = time.time() - drop_start

    gen_start = time.time()
    mixed = add_list + drop_list
    mixed.sort(key=lambda x: x[2], reverse=True)
    for k, j, _score, op_type in mixed:
        neighbor_pred = current_pred.copy()
        neighbor_pred[k, j] = 1 if op_type == "add" else 0
        neighbors.append(score_mod.convert_pred_assort_to_assignment(neighbor_pred))
    timing["neighbor_generation_time"] = time.time() - gen_start
    return neighbors, timing


def local_search_k(score_mod, initial_pred, prob, meta, strategy: str, max_iterations: int = 50, tolerance: float = 1e-6):
    n, segment_num, unit_cs, ship_cs, unit_us, Ns, _opt_bundles, _opt_prices, opt_rev, _running_time, _gap, stored_cs, stored_Rs = meta
    m = segment_num
    K = k_for_strategy(strategy, n, m)

    initial_ratio, _initial_time, current_assignment = score_mod.revenue_ratio_with_optimal_bundle(
        n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, initial_pred, stored_cs, stored_Rs
    )
    current_revenue, initial_lp_time = score_mod.revenue_ratio_LP(
        n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, current_assignment, stored_cs, stored_Rs
    )
    search_info = {
        "initial_milp_revenue": initial_ratio,
        "initial_lp_revenue": current_revenue,
        "iterations": 0,
        "improvements": 0,
        "lp_solver_calls": 1,
        "milp_solver_calls": 1,
        "total_iteration_time": 0.0,
        "total_lp_solve_time": initial_lp_time,
        "total_add_candidate_time": 0.0,
        "total_drop_candidate_time": 0.0,
        "total_neighbor_generation_time": 0.0,
        "total_neighbor_iteration_time": 0.0,
        "K": K,
        "max_neighbors_per_iter": 2 * K,
    }

    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        iter_start = time.time()
        neighbors, neighbor_timing = generate_neighbors_for_k(score_mod, current_assignment, prob, n, m, strategy)
        search_info["total_add_candidate_time"] += neighbor_timing["add_candidate_time"]
        search_info["total_drop_candidate_time"] += neighbor_timing["drop_candidate_time"]
        search_info["total_neighbor_generation_time"] += neighbor_timing["neighbor_generation_time"]

        for neighbor_assignment in neighbors:
            is_feasible, neighbor_revenue, lp_time = score_mod.check_lp_feasibility_and_revenue(
                neighbor_assignment, n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, stored_cs, stored_Rs
            )
            search_info["lp_solver_calls"] += 1
            search_info["total_lp_solve_time"] += lp_time
            search_info["total_neighbor_iteration_time"] += 0.000001
            if is_feasible and neighbor_revenue > current_revenue + tolerance:
                current_assignment = neighbor_assignment
                current_revenue = neighbor_revenue
                improved = True
                search_info["improvements"] += 1
                break
        search_info["total_iteration_time"] += time.time() - iter_start

    final_pred = score_mod.assignment_to_pred_assort(current_assignment, n, m)
    final_ratio, final_time = score_mod.revenue_ratio_with_optimal_bundle(
        n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, final_pred, stored_cs, stored_Rs
    )[:2]
    search_info["iterations"] = iteration
    search_info["final_milp_revenue"] = final_ratio
    search_info["final_milp_time"] = final_time
    search_info["milp_solver_calls"] += 1
    search_info["total_improvement"] = final_ratio - initial_ratio
    return final_pred, final_ratio, search_info


def run_k_justify(sample_count: int, time_limit: int | None, resume: bool):
    _test_fcp, _test_pcp, score = load_modules(time_limit=time_limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_seed1_model(score, device)
    outdir = OUT_ROOT / "k_justify"
    outdir.mkdir(parents=True, exist_ok=True)

    samples = []
    for ds in MB_DATASETS:
        for fname in selected_files(ds, sample_count):
            fpath = DATASET_DIR / ds / fname
            dat, misc = score.process_data(str(fpath))
            samples.append((ds, fname, dat, misc))

    long_path = outdir / "justify_K_all_results_long.csv"
    fieldnames = [
        "dataset", "sample_id", "strategy", "n_products", "revenue_ratio", "runtime_ratio",
        "total_time", "base_running_time", "improvement", "iterations", "improvements",
        "lp_solver_calls", "milp_solver_calls", "K", "max_neighbors_per_iter",
    ]
    done = set()
    if resume and long_path.exists():
        old = pd.read_csv(long_path)
        for r in old[["dataset", "sample_id", "strategy"]].itertuples(index=False):
            done.add((r.dataset, r.sample_id, r.strategy))
    write_header = not long_path.exists() or not resume
    with long_path.open("a" if resume else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for strategy in tqdm(K_STRATEGIES, desc="K strategy sweep"):
            for ds, fname, dat, misc in samples:
                if (ds, fname, strategy) in done:
                    continue
                n, segment_num, *_rest, running_time, _gap, _stored_cs, _stored_Rs = misc
                start = time.time()
                initial_pred, prob = score.predict_initial_bundles(dat, misc, model, device)
                _best_pred, best_rev, info = local_search_k(score, initial_pred, prob, misc, strategy)
                total_time = time.time() - start
                writer.writerow({
                    "dataset": ds,
                    "sample_id": fname,
                    "strategy": strategy,
                    "n_products": n,
                    "revenue_ratio": best_rev,
                    "runtime_ratio": total_time / running_time if running_time > 0 else np.nan,
                    "total_time": total_time,
                    "base_running_time": running_time,
                    "improvement": info.get("total_improvement", np.nan),
                    "iterations": info.get("iterations", 0),
                    "improvements": info.get("improvements", 0),
                    "lp_solver_calls": info.get("lp_solver_calls", 0),
                    "milp_solver_calls": info.get("milp_solver_calls", 0),
                    "K": info.get("K", 0),
                    "max_neighbors_per_iter": info.get("max_neighbors_per_iter", 0),
                })
                fh.flush()

    df = pd.read_csv(long_path)
    write_k_justify_compat_csvs(df, outdir)
    plot_k_justify(df, outdir)
    return outdir


def write_k_justify_compat_csvs(df: pd.DataFrame, outdir: Path):
    cols = [
        "n_products", "revenue_ratio", "runtime_ratio", "total_time", "base_running_time",
        "improvement", "iterations", "improvements", "lp_solver_calls", "milp_solver_calls",
        "K", "max_neighbors_per_iter",
    ]
    for (strategy, dataset), g in df.groupby(["strategy", "dataset"]):
        g[cols].to_csv(outdir / f"justify_K_{strategy}_{dataset}.csv", index=False)


def plot_k_justify(df: pd.DataFrame, outdir: Path):
    for col in ["revenue_ratio", "runtime_ratio", "total_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=220)
    for strategy in K_STRATEGIES:
        x_vals, y_vals, labels = [], [], []
        for ds in MB_DATASETS:
            g = df[(df["strategy"] == strategy) & (df["dataset"] == ds)]
            if g.empty:
                continue
            x_vals.append(g["total_time"].mean())
            y_vals.append(g["revenue_ratio"].mean())
            labels.append(DATASET_COMPACT_LABELS[ds])
        if x_vals:
            ax.plot(
                x_vals, y_vals, marker=K_MARKERS[strategy], linestyle="--",
                color=K_COLORS[strategy], label=K_LABELS[strategy],
                markersize=7, linewidth=1.8, alpha=0.82,
            )
            for xi, yi, label in zip(x_vals, y_vals, labels):
                ax.annotate(label, (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=7.5, alpha=0.72)
    ax.set_title("Competitive Ratio vs Absolute Time (Pareto View)", fontsize=12)
    ax.set_xlabel("Absolute Time (s) (Lower is Better)")
    ax.set_ylabel("Competitive Ratio (Higher is Better)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8.5)
    fig.suptitle("MB Datasets - Competitive Ratio vs Time Trade-off (Pareto View)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "K_justify_pareto_MB.png", bbox_inches="tight")
    fig.savefig(outdir / "K_justify_pareto_MB.pdf", bbox_inches="tight")
    plt.close(fig)
    plot_k_balance_bars(df, outdir)


def plot_k_balance_bars(df: pd.DataFrame, outdir: Path):
    agg = (
        df.groupby(["dataset", "strategy"])[["revenue_ratio", "total_time"]]
        .mean()
        .reset_index()
    )
    datasets = [ds for ds in MB_DATASETS if ds in set(agg["dataset"])]
    strategies = [s for s in K_STRATEGIES if s in set(agg["strategy"])]
    x = np.arange(len(datasets))
    width = 0.105

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.1), dpi=220, sharex=True)
    for idx, strategy in enumerate(strategies):
        offset = (idx - (len(strategies) - 1) / 2) * width
        rev_vals, time_vals = [], []
        for ds in datasets:
            row = agg[(agg["dataset"] == ds) & (agg["strategy"] == strategy)]
            rev_vals.append(float(row["revenue_ratio"].iloc[0]) if not row.empty else np.nan)
            time_vals.append(float(row["total_time"].iloc[0]) if not row.empty else np.nan)
        edgecolor = "#111111" if strategy == "K_sqrt_m" else "none"
        linewidth = 1.0 if strategy == "K_sqrt_m" else 0.0
        axes[0].bar(
            x + offset, rev_vals, width=width, label=K_LABELS[strategy],
            color=K_COLORS[strategy], alpha=0.86, edgecolor=edgecolor, linewidth=linewidth,
        )
        axes[1].bar(
            x + offset, time_vals, width=width, label=K_LABELS[strategy],
            color=K_COLORS[strategy], alpha=0.86, edgecolor=edgecolor, linewidth=linewidth,
        )

    axes[0].set_ylabel("Competitive Ratio")
    axes[0].set_ylim(max(0.965, float(np.nanmin(agg["revenue_ratio"])) - 0.004), min(1.0, float(np.nanmax(agg["revenue_ratio"])) + 0.003))
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].set_title("Solution Quality by K Strategy", fontsize=12, fontweight="bold")

    axes[1].set_ylabel("Absolute Time (s, log scale)")
    axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", alpha=0.25, which="both")
    axes[1].set_title("Runtime Cost by K Strategy", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DATASET_COMPACT_LABELS[ds] for ds in datasets])
    axes[1].set_xlabel("MB Dataset")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.925))
    fig.suptitle("K Strategy Quality-Time Balance on MB Datasets", fontsize=14, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.865])
    fig.savefig(outdir / "K_justify_balance_MB.png", bbox_inches="tight")
    fig.savefig(outdir / "K_justify_balance_MB.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cutoff(df: pd.DataFrame, outdir: Path):
    metric_cols = [
        "revenue_ratio", "time_ratio", "accuracy", "precision", "recall",
        "tpr", "tnr", "fpr", "fnr", "error_rate",
    ]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    agg = df.groupby(["cutoff", "strategy"])[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    colors = {"FCP": "#4C9BD6", "PCP": "#F28E2B"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), dpi=220)
    specs = [
        ("revenue_ratio", "Revenue Ratio", axes[0, 0], "(a) Revenue Ratio Comparison"),
        ("time_ratio", "Time Ratio", axes[0, 1], "(b) Time Ratio Comparison"),
        ("accuracy", "Accuracy", axes[1, 0], "(c) Accuracy Comparison"),
        ("recall", "Recall", axes[1, 1], "(d) Recall Comparison"),
    ]
    for metric, ylabel, ax, title in specs:
        for strategy in ["FCP", "PCP"]:
            g = agg[agg["strategy"] == strategy].sort_values("cutoff")
            x = g["cutoff"].to_numpy()
            y = g[f"{metric}_mean"].to_numpy()
            std = g[f"{metric}_std"].fillna(0).to_numpy()
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4.8, label=strategy, color=colors[strategy])
            ax.fill_between(x, y - std, y + std, color=colors[strategy], alpha=0.15)
        ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Cutoff Threshold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(CUTOFFS)
        ax.legend()
    fig.suptitle("FCP vs PCP: Cutoff Sensitivity Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "cutoff_comparison_FCP_PCP_combined.png", bbox_inches="tight")
    fig.savefig(outdir / "cutoff_comparison_FCP_PCP_combined.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cutoff", "k"], default="cutoff")
    parser.add_argument("--fcp-sample-count", type=int, default=30)
    parser.add_argument("--pcp-sample-count", type=int, default=10)
    parser.add_argument("--k-sample-count", type=int, default=30)
    # The final seed-1 Appendix C/D rerun used a 60-second per-solve limit.
    # This matters for PCP rows that return ObjBound when Gurobi reaches TIME_LIMIT.
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "package_root": str(PACKAGE_ROOT),
        "dataset_dir": str(DATASET_DIR),
        "model_path": str(MODEL_PATH),
        "time_limit": args.time_limit,
        "fcp_sample_count": args.fcp_sample_count,
        "pcp_sample_count": args.pcp_sample_count,
        "k_sample_count": args.k_sample_count,
        "note": "Reconstructed from the final 2026-07-02 seed-1 provenance; the original /tmp driver was not archived.",
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.mode == "cutoff":
        outdir = run_cutoff(args.fcp_sample_count, args.pcp_sample_count, args.time_limit, args.resume)
    elif args.mode == "k":
        outdir = run_k_justify(args.k_sample_count, args.time_limit, args.resume)
    print(f"output: {outdir}")


if __name__ == "__main__":
    main()
