#!/usr/bin/env python3
"""Guarded appendix reruns for cutoff, K justify, and LP/MILP checks.

This runner intentionally imports the May 29 cached-LP Local Search module:
src/test/test_FCPLS_score_cached_lp.py.  It does not import LS_Path_Test.py.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import __main__
import json
import math
import os
from pathlib import Path
import py_compile
import re
import sys
import time
import types
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "results/.mplconfig"))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_SHA256 = "da4e9c4942eb2b2d993d90a521a2be77952f26ede6a072272260416c577839dd"
CODE_ROOT = Path(__file__).resolve().parents[2]
GUARD_NOTE = CODE_ROOT / "provenance/codex_guard_sensitivity_latest_local_search_20260529.md"
PRIMARY_LS = CODE_ROOT / "src/test/test_FCPLS_score_cached_lp.py"
COMPAT_LS = CODE_ROOT / "src/test/test_FCPLS_score_cached_lp_1.py"
TEST_DIR = CODE_ROOT / "src/test"
DATASET2_DIR = CODE_ROOT / "dataset2_4_2026"
DATASET_DIR = CODE_ROOT / "Dataset"
MODEL_DIR = CODE_ROOT / "models_multi_layer_edge_update"

sys.path.insert(0, str(TEST_DIR))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_guard(output_dir: Path) -> Dict[str, object]:
    if not GUARD_NOTE.exists():
        raise FileNotFoundError(f"Guard note missing: {GUARD_NOTE}")
    note_text = GUARD_NOTE.read_text(encoding="utf-8")
    if EXPECTED_SHA256 not in note_text:
        raise RuntimeError("Guard note does not contain expected sha256")

    hashes = {
        str(PRIMARY_LS.relative_to(CODE_ROOT)): sha256_file(PRIMARY_LS),
        str(COMPAT_LS.relative_to(CODE_ROOT)): sha256_file(COMPAT_LS),
    }
    bad = {p: h for p, h in hashes.items() if h != EXPECTED_SHA256}
    if bad:
        raise RuntimeError(f"Local Search hash mismatch: {bad}")

    pycache_dir = output_dir / "_py_compile_cache"
    pycache_dir.mkdir(parents=True, exist_ok=True)
    compile_targets = []
    for path in (PRIMARY_LS, COMPAT_LS):
        cfile = pycache_dir / (path.name + ".pyc")
        py_compile.compile(str(path), cfile=str(cfile), doraise=True)
        compile_targets.append(str(cfile.relative_to(output_dir)))

    return {
        "guard_note": str(GUARD_NOTE),
        "expected_sha256": EXPECTED_SHA256,
        "hashes": hashes,
        "py_compile": "passed",
        "py_compile_cache": compile_targets,
    }


def load_ls_module():
    return importlib.import_module("test_FCPLS_score_cached_lp")


def load_pcp_module():
    path = TEST_DIR / "test_PCP.py"
    source = path.read_text(encoding="utf-8")
    module = types.ModuleType("test_PCP_future_annotations")
    module.__file__ = str(path)
    module.__dict__["__name__"] = module.__name__
    compiled = compile("from __future__ import annotations\n" + source, str(path), "exec")
    exec(compiled, module.__dict__)
    return module


def load_model(ls, layer: int, seed: int, device: torch.device):
    candidates = [
        MODEL_DIR / f"best_model_edge_{layer}layer_seed{seed}.pt",
        MODEL_DIR / f"model_edge_{layer}layer_seed{seed}.pt",
        MODEL_DIR / f"model-{layer}layer_seed{seed}.pt",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No model found for layer={layer}, seed={seed} in {MODEL_DIR}")

    previous = getattr(__main__, "EdgeScoringGCN", None)
    __main__.EdgeScoringGCN = ls.EdgeScoringGCN
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    finally:
        if previous is None:
            with contextlib.suppress(AttributeError):
                delattr(__main__, "EdgeScoringGCN")
        else:
            __main__.EdgeScoringGCN = previous
    model.to(device)
    model.eval()
    return model, str(path)


def sorted_msgpacks(dataset_path: Path) -> List[Path]:
    def natural_key(path: Path):
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]

    return sorted(
        (p for p in dataset_path.iterdir() if p.suffix == ".msgpack" and p.name != ".DS_Store"),
        key=natural_key,
    )


def sampled_msgpacks(dataset_path: Path, sample_count: int, seed: int = 42, random_sample: bool = False) -> List[Path]:
    files = sorted_msgpacks(dataset_path)
    if not random_sample or len(files) <= sample_count:
        return files[:sample_count]
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(files), size=sample_count, replace=False).tolist())
    return [files[i] for i in idx]


def has_size_limited_failure(path: Path) -> bool:
    if not path.exists():
        return False
    return "Model too large for size-limited license" in path.read_text(encoding="utf-8", errors="ignore")


def complete_csv(path: Path, expected_rows: int) -> bool:
    if not path.exists() or has_size_limited_failure(path):
        return False
    try:
        return len(pd.read_csv(path)) == expected_rows
    except Exception:
        return False


def infer_prob(ls, dat, meta, model, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        initial_pred, prob = ls.predict_initial_bundles(dat, meta, model, device)
    return initial_pred, prob


def compute_bundle_metrics(pred_assort: np.ndarray, opt_bundles, n: int, m: int) -> Dict[str, float]:
    opt_raw = np.array(opt_bundles)
    if opt_raw.shape == (m, n):
        opt_mn = opt_raw
    elif opt_raw.shape == (n, m):
        opt_mn = opt_raw.T
    else:
        opt_mn = opt_raw.T if opt_raw.shape[0] == n else opt_raw
    pred = np.asarray(pred_assort).astype(int)
    opt = np.asarray(opt_mn).astype(int)

    tp = int(np.sum((pred == 1) & (opt == 1)))
    fn = int(np.sum((pred == 0) & (opt == 1)))
    fp = int(np.sum((pred == 1) & (opt == 0)))
    tn = int(np.sum((pred == 0) & (opt == 0)))
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


def selected_products_to_pred_assort(selected_products: Sequence[Sequence[int]], n: int, m: int) -> np.ndarray:
    pred_assort = np.zeros((m, n), dtype=int)
    for k, prod_list in enumerate(selected_products):
        for j in prod_list:
            if 0 <= int(j) < n:
                pred_assort[k, int(j)] = 1
    return pred_assort


def run_cutoff(args, output_dir: Path, ls) -> Dict[str, object]:
    pcp = load_pcp_module()
    out_dir = output_dir / "cutoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_path = load_model(ls, args.layer, args.seed, device)

    cutoffs = [float(x) for x in args.cutoffs.split(",")]
    datasets = ["test_m10n10_1e_3", "test_m20n10_1e_3", "test_m30n10_1e_3"]
    fcp_rows: List[Dict[str, object]] = []
    pcp_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        fcp_chunk_csv = out_dir / f"cutoff_sensitivity_FCP_{dataset}.csv"
        pcp_chunk_csv = out_dir / f"cutoff_sensitivity_PCP_{dataset}.csv"
        files_fcp = sampled_msgpacks(DATASET2_DIR / dataset, args.cutoff_fcp_samples, random_sample=True)
        files_pcp = sampled_msgpacks(DATASET2_DIR / dataset, args.cutoff_pcp_samples, random_sample=True)
        expected_fcp_rows = len(files_fcp) * len(cutoffs)
        expected_pcp_rows = len(files_pcp) * len(cutoffs)
        if complete_csv(fcp_chunk_csv, expected_fcp_rows) and complete_csv(pcp_chunk_csv, expected_pcp_rows):
            print(f"Skipping completed cutoff dataset {dataset}")
            fcp_rows.extend(pd.read_csv(fcp_chunk_csv).to_dict("records"))
            pcp_rows.extend(pd.read_csv(pcp_chunk_csv).to_dict("records"))
            continue

        dataset_fcp_rows: List[Dict[str, object]] = []
        dataset_pcp_rows: List[Dict[str, object]] = []
        needed = {p.name: p for p in files_fcp + files_pcp}
        cached = {}
        for sample_name, path in tqdm(sorted(needed.items()), desc=f"cutoff inference {dataset}"):
            dat, meta = ls.process_data(str(path))
            n, m, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = meta
            t0 = time.time()
            _, prob_mn = infer_prob(ls, dat, meta, model, device)
            gcn_time = time.time() - t0
            cached[sample_name] = (meta, prob_mn, gcn_time)

        for path in tqdm(files_fcp, desc=f"FCP cutoff {dataset}"):
            meta, prob_mn, gcn_time = cached[path.name]
            n, m, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = meta
            for cutoff in cutoffs:
                pred_assort = (prob_mn >= cutoff).astype(int)
                try:
                    ratio, milp_time, _ = ls.competitive_ratio_with_optimal_bundle(
                        n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, pred_assort, stored_cs, stored_Rs
                    )
                    error = ""
                except Exception as exc:
                    ratio, milp_time = np.nan, np.nan
                    error = str(exc)
                    print(f"FCP cutoff failed {dataset}/{path.name} cutoff={cutoff}: {exc}")
                metrics = compute_bundle_metrics(pred_assort, opt_bundles, n, m)
                row = {
                    "strategy": "FCP",
                    "dataset": dataset,
                    "sample_id": path.name,
                    "cutoff": cutoff,
                    "revenue_ratio": ratio,
                    "time_ratio": (gcn_time + (milp_time if np.isfinite(milp_time) else 0.0)) / running_time if running_time > 0 else np.nan,
                    "error": error,
                }
                row.update(metrics)
                fcp_rows.append(row)
                dataset_fcp_rows.append(row)

        pd.DataFrame(dataset_fcp_rows).to_csv(fcp_chunk_csv, index=False)

        for path in tqdm(files_pcp, desc=f"PCP cutoff {dataset}"):
            meta, prob_mn, gcn_time = cached[path.name]
            n, m, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = meta
            prob_nm = prob_mn.T
            for cutoff in cutoffs:
                selected_products = pcp.top_m_selection(prob_nm, m=n, threshold=cutoff)
                feasible_bundles = pcp.generate_progressive_bundles(selected_products, n)
                pred_assort = selected_products_to_pred_assort(selected_products, n, m)
                try:
                    ratio, milp_time = pcp.revenue_ratio(
                        n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, feasible_bundles, selected_products, stored_cs, stored_Rs
                    )
                    error = ""
                except Exception as exc:
                    ratio, milp_time = np.nan, np.nan
                    error = str(exc)
                    print(f"PCP cutoff failed {dataset}/{path.name} cutoff={cutoff}: {exc}")
                metrics = compute_bundle_metrics(pred_assort, opt_bundles, n, m)
                row = {
                    "strategy": "PCP",
                    "dataset": dataset,
                    "sample_id": path.name,
                    "cutoff": cutoff,
                    "revenue_ratio": ratio,
                    "time_ratio": (gcn_time + (milp_time if np.isfinite(milp_time) else 0.0)) / running_time if running_time > 0 else np.nan,
                    "error": error,
                }
                row.update(metrics)
                pcp_rows.append(row)
                dataset_pcp_rows.append(row)

        pd.DataFrame(dataset_pcp_rows).to_csv(pcp_chunk_csv, index=False)

    fcp_csv = out_dir / "cutoff_sensitivity_FCP_results.csv"
    pcp_csv = out_dir / "cutoff_sensitivity_PCP_results.csv"
    combined_csv = out_dir / "cutoff_sensitivity_FCP_PCP_combined.csv"
    pd.DataFrame(fcp_rows).to_csv(fcp_csv, index=False)
    pd.DataFrame(pcp_rows).to_csv(pcp_csv, index=False)
    combined = pd.concat([pd.DataFrame(fcp_rows), pd.DataFrame(pcp_rows)], ignore_index=True)
    combined.to_csv(combined_csv, index=False)

    agg = combined.groupby(["strategy", "cutoff"], as_index=False).agg(
        revenue_ratio_mean=("revenue_ratio", "mean"),
        revenue_ratio_std=("revenue_ratio", "std"),
        time_ratio_mean=("time_ratio", "mean"),
        error_rate_mean=("error_rate", "mean"),
        fnr_mean=("fnr", "mean"),
        count=("revenue_ratio", "count"),
    )
    agg_csv = out_dir / "cutoff_sensitivity_aggregate.csv"
    agg.to_csv(agg_csv, index=False)

    for metric, ylabel, png_name in [
        ("revenue_ratio_mean", "Revenue Ratio", "cutoff_revenue_ratio.png"),
        ("time_ratio_mean", "Time Ratio", "cutoff_time_ratio.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in ["FCP", "PCP"]:
            data = agg[agg["strategy"] == strategy].sort_values("cutoff")
            ax.plot(data["cutoff"], data[metric], marker="o", linewidth=2, label=strategy)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("Cutoff")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / png_name, dpi=200)
        plt.close(fig)

    return {
        "status": "complete",
        "model_path": str(model_path),
        "csv": [str(fcp_csv), str(pcp_csv), str(combined_csv), str(agg_csv)],
        "figures": [str(out_dir / "cutoff_revenue_ratio.png"), str(out_dir / "cutoff_time_ratio.png")],
        "rows": {"fcp": len(fcp_rows), "pcp": len(pcp_rows), "combined": len(combined)},
    }


def k_value(strategy: str, m: int, n: int) -> int:
    if strategy == "K_m":
        return m
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
    if strategy == "original":
        return m
    raise ValueError(strategy)


def make_neighbor_generator(ls, strategy: str) -> Callable:
    def generator(current_assignment, prob, n, m):
        current_pred = ls.assignment_to_pred_assort(current_assignment, n, m)
        neighbors = []
        if strategy == "original":
            for k in range(m):
                pk = prob[k]
                zero_idx = np.where(current_pred[k] == 0)[0]
                if zero_idx.size:
                    add_j = int(zero_idx[np.argmax(pk[zero_idx])])
                    neighbor_pred = current_pred.copy()
                    neighbor_pred[k, add_j] = 1
                    neighbors.append(ls.convert_pred_assort_to_assignment(neighbor_pred))
            for k in range(m):
                pk = prob[k]
                one_idx = np.where(current_pred[k] == 1)[0]
                if one_idx.size:
                    rm_j = int(one_idx[np.argmax(1.0 - pk[one_idx])])
                    neighbor_pred = current_pred.copy()
                    neighbor_pred[k, rm_j] = 0
                    neighbors.append(ls.convert_pred_assort_to_assignment(neighbor_pred))
            return neighbors

        K = k_value(strategy, m, n)
        add_candidates = []
        for k in range(m):
            for j in range(n):
                if current_pred[k, j] == 0:
                    add_candidates.append((k, j, float(prob[k, j]), "add"))
        add_candidates.sort(key=lambda x: x[2], reverse=True)

        drop_candidates = []
        for k in range(m):
            for j in range(n):
                if current_pred[k, j] == 1:
                    drop_candidates.append((k, j, float(1.0 - prob[k, j]), "drop"))
        drop_candidates.sort(key=lambda x: x[2], reverse=True)

        mixed = add_candidates[:K] + drop_candidates[:K]
        mixed.sort(key=lambda x: x[2], reverse=True)
        for k, j, score, op_type in mixed:
            neighbor_pred = current_pred.copy()
            neighbor_pred[k, j] = 1 if op_type == "add" else 0
            neighbors.append(ls.convert_pred_assort_to_assignment(neighbor_pred))
        return neighbors

    return generator


@contextlib.contextmanager
def patched_neighbor_generator(ls, strategy: str):
    original = ls.generate_neighbor_assignments_global_topk
    ls.generate_neighbor_assignments_global_topk = make_neighbor_generator(ls, strategy)
    try:
        yield
    finally:
        ls.generate_neighbor_assignments_global_topk = original


def run_one_local_search_strategy(ls, strategy: str, initial_assignment, initial_ratio, prob, meta, max_iterations: int, tolerance: float):
    with patched_neighbor_generator(ls, strategy):
        return ls.local_search_with_lp_global_topk(
            initial_assignment,
            initial_ratio,
            prob,
            meta,
            max_iterations=max_iterations,
            tolerance=tolerance,
            silent=True,
            record_history=False,
        )


def run_k_justify(args, output_dir: Path, ls) -> Dict[str, object]:
    out_dir = output_dir / "k_justify"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_path = load_model(ls, args.layer, args.seed, device)
    strategies = ["original", "K_m", "K_const_5", "K_const_10", "K_sqrt_mn", "K_sqrt_m", "K_2sqrt_m"]
    datasets = {
        "test_m10n10_1e_3": DATASET2_DIR / "test_m10n10_1e_3",
        "test_m20n10_1e_3": DATASET2_DIR / "test_m20n10_1e_3",
        "test_m30n10_1e_3": DATASET2_DIR / "test_m30n10_1e_3",
        "test_BSP_m10n20_1e_3": DATASET2_DIR / "test_BSP_m10n20_1e_3",
        "test_BSP_m20n20_1e_3": DATASET2_DIR / "test_BSP_m20n20_1e_3",
    }

    csv_paths = []
    rows_all = []
    header = [
        "sample_id",
        "n_products",
        "m_segments",
        "revenue_ratio",
        "runtime_ratio",
        "total_time",
        "base_running_time",
        "initial_revenue",
        "improvement",
        "iterations",
        "improvements",
        "lp_solver_calls",
        "milp_solver_calls",
        "K",
        "max_neighbors_per_iter",
        "lp_cache_models",
        "lp_cache_hits",
    ]

    for dataset, dataset_path in datasets.items():
        files = sampled_msgpacks(dataset_path, args.k_samples, random_sample=False)
        for strategy in strategies:
            csv_path = out_dir / f"justify_K_{strategy}_{dataset}.csv"
            if complete_csv(csv_path, len(files)):
                print(f"Skipping completed K justify {strategy} {dataset}")
                existing = pd.read_csv(csv_path)
                csv_paths.append(str(csv_path))
                rows_all.extend(existing.to_dict("records"))
                continue

            strategy_rows = []
            for path in tqdm(files, desc=f"K {strategy} {dataset}"):
                n = np.nan
                m = np.nan
                running_time = np.nan
                try:
                    dat, meta = ls.process_data(str(path))
                    n, m, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = meta
                    t0 = time.time()
                    initial_pred, prob = infer_prob(ls, dat, meta, model, device)
                    initial_ratio, initial_milp_time, initial_assignment = ls.competitive_ratio_with_optimal_bundle(
                        n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, initial_pred, stored_cs, stored_Rs
                    )
                    _, final_ratio, info = run_one_local_search_strategy(
                        ls, strategy, initial_assignment, initial_ratio, prob, meta, args.max_iterations, args.tolerance
                    )
                    total_time = time.time() - t0
                    row = {
                        "strategy": strategy,
                        "dataset": dataset,
                        "sample_id": path.name,
                        "n_products": n,
                        "m_segments": m,
                        "revenue_ratio": final_ratio,
                        "runtime_ratio": total_time / running_time if running_time > 0 else np.nan,
                        "total_time": total_time,
                        "base_running_time": running_time,
                        "initial_revenue": initial_ratio,
                        "improvement": info.get("total_improvement", np.nan),
                        "iterations": info.get("iterations", np.nan),
                        "improvements": info.get("improvements", np.nan),
                        "lp_solver_calls": info.get("lp_solver_calls", np.nan),
                        "milp_solver_calls": info.get("milp_solver_calls", np.nan),
                        "K": k_value(strategy, m, n),
                        "max_neighbors_per_iter": 2 * k_value(strategy, m, n),
                        "lp_cache_models": info.get("lp_cache_models", np.nan),
                        "lp_cache_hits": info.get("lp_cache_hits", np.nan),
                    }
                    strategy_rows.append(row)
                    rows_all.append(row)
                except Exception as exc:
                    row = {
                        "strategy": strategy,
                        "dataset": dataset,
                        "sample_id": path.name,
                        "n_products": n,
                        "m_segments": m,
                        "revenue_ratio": np.nan,
                        "runtime_ratio": np.nan,
                        "total_time": np.nan,
                        "base_running_time": running_time,
                        "initial_revenue": np.nan,
                        "improvement": np.nan,
                        "iterations": np.nan,
                        "improvements": np.nan,
                        "lp_solver_calls": np.nan,
                        "milp_solver_calls": np.nan,
                        "K": k_value(strategy, int(m), int(n)) if np.isfinite(m) and np.isfinite(n) else np.nan,
                        "max_neighbors_per_iter": 2 * k_value(strategy, int(m), int(n)) if np.isfinite(m) and np.isfinite(n) else np.nan,
                        "lp_cache_models": np.nan,
                        "lp_cache_hits": np.nan,
                        "error": str(exc),
                    }
                    strategy_rows.append(row)
                    rows_all.append(row)
                    print(f"K justify failed {strategy} {dataset}/{path.name}: {exc}")
                finally:
                    ls.clear_internal_caches()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header + ["error"])
                writer.writeheader()
                for row in strategy_rows:
                    writer.writerow({k: row.get(k) for k in header + ["error"]})
            csv_paths.append(str(csv_path))

    all_df = pd.DataFrame(rows_all)
    all_csv = out_dir / "justify_K_all_results_long.csv"
    all_df.to_csv(all_csv, index=False)
    summary = all_df.groupby(["strategy", "dataset"], as_index=False).agg(
        revenue_ratio_mean=("revenue_ratio", "mean"),
        revenue_ratio_std=("revenue_ratio", "std"),
        runtime_ratio_mean=("runtime_ratio", "mean"),
        runtime_ratio_std=("runtime_ratio", "std"),
        lp_solver_calls_mean=("lp_solver_calls", "mean"),
        iterations_mean=("iterations", "mean"),
        count=("revenue_ratio", "count"),
    )
    summary_csv = out_dir / "justify_K_summary.csv"
    summary.to_csv(summary_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for strategy in strategies:
        data = summary[summary["strategy"] == strategy]
        ax.scatter(data["runtime_ratio_mean"], data["revenue_ratio_mean"], label=strategy, s=45)
    ax.set_xlabel("Mean Runtime Ratio")
    ax.set_ylabel("Mean Revenue Ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    pareto_png = out_dir / "K_justify_pareto.png"
    fig.savefig(pareto_png, dpi=200)
    plt.close(fig)

    return {
        "status": "complete",
        "model_path": str(model_path),
        "csv": csv_paths + [str(all_csv), str(summary_csv)],
        "figures": [str(pareto_png)],
        "rows": len(rows_all),
    }


def run_lp_milp(args, output_dir: Path, ls) -> Dict[str, object]:
    out_dir = output_dir / "lp_milp"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_path = load_model(ls, args.layer, args.seed, device)
    datasets = {
        "m10_n10_sample_100": DATASET_DIR / "m10_n10_sample_100",
        "m20_n10_sample_100": DATASET_DIR / "m20_n10_sample_100",
        "m30_n10_sample_100": DATASET_DIR / "m30_n10_sample_100",
        "test_BSP_m10n15": DATASET_DIR / "test_BSP_m10n15",
        "test_BSP_m10n20": DATASET_DIR / "test_BSP_m10n20",
        "test_BSP_m10n25": DATASET_DIR / "test_BSP_m10n25",
    }
    checkpoint_json = out_dir / "LP_MILP_verification_results_checkpoint.json"
    checkpoint_failures_json = out_dir / "LP_MILP_verification_failures_checkpoint.json"
    if checkpoint_json.exists():
        all_results = json.loads(checkpoint_json.read_text(encoding="utf-8"))
    else:
        all_results = []
    if checkpoint_failures_json.exists():
        failures = json.loads(checkpoint_failures_json.read_text(encoding="utf-8"))
    else:
        failures = []
    done = {
        (str(res.get("dataset_name")), str(res.get("file_name")))
        for res in all_results
    } | {
        (str(res.get("dataset")), str(res.get("file_name")))
        for res in failures
        if "Model too large for size-limited license" not in str(res.get("error", ""))
    }

    for dataset, dataset_path in datasets.items():
        files = sampled_msgpacks(dataset_path, args.lp_milp_samples, random_sample=False)
        for sample_idx, path in enumerate(tqdm(files, desc=f"LP-MILP {dataset}")):
            if (dataset, path.name) in done:
                print(f"Skipping completed LP-MILP {dataset}/{path.name}")
                continue
            try:
                dat, meta = ls.process_data(str(path))
                n, m, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = meta
                initial_pred, prob = infer_prob(ls, dat, meta, model, device)
                initial_milp_ratio, initial_milp_time, current_assignment = ls.competitive_ratio_with_optimal_bundle(
                    n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, initial_pred, stored_cs, stored_Rs
                )
                lp_cache = ls.ReusableBundleLPCache(n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, stored_cs, stored_Rs)
                ok, current_lp_ratio, current_obj, initial_lp_time = lp_cache.solve(current_assignment)
                if not ok:
                    raise RuntimeError("initial cached LP infeasible")

                lp_path = [current_lp_ratio]
                milp_path = [initial_milp_ratio]
                translations = []
                iteration = 0
                improved = True
                current_milp = initial_milp_ratio
                while improved and iteration < args.max_iterations:
                    improved = False
                    iteration += 1
                    neighbors = ls.generate_neighbor_assignments_global_topk(current_assignment, prob, n, m)
                    for neighbor in neighbors:
                        feasible, neighbor_lp_ratio, neighbor_obj, _ = lp_cache.solve(neighbor)
                        objective_improvement = neighbor_obj - current_obj
                        objective_improvement_ratio = (
                            objective_improvement / max(abs(current_obj), 1e-12)
                            if np.isfinite(current_obj)
                            else -np.inf
                        )
                        if feasible and objective_improvement_ratio > args.tolerance:
                            neighbor_pred = ls.assignment_to_pred_assort(neighbor, n, m)
                            neighbor_milp, _, _ = ls.competitive_ratio_with_optimal_bundle(
                                n, m, unit_cs, ship_cs, unit_us, Ns, opt_rev, neighbor_pred, stored_cs, stored_Rs
                            )
                            lp_path.append(neighbor_lp_ratio)
                            milp_path.append(neighbor_milp)
                            translations.append(bool(neighbor_milp > current_milp + 1e-12))
                            current_assignment = neighbor
                            current_lp_ratio = neighbor_lp_ratio
                            current_obj = neighbor_obj
                            current_milp = neighbor_milp
                            improved = True
                            break

                lp_cache.dispose()
                result = {
                    "dataset_name": dataset,
                    "sample_id": sample_idx,
                    "file_name": path.name,
                    "m": m,
                    "n": n,
                    "opt_rev": float(opt_rev),
                    "lp_path": lp_path,
                    "milp_path": milp_path,
                    "lp_to_milp_translation": translations,
                    "all_translated": all(translations) if translations else True,
                    "improvement_count": len(translations),
                    "initial_lp_profit": lp_path[0],
                    "initial_milp_profit": milp_path[0],
                    "final_lp_profit": lp_path[-1],
                    "final_milp_profit": milp_path[-1],
                    "iterations": iteration,
                }
                all_results.append(result)
                done.add((dataset, path.name))
            except Exception as exc:
                failure = {"dataset": dataset, "file_name": path.name, "error": str(exc)}
                failures.append(failure)
                if "Model too large for size-limited license" not in str(exc):
                    done.add((dataset, path.name))
                print(f"LP-MILP failed {dataset}/{path.name}: {exc}")
            finally:
                checkpoint_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
                checkpoint_failures_json.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
                ls.clear_internal_caches()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    stats = calculate_lp_milp_stats(all_results)
    results_json = out_dir / "LP_MILP_verification_results.json"
    stats_json = out_dir / "LP_MILP_verification_statistics.json"
    failures_json = out_dir / "LP_MILP_verification_failures.json"
    results_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    failures_json.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for res in all_results:
        rows.append(
            {
                "dataset": res["dataset_name"],
                "sample_id": res["sample_id"],
                "file_name": res["file_name"],
                "improvement_count": res["improvement_count"],
                "all_translated": res["all_translated"],
                "initial_lp": res["initial_lp_profit"],
                "final_lp": res["final_lp_profit"],
                "initial_milp": res["initial_milp_profit"],
                "final_milp": res["final_milp_profit"],
            }
        )
    summary_csv = out_dir / "LP_MILP_verification_sample_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(stats["by_dataset"].keys())
    rates = [stats["by_dataset"][k]["translation_rate"] for k in labels]
    ax.bar(range(len(labels)), rates)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Translation Rate")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    stats_png = out_dir / "LP_MILP_verification_statistics.png"
    fig.savefig(stats_png, dpi=200)
    plt.close(fig)

    return {
        "status": "complete",
        "model_path": str(model_path),
        "json": [str(results_json), str(stats_json), str(failures_json)],
        "csv": [str(summary_csv)],
        "figures": [str(stats_png)],
        "samples": len(all_results),
        "failures": len(failures),
    }


def calculate_lp_milp_stats(all_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for res in all_results:
        by_dataset.setdefault(str(res["dataset_name"]), []).append(res)

    stats = {"by_dataset": {}, "by_step": {}, "overall": {"total": 0, "success": 0, "translation_rate": 0.0}}
    for name, samples in by_dataset.items():
        total = int(sum(int(s["improvement_count"]) for s in samples))
        success = int(sum(sum(bool(x) for x in s["lp_to_milp_translation"]) for s in samples))
        all_translated = int(sum(1 for s in samples if bool(s["all_translated"])))
        stats["by_dataset"][name] = {
            "total_improvements": total,
            "successful_translations": success,
            "translation_rate": success / total if total else 1.0,
            "samples_with_all_translated": all_translated,
            "total_samples": len(samples),
            "sample_success_rate": all_translated / len(samples) if samples else 0.0,
        }
    improved_samples = [s for s in all_results if s["lp_to_milp_translation"]]
    max_steps = max((len(s["lp_to_milp_translation"]) for s in improved_samples), default=0)
    for step in range(max_steps):
        vals = [bool(s["lp_to_milp_translation"][step]) for s in improved_samples if len(s["lp_to_milp_translation"]) > step]
        if vals:
            stats["by_step"][str(step + 1)] = {"success": int(sum(vals)), "total": len(vals), "rate": float(sum(vals) / len(vals))}
    stats["overall"]["total"] = int(sum(s["improvement_count"] for s in all_results))
    stats["overall"]["success"] = int(sum(sum(bool(x) for x in s["lp_to_milp_translation"]) for s in all_results))
    if stats["overall"]["total"]:
        stats["overall"]["translation_rate"] = stats["overall"]["success"] / stats["overall"]["total"]
    else:
        stats["overall"]["translation_rate"] = 1.0
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["cutoff", "k_justify", "lp_milp", "all"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cutoffs", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--cutoff-fcp-samples", type=int, default=30)
    parser.add_argument("--cutoff-pcp-samples", type=int, default=10)
    parser.add_argument("--k-samples", type=int, default=100)
    parser.add_argument("--lp-milp-samples", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_result = verify_guard(output_dir)
    ls = load_ls_module()

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "code_root": str(CODE_ROOT),
        "guard": guard_result,
        "experiments": {},
        "args": vars(args) | {"output_dir": str(args.output_dir)},
    }

    if args.experiment in ("cutoff", "all"):
        manifest["experiments"]["cutoff"] = run_cutoff(args, output_dir, ls)
    if args.experiment in ("k_justify", "all"):
        manifest["experiments"]["k_justify"] = run_k_justify(args, output_dir, ls)
    if args.experiment in ("lp_milp", "all"):
        manifest["experiments"]["lp_milp"] = run_lp_milp(args, output_dir, ls)

    manifest_path = output_dir / "rerun_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest saved: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
