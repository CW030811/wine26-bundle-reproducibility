#!/usr/bin/env python3
"""Compare fresh Figures 9--10 CSVs with the archived paper-run CSVs.

Runtime fields are reported but excluded from the pass/fail decision because they
are machine-dependent. All experiment keys and every other field must match.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "figure9": {
        "reference": ROOT / "artifacts/appendix_cd/cutoff_sensitivity_FCP_PCP_combined.csv",
        "fresh": ROOT / "results/appendix_cd_seed1/cutoff/cutoff_sensitivity_FCP_PCP_combined.csv",
        "keys": ["dataset", "sample_id", "cutoff", "strategy"],
        "ignored": ["time_ratio"],
    },
    "figure10": {
        "reference": ROOT / "artifacts/appendix_cd/justify_K_all_results_long.csv",
        "fresh": ROOT / "results/appendix_cd_seed1/k_justify/justify_K_all_results_long.csv",
        "keys": ["dataset", "sample_id", "strategy"],
        "ignored": ["runtime_ratio", "total_time", "base_running_time"],
    },
}


def compare(name: str, atol: float) -> bool:
    case = CASES[name]
    reference = pd.read_csv(case["reference"])
    fresh = pd.read_csv(case["fresh"])
    keys = case["keys"]
    ignored = case["ignored"]

    print(f"\n=== {name} ===")
    print(f"rows: archived={len(reference)}, fresh={len(fresh)}")
    if reference.duplicated(keys).any() or fresh.duplicated(keys).any():
        print("FAIL: duplicate experiment keys")
        return False

    merged = reference.merge(
        fresh,
        on=keys,
        how="outer",
        suffixes=("_archived", "_fresh"),
        indicator=True,
        validate="one_to_one",
    )
    key_match = bool((merged["_merge"] == "both").all())
    print(f"key sets identical: {key_match}")
    passed = key_match

    for column in [c for c in reference.columns if c not in keys + ignored]:
        archived = merged[f"{column}_archived"]
        current = merged[f"{column}_fresh"]
        if pd.api.types.is_numeric_dtype(archived):
            a = archived.to_numpy(dtype=float)
            b = current.to_numpy(dtype=float)
            equal_nan = np.isnan(a) & np.isnan(b)
            differences = np.abs(a - b)
            mismatches = ~(equal_nan | (differences <= atol))
            count = int(mismatches.sum())
            maximum = float(np.nanmax(differences)) if differences.size else 0.0
            print(f"{column}: mismatches={count}, max_abs_diff={maximum:.17g}")
        else:
            mismatches = archived.fillna("<NA>").astype(str) != current.fillna("<NA>").astype(str)
            count = int(mismatches.sum())
            print(f"{column}: mismatches={count}")
        if count:
            detail_columns = keys + [f"{column}_archived", f"{column}_fresh"]
            print(merged.loc[mismatches, detail_columns].head(10).to_string(index=False))
        passed &= count == 0

    print(f"ignored runtime fields: {', '.join(ignored)}")
    print("PASS" if passed else "FAIL")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=["figure9", "figure10", "all"], default="all")
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    names = list(CASES) if args.figure == "all" else [args.figure]
    results = [compare(name, args.atol) for name in names]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
