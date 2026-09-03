#!/usr/bin/env python3
"""Verify fresh non-runtime results for Figures 6-8 and Tables 3-4."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def matches_paper_display(raw_value: float, published_value: float, decimals: int) -> bool:
    """Apply the agreed bounded tolerance around a displayed paper value."""
    display_unit = 10 ** -decimals
    return abs(raw_value - published_value) <= 0.55 * display_unit + 1e-12


def aggregate_complete_seed_samples(
    csv_path: Path,
    value_column: str,
    required_seeds: int = 10,
) -> dict:
    values = defaultdict(list)
    seeds = defaultdict(set)
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        sample = row["sample_file"]
        values[sample].append(float(row[value_column]))
        seeds[sample].add(int(row["seed"]))
    complete_values = {
        sample: statistics.fmean(values[sample])
        for sample in values
        if len(values[sample]) == required_seeds and len(seeds[sample]) == required_seeds
    }
    partial_samples = {
        sample: sorted(seeds[sample])
        for sample in values
        if sample not in complete_values
    }
    return {
        "raw_rows": len(rows),
        "complete_values": complete_values,
        "partial_samples": partial_samples,
    }


def _value_statistics(values: list[float]) -> dict:
    population_std = statistics.pstdev(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else population_std
    return {
        "mean": statistics.fmean(values),
        "population_std": population_std,
        "sample_std": sample_std,
    }


def select_matching_samples(
    values_by_sample: dict[str, float],
    expected_count: int,
    expected_mean: float,
    expected_std: float,
    decimals: int,
) -> dict:
    names = sorted(values_by_sample)
    remove_count = len(names) - expected_count
    if remove_count < 0 or remove_count > 1:
        return {
            "matched": False,
            "selected_samples": names,
            "excluded_samples": [],
            "reason": f"cannot select {expected_count} from {len(names)} complete samples",
        }

    matches = []
    exclusions = itertools.combinations(names, remove_count)
    for excluded_tuple in exclusions:
        excluded = set(excluded_tuple)
        selected = [name for name in names if name not in excluded]
        stats = _value_statistics([values_by_sample[name] for name in selected])
        mean_matches = matches_paper_display(stats["mean"], expected_mean, decimals)
        population_matches = matches_paper_display(
            stats["population_std"], expected_std, decimals
        )
        sample_matches = matches_paper_display(stats["sample_std"], expected_std, decimals)
        if mean_matches and (population_matches or sample_matches):
            matches.append(
                {
                    "matched": True,
                    "selected_samples": selected,
                    "excluded_samples": sorted(excluded),
                    "statistics": stats,
                    "accepted_std_convention": (
                        "population" if population_matches else "sample"
                    ),
                }
            )
    if len(matches) == 1:
        return matches[0]
    return {
        "matched": False,
        "selected_samples": names,
        "excluded_samples": [],
        "reason": f"matching subsets={len(matches)}, expected exactly one",
        "matching_candidates": matches,
    }


def _learned_path(output_root: Path, model_tag: str, method: str, dataset: str) -> Path:
    return (
        output_root
        / "combined"
        / model_tag
        / method.lower()
        / f"{dataset}_seed_sample.csv"
    )


def _bsp_path(output_root: Path, dataset: str) -> Path:
    return output_root / "combined" / "baseline" / "bsp" / f"{dataset}.csv"


def _verify_learned_cell(
    output_root: Path,
    experiment: str,
    model_tag: str,
    method: str,
    dataset: str,
    expected_count: int,
    expected_mean: float,
    expected_std: float,
    decimals: int,
) -> dict:
    csv_path = _learned_path(output_root, model_tag, method, dataset)
    value_column = "competitive_ratio" if method == "FCPLS" else "revenue_ratio"
    if not csv_path.exists():
        return {
            "experiment": experiment,
            "method": method,
            "dataset": dataset,
            "matched": False,
            "error": f"missing result: {csv_path}",
        }
    aggregate = aggregate_complete_seed_samples(csv_path, value_column)
    selection = select_matching_samples(
        aggregate["complete_values"],
        expected_count,
        expected_mean,
        expected_std,
        decimals,
    )
    return {
        "experiment": experiment,
        "method": method,
        "model_tag": model_tag,
        "dataset": dataset,
        "raw_rows": aggregate["raw_rows"],
        "complete_samples": len(aggregate["complete_values"]),
        "partial_samples": aggregate["partial_samples"],
        "expected_samples": expected_count,
        "expected_mean": expected_mean,
        "expected_std": expected_std,
        **selection,
    }


def _verify_bsp_cell(
    output_root: Path,
    experiment: str,
    dataset: str,
    expected_count: int,
    decimals: int,
) -> dict:
    csv_path = _bsp_path(output_root, dataset)
    if not csv_path.exists():
        return {
            "experiment": experiment,
            "method": "BSP",
            "dataset": dataset,
            "matched": False,
            "error": f"missing result: {csv_path}",
        }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [float(row["revenue_ratio"]) for row in rows]
    mean = statistics.fmean(values)
    return {
        "experiment": experiment,
        "method": "BSP",
        "dataset": dataset,
        "raw_rows": len(rows),
        "expected_samples": expected_count,
        "raw_mean": mean,
        "expected_mean": 1.0,
        "matched": len(rows) == expected_count and matches_paper_display(mean, 1.0, decimals),
    }


def _report(experiment: str, details: list[dict]) -> dict:
    mismatches = [
        f"{item.get('method')}/{item.get('dataset')}: {item.get('error', item.get('reason', 'value mismatch'))}"
        for item in details
        if not item.get("matched")
    ]
    return {
        "experiment": experiment,
        "passed": not mismatches,
        "checked_cells": len(details),
        "runtime_fields_checked": 0,
        "details": details,
        "mismatches": mismatches,
    }


def verify_figure6(published: dict, output_root: Path) -> dict:
    expected = published["figure6"]
    details = []
    for index, m_value in enumerate(expected["m"]):
        dataset = f"test_BSP_m{m_value}n20_correct_1e_3"
        count = int(expected["sample_count"][index])
        details.append(
            _verify_learned_cell(
                output_root,
                "figure6",
                "base",
                "FCP",
                dataset,
                count,
                float(expected["FCP_profit_mean"][index]),
                float(expected["FCP_profit_std"][index]),
                4,
            )
        )
        bsp_count = 30 if m_value == 80 else count
        details.append(_verify_bsp_cell(output_root, "figure6", dataset, bsp_count, 4))
    return _report("figure6", details)


def verify_figure7(published: dict, output_root: Path) -> dict:
    expected = published["figure7"]
    details = []
    for index, n_value in enumerate(expected["n"]):
        dataset = f"test_BSP_m10n{n_value}_correct_1e_3"
        for method in ("FCP", "PCP"):
            details.append(
                _verify_learned_cell(
                    output_root,
                    "figure7",
                    "base",
                    method,
                    dataset,
                    100,
                    float(expected[f"{method}_profit_mean"][index]),
                    float(expected[f"{method}_profit_std"][index]),
                    4,
                )
            )
        details.append(_verify_bsp_cell(output_root, "figure7", dataset, 100, 4))
    return _report("figure7", details)


def verify_figure8(published: dict, output_root: Path) -> dict:
    expected = published["figure8"]
    details = []
    for index, n_value in enumerate(expected["n"]):
        dataset = f"test_BSP_m10n{n_value}_correct_1e_3"
        for method, prefix in (("FCP", "FCP_I"), ("PCP", "PCP_I")):
            details.append(
                _verify_learned_cell(
                    output_root,
                    "figure8",
                    "self_improved",
                    method,
                    dataset,
                    100,
                    float(expected[f"{prefix}_profit_mean"][index]),
                    float(expected[f"{prefix}_profit_std"][index]),
                    4,
                )
            )
    return _report("figure8", details)


def _verify_table(
    experiment: str,
    expected: dict,
    output_root: Path,
    model_tag: str,
) -> dict:
    details = []
    method_mapping = {
        "FCP": "FCP",
        "PCP": "PCP",
        "FCPLS": "FCPLS",
        "FCP_I": "FCP",
        "PCP_I": "PCP",
        "FCPLS_I": "FCPLS",
    }
    for size, methods in expected.items():
        dataset = f"test_BSP_{size}_correct_1e_3"
        for paper_method, paper_values in methods.items():
            if paper_values is None:
                continue
            method = method_mapping[paper_method]
            detail = _verify_learned_cell(
                output_root,
                experiment,
                model_tag,
                method,
                dataset,
                100,
                float(paper_values[0]),
                float(paper_values[1]),
                3,
            )
            detail["paper_method"] = paper_method
            details.append(detail)
    return _report(experiment, details)


def verify_experiment(experiment: str, published: dict, output_root: Path) -> dict:
    if experiment == "figure6":
        return verify_figure6(published, output_root)
    if experiment == "figure7":
        return verify_figure7(published, output_root)
    if experiment == "figure8":
        return verify_figure8(published, output_root)
    if experiment == "table3":
        return _verify_table("table3", published["table3"], output_root, "base")
    if experiment == "table4":
        return _verify_table("table4", published["table4"], output_root, "self_improved")
    raise ValueError(f"unknown experiment: {experiment}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["figure6", "figure7", "figure8", "table3", "table4", "figures", "tables", "all"],
        required=True,
    )
    parser.add_argument("--published", type=Path, default=ROOT / "provenance" / "PUBLISHED_VALUES.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "remaining_exact_rerun")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    published = json.loads(args.published.read_text(encoding="utf-8"))
    groups = {
        "figures": ["figure6", "figure7", "figure8"],
        "tables": ["table3", "table4"],
        "all": ["figure6", "figure7", "figure8", "table3", "table4"],
    }
    experiments = groups.get(args.experiment, [args.experiment])
    reports = [verify_experiment(item, published, args.output_root) for item in experiments]
    payload = reports[0] if len(reports) == 1 else {
        "experiment": args.experiment,
        "passed": all(report["passed"] for report in reports),
        "runtime_fields_checked": 0,
        "reports": reports,
    }
    report_path = args.report or args.output_root / "verification" / f"{args.experiment}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
