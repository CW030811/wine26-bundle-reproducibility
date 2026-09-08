#!/usr/bin/env python3
"""Verify regenerated main-paper statistics while excluding runtime fields."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _display_3(value: float) -> float:
    """Round to the paper's three-decimal display precision."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _within_paper_display_tolerance(
    raw_value: float,
    published_value: float,
    decimals: int = 3,
    boundary_slack: float = 0.00005,
) -> bool:
    """Accept only rounding-bin noise plus a small, explicit numerical slack."""
    half_display_unit = 0.5 * (10 ** -decimals)
    return abs(raw_value - published_value) <= half_display_unit + boundary_slack + 1e-12


def verify_table2(
    published_path: Path,
    result_paths: dict[tuple[str, str], Path],
    expected_samples: int = 100,
    expected_seeds: int = 10,
) -> dict:
    """Verify Table 2 profit statistics; runtime is intentionally excluded."""
    published = json.loads(Path(published_path).read_text(encoding="utf-8"))["table2"]
    value_columns = {
        "FCP": "revenue_ratio",
        "PCP": "revenue_ratio",
        "FCPLS": "competitive_ratio",
        "BSP": "revenue_ratio",
    }
    seed_methods = {"FCP", "PCP", "FCPLS"}
    details = []
    mismatches = []

    for size, methods in published.items():
        for method, expected in methods.items():
            path = Path(result_paths[(size, method)])
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            completeness_errors = []
            if method in seed_methods:
                sample_values = defaultdict(list)
                sample_seeds = defaultdict(set)
                for row in rows:
                    sample = row["sample_file"]
                    sample_values[sample].append(float(row[value_columns[method]]))
                    sample_seeds[sample].add(int(row["seed"]))
                values = [statistics.fmean(group) for group in sample_values.values()]
                for sample in sample_values:
                    if len(sample_values[sample]) != expected_seeds or len(sample_seeds[sample]) != expected_seeds:
                        completeness_errors.append(
                            f"{sample}: rows={len(sample_values[sample])}, unique_seeds={len(sample_seeds[sample])}"
                        )
            else:
                values = [float(row[value_columns[method]]) for row in rows]

            if len(values) != expected_samples:
                completeness_errors.append(
                    f"aggregated_samples={len(values)}, expected={expected_samples}"
                )

            wanted = [float(expected[0]), float(expected[1])]
            mean_raw = statistics.fmean(values)
            population_std_raw = statistics.pstdev(values)
            sample_std_raw = statistics.stdev(values) if len(values) > 1 else population_std_raw
            mean_display = _display_3(mean_raw)
            population_std_display = _display_3(population_std_raw)
            sample_std_display = _display_3(sample_std_raw)
            if mean_display == wanted[0]:
                mean_match_rule = "rounded_display"
            elif _within_paper_display_tolerance(mean_raw, wanted[0]):
                mean_match_rule = "rounding_boundary_tolerance"
            else:
                mean_match_rule = None
            if population_std_display == wanted[1]:
                accepted_std_convention = "population"
                std_match_rule = "rounded_display"
                accepted_std_display = population_std_display
            elif sample_std_display == wanted[1]:
                accepted_std_convention = "sample"
                std_match_rule = "rounded_display"
                accepted_std_display = sample_std_display
            elif _within_paper_display_tolerance(population_std_raw, wanted[1]):
                accepted_std_convention = "population"
                std_match_rule = "rounding_boundary_tolerance"
                accepted_std_display = population_std_display
            elif _within_paper_display_tolerance(sample_std_raw, wanted[1]):
                accepted_std_convention = "sample"
                std_match_rule = "rounding_boundary_tolerance"
                accepted_std_display = sample_std_display
            else:
                accepted_std_convention = None
                std_match_rule = None
                accepted_std_display = population_std_display

            observed = [mean_display, accepted_std_display]
            matched = (
                mean_match_rule is not None
                and accepted_std_convention is not None
                and not completeness_errors
            )
            details.append(
                {
                    "size": size,
                    "method": method,
                    "raw_rows": len(rows),
                    "aggregated_samples": len(values),
                    "observed_non_runtime": observed,
                    "published_non_runtime": wanted,
                    "raw_mean": mean_raw,
                    "raw_population_std": population_std_raw,
                    "raw_sample_std": sample_std_raw,
                    "mean_match_rule": mean_match_rule,
                    "accepted_std_convention": accepted_std_convention,
                    "std_match_rule": std_match_rule,
                    "completeness_errors": completeness_errors,
                    "matched": matched,
                }
            )
            if not matched:
                mismatches.append(
                    f"{size}/{method}: observed={observed}, published={wanted}, "
                    f"mean_rule={mean_match_rule}, std_rule={std_match_rule}, "
                    f"completeness={completeness_errors}"
                )

    expected_cells = sum(len(methods) for methods in published.values())
    return {
        "experiment": "table2",
        "passed": not mismatches and len(details) == expected_cells,
        "checked_cells": len(details),
        "runtime_fields_checked": 0,
        "details": details,
        "mismatches": mismatches,
    }


def verify_table6(
    published_path: Path,
    training_summary_path: Path,
    seed_sample_path: Path,
    require_complete: bool = True,
) -> dict:
    published = json.loads(Path(published_path).read_text(encoding="utf-8"))["table6"]
    with Path(training_summary_path).open(newline="", encoding="utf-8") as handle:
        summaries = {str(int(row["seed"])): row for row in csv.DictReader(handle)}

    ratios = defaultdict(list)
    identities = defaultdict(list)
    with Path(seed_sample_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ratios[str(int(row["seed"]))].append(float(row["revenue_ratio"]))
            identities[str(int(row['seed']))].append(row.get('sample_file'))

    details = []
    mismatches = []
    expected_inputs = {path.name for path in (ROOT / 'data/deterministic/test_m10n10_correct_1e_3').glob('*.msgpack')}
    if require_complete and set(ratios) != set(published):
        mismatches.append('Unexpected or missing seed groups')
    for seed, expected in sorted(published.items(), key=lambda item: int(item[0])):
        if seed not in summaries or seed not in ratios:
            mismatches.append(f"seed {seed}: missing training summary or regenerated results")
            continue

        observed = [
            round(float(summaries[seed]["final_train_loss"]), 4),
            round(float(summaries[seed]["best_val_loss"]), 4),
            round(statistics.fmean(ratios[seed]), 3),
            round(statistics.pstdev(ratios[seed]), 3),
        ]
        wanted = [float(expected[0]), float(expected[1]), float(expected[2]), float(expected[3])]
        complete = (not require_complete or (len(identities[seed]) == 100 and len(set(identities[seed])) == 100
                    and set(identities[seed]) == expected_inputs))
        matched = observed == wanted and complete and all(math.isfinite(value) for value in ratios[seed])
        details.append(
            {
                "seed": int(seed),
                "samples": len(ratios[seed]),
                "observed_non_runtime": observed,
                "published_non_runtime": wanted,
                "matched": matched,
            }
        )
        if not matched:
            mismatches.append(f"seed {seed}: observed={observed}, published={wanted}")

    return {
        "experiment": "table6",
        "passed": not mismatches and len(details) == len(published),
        "checked_seeds": len(details),
        "runtime_fields_checked": 0,
        "details": details,
        "mismatches": mismatches,
    }


def verify_table7(published_path: Path, result_paths: dict[str, Path], require_complete: bool = True) -> dict:
    published = json.loads(Path(published_path).read_text(encoding="utf-8"))["table7"]
    details = []
    mismatches = []
    for variant, expected in published.items():
        path = Path(result_paths[variant])
        with path.open(newline="", encoding="utf-8") as handle:
            ratios = [float(row["revenue_ratio"]) for row in csv.DictReader(handle)]
        complete = True
        if require_complete:
            long_path = path.with_name(path.stem + '_seed_sample.csv')
            grouped = defaultdict(list)
            with long_path.open(newline='', encoding='utf-8') as handle:
                for row in csv.DictReader(handle):
                    grouped[row['sample_file']].append((int(row['seed']), float(row['revenue_ratio'])))
            dataset = path.stem.removeprefix('test_result_FCP_4layer_')
            expected_inputs = {item.name for item in (ROOT / 'data/ood' / dataset).glob('*.msgpack')}
            complete = (len(grouped) == 100 and set(grouped) == expected_inputs and len(ratios) == 100
                        and all(len(values) == 10 and {seed for seed, _ in values} == set(range(1, 11))
                                and all(math.isfinite(value) for _, value in values) for values in grouped.values()))
            from_long = [statistics.fmean(value for _, value in values) for values in grouped.values()]
            complete = complete and all(math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10)
                                        for a, b in zip(sorted(ratios), sorted(from_long)))
            ratios = from_long
        wanted = [float(expected[0]), float(expected[1])]
        mean_raw = statistics.fmean(ratios)
        population_std_raw = statistics.pstdev(ratios)
        sample_std_raw = statistics.stdev(ratios) if len(ratios) > 1 else population_std_raw
        mean_display = round(mean_raw, 3)
        population_std_display = round(population_std_raw, 3)
        sample_std_display = round(sample_std_raw, 3)
        if population_std_display == wanted[1]:
            accepted_std_convention = "population"
            accepted_std_display = population_std_display
        elif sample_std_display == wanted[1]:
            accepted_std_convention = "sample"
            accepted_std_display = sample_std_display
        else:
            accepted_std_convention = None
            accepted_std_display = population_std_display
        observed = [mean_display, accepted_std_display]
        matched = mean_display == wanted[0] and accepted_std_convention is not None and complete
        details.append(
            {
                "variant": variant,
                "samples": len(ratios),
                "observed_non_runtime": observed,
                "published_non_runtime": wanted,
                "raw_mean": mean_raw,
                "raw_population_std": population_std_raw,
                "raw_sample_std": sample_std_raw,
                "accepted_std_convention": accepted_std_convention,
                "matched": matched,
            }
        )
        if not matched:
            mismatches.append(f"{variant}: observed={observed}, published={wanted}")

    return {
        "experiment": "table7",
        "passed": not mismatches and len(details) == len(published),
        "checked_variants": len(details),
        "runtime_fields_checked": 0,
        "details": details,
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["table2", "table6", "table7"], default="table6")
    parser.add_argument(
        "--published",
        type=Path,
        default=ROOT / "provenance" / "PUBLISHED_VALUES.json",
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=ROOT / "models" / "main_base_4layer_correct_lr_3" / "seed_training_summary.csv",
    )
    parser.add_argument(
        "--seed-sample-results",
        type=Path,
        default=(
            ROOT
            / "results"
            / "main_exact_rerun"
            / "table6"
            / "test_result_FCP_4layer_test_m10n10_correct_1e_3_seed_sample.csv"
        ),
    )
    parser.add_argument(
        "--table2-results-root",
        type=Path,
        default=ROOT / "results" / "main_exact_rerun" / "table2",
    )
    parser.add_argument(
        "--table7-results-root",
        type=Path,
        default=ROOT / "results" / "main_exact_rerun" / "table7",
    )
    args = parser.parse_args()

    if args.experiment == "table2":
        table2_paths = {}
        for size in ("m10", "m20", "m30"):
            dataset = f"test_{size}n10_correct_1e_3"
            table2_paths[(size, "FCP")] = (
                args.table2_results_root
                / "fcp"
                / f"test_result_FCP_4layer_{dataset}_seed_sample.csv"
            )
            table2_paths[(size, "PCP")] = (
                args.table2_results_root
                / "pcp"
                / f"test_result_PCP_cp_4layer_{dataset}_seed_sample.csv"
            )
            table2_paths[(size, "FCPLS")] = (
                args.table2_results_root
                / "fcpls"
                / f"test_result_FCPLS_4layer_{dataset}_seed_sample.csv"
            )
            table2_paths[(size, "BSP")] = (
                args.table2_results_root
                / "bsp"
                / f"test_result_BSP_{dataset}.csv"
            )
        report = verify_table2(args.published, table2_paths)
    elif args.experiment == "table6":
        report = verify_table6(args.published, args.training_summary, args.seed_sample_results)
    else:
        table7_paths = {
            "log1p": args.table7_results_root / "log" / "test_result_FCP_4layer_test_m10n10_log_correct_1e_3.csv",
            "cube_root": args.table7_results_root / "cube_root" / "test_result_FCP_4layer_test_m10n10_f0.33_correct_1e_3.csv",
            "beta_5_5": args.table7_results_root / "beta" / "test_result_FCP_4layer_test_m10n10_beta_5_5_correct_1e_3.csv",
            "beta_half_half": args.table7_results_root / "beta" / "test_result_FCP_4layer_test_m10n10_beta_half_half_correct_1e_3.csv",
        }
        report = verify_table7(args.published, table7_paths)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
