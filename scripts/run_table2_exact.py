#!/usr/bin/env python3
"""Run all four Table 2 methods sequentially and verify non-runtime statistics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ";".join(
    [
        "data/deterministic/test_m10n10_correct_1e_3",
        "data/deterministic/test_m20n10_correct_1e_3",
        "data/deterministic/test_m30n10_correct_1e_3",
    ]
)
FINAL_MODEL_DIR = "models/main_base_4layer_correct_lr_3"


def build_commands(root: Path, python: Path, methods: list[str], seeds: str) -> list[dict]:
    """Build the exact, final-code commands without executing them."""
    root = Path(root).resolve()
    python = str(python)
    source = root / "src" / "deterministic"
    result_root = root / "results" / "main_exact_rerun" / "table2"
    shared_model_args = [
        "--data_dir",
        str(root),
        "--test_subdirs",
        DATASETS,
        "--model_dir",
        FINAL_MODEL_DIR,
        "--layers",
        "4",
        "--seeds",
        seeds,
    ]
    definitions = {
        "FCP": [
            python,
            str(source / "test_FCP_multi_model_avg.py"),
            *shared_model_args,
            "--result_dir",
            str(result_root / "fcp"),
        ],
        "PCP": [
            python,
            str(source / "test_PCP_cp_multi_model_avg.py"),
            *shared_model_args,
            "--result_dir",
            str(result_root / "pcp"),
        ],
        "FCPLS": [
            python,
            str(source / "test_FCPLS_score_cached_lp.py"),
            *shared_model_args,
            "--result_dir",
            str(result_root / "fcpls"),
            "--no-plot",
        ],
        "BSP": [
            python,
            str(source / "test_BSP.py"),
            "--data_dir",
            str(root),
            "--test_subdirs",
            DATASETS,
            "--result_dir",
            str(result_root / "bsp"),
        ],
    }
    return [{"method": method, "command": definitions[method]} for method in methods]


def verify_full_gurobi_license() -> None:
    """Fail fast if the active Gurobi environment is size-limited."""
    import gurobipy as gp

    model = gp.Model("wine26_full_license_probe")
    variables = model.addVars(3000, vtype=gp.GRB.BINARY)
    model.addConstr(gp.quicksum(variables.values()) <= 1)
    model.setObjective(gp.quicksum(variables.values()), gp.GRB.MAXIMIZE)
    model.Params.OutputFlag = 0
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL or abs(model.ObjVal - 1.0) > 1e-9:
        raise RuntimeError(
            f"Gurobi full-license probe did not solve correctly: status={model.Status}"
        )
    model.dispose()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--methods",
        default="FCP,PCP,FCPLS,BSP",
        help="Comma-separated subset, in execution order.",
    )
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--gurobi-license", type=Path)
    parser.add_argument("--skip-license-probe", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    methods = [item.strip().upper() for item in args.methods.split(",") if item.strip()]
    unknown = [method for method in methods if method not in {"FCP", "PCP", "FCPLS", "BSP"}]
    if unknown:
        parser.error(f"unknown methods: {unknown}")

    environment = os.environ.copy()
    license_path = args.gurobi_license
    if license_path is None and environment.get("GRB_LICENSE_FILE"):
        license_path = Path(environment["GRB_LICENSE_FILE"])
    if license_path is None:
        license_path = Path.home() / ".gurobi" / "gurobi.lic"
    if license_path.exists():
        environment["GRB_LICENSE_FILE"] = str(license_path.resolve())
        os.environ["GRB_LICENSE_FILE"] = environment["GRB_LICENSE_FILE"]

    if not args.skip_license_probe:
        verify_full_gurobi_license()

    result_root = root / "results" / "main_exact_rerun" / "table2"
    result_root.mkdir(parents=True, exist_ok=True)
    commands = build_commands(root, args.python, methods, args.seeds)
    run_records = []
    for item in commands:
        method = item["method"]
        log_path = result_root / f"{method.lower()}_rerun.log"
        started = datetime.now(timezone.utc).isoformat()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                item["command"],
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        run_records.append(
            {
                "method": method,
                "command": item["command"],
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "returncode": completed.returncode,
                "log": str(log_path.relative_to(root)),
            }
        )
        if completed.returncode != 0:
            break

    manifest = {
        "experiment": "table2",
        "runtime_comparison_required": False,
        "methods_requested": methods,
        "runs": run_records,
    }
    manifest_path = result_root / "rerun_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if len(run_records) != len(commands) or any(item["returncode"] for item in run_records):
        return 1
    if set(methods) == {"FCP", "PCP", "FCPLS", "BSP"}:
        verification = subprocess.run(
            [
                str(args.python),
                str(root / "scripts" / "verify_main_results.py"),
                "--experiment",
                "table2",
                "--table2-results-root",
                str(result_root),
            ],
            cwd=root,
            env=environment,
            check=False,
        )
        return verification.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
