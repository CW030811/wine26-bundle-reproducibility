#!/usr/bin/env python3
"""Resumable runner for Figures 6-8 and Tables 3-4.

Each learned-method subprocess evaluates one dataset with one seed. This keeps
the longest checkpoint unit bounded and makes the multi-day replay resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = "models/main_base_4layer_correct_lr_3"
SELF_IMPROVED_MODEL_DIR = "models/self_improved_m10n50_correct_lr_3"
SEEDS = tuple(range(1, 11))


class RunTask(NamedTuple):
    task_id: str
    method: str
    model_tag: str
    dataset: str
    expected_samples: int
    minimum_successes: int
    seed: int | None
    model_dir: str


FIGURE6_DATASETS = [
    ("test_BSP_m20n20_correct_1e_3", 100),
    ("test_BSP_m40n20_correct_1e_3", 100),
    ("test_BSP_m60n20_correct_1e_3", 50),
    ("test_BSP_m80n20_correct_1e_3", 30),
]
FIGURE7_DATASETS = [
    ("test_BSP_m10n20_correct_1e_3", 100),
    ("test_BSP_m10n30_correct_1e_3", 100),
    ("test_BSP_m10n40_correct_1e_3", 100),
    ("test_BSP_m10n50_correct_1e_3", 100),
]
TABLE3_DATASETS = [
    ("test_BSP_m10n20_correct_1e_3", 100),
    ("test_BSP_m10n30_correct_1e_3", 100),
    ("test_BSP_m10n40_correct_1e_3", 100),
    ("test_BSP_m20n20_correct_1e_3", 100),
    ("test_BSP_m20n30_correct_1e_3", 100),
    ("test_BSP_m20n40_correct_1e_3", 100),
]
TABLE4_DATASETS = [
    ("test_BSP_m10n60_correct_1e_3", 100),
    ("test_BSP_m10n80_correct_1e_3", 100),
    ("test_BSP_m10n100_correct_1e_3", 100),
    ("test_BSP_m20n60_correct_1e_3", 100),
    ("test_BSP_m20n80_correct_1e_3", 100),
    ("test_BSP_m20n100_correct_1e_3", 100),
]


def _learned_tasks(
    method: str,
    model_tag: str,
    model_dir: str,
    datasets: list[tuple[str, int]],
) -> list[RunTask]:
    tasks = []
    for dataset, expected_samples in datasets:
        minimum_successes = expected_samples
        if method == "FCP" and model_tag == "base" and dataset == "test_BSP_m80n20_correct_1e_3":
            minimum_successes = 29
        for seed in SEEDS:
            tasks.append(
                RunTask(
                    task_id=f"{model_tag}-{method.lower()}-{dataset}-seed{seed}",
                    method=method,
                    model_tag=model_tag,
                    dataset=dataset,
                    expected_samples=expected_samples,
                    minimum_successes=minimum_successes,
                    seed=seed,
                    model_dir=model_dir,
                )
            )
    return tasks


def _bsp_tasks(datasets: list[tuple[str, int]]) -> list[RunTask]:
    return [
        RunTask(
            task_id=f"bsp-{dataset}",
            method="BSP",
            model_tag="baseline",
            dataset=dataset,
            expected_samples=expected_samples,
            minimum_successes=expected_samples,
            seed=None,
            model_dir="",
        )
        for dataset, expected_samples in datasets
    ]


def _figure_tasks() -> list[RunTask]:
    tasks = []
    tasks += _learned_tasks("FCP", "base", BASE_MODEL_DIR, FIGURE6_DATASETS)
    tasks += _bsp_tasks(FIGURE6_DATASETS)
    tasks += _learned_tasks("FCP", "base", BASE_MODEL_DIR, FIGURE7_DATASETS)
    tasks += _learned_tasks("PCP", "base", BASE_MODEL_DIR, FIGURE7_DATASETS)
    tasks += _bsp_tasks(FIGURE7_DATASETS)
    tasks += _learned_tasks("FCP", "self_improved", SELF_IMPROVED_MODEL_DIR, FIGURE7_DATASETS)
    tasks += _learned_tasks("PCP", "self_improved", SELF_IMPROVED_MODEL_DIR, FIGURE7_DATASETS)
    return tasks


def _table_tasks() -> list[RunTask]:
    tasks = []
    tasks += _learned_tasks("FCP", "base", BASE_MODEL_DIR, TABLE3_DATASETS)
    table3_pcp = [item for item in TABLE3_DATASETS if item[0] != "test_BSP_m20n40_correct_1e_3"]
    tasks += _learned_tasks("PCP", "base", BASE_MODEL_DIR, table3_pcp)
    tasks += _learned_tasks("FCPLS", "base", BASE_MODEL_DIR, TABLE3_DATASETS)
    tasks += _learned_tasks("FCP", "self_improved", SELF_IMPROVED_MODEL_DIR, TABLE4_DATASETS)
    table4_pcp = [item for item in TABLE4_DATASETS if item[0].startswith("test_BSP_m10n")]
    tasks += _learned_tasks("PCP", "self_improved", SELF_IMPROVED_MODEL_DIR, table4_pcp)
    tasks += _learned_tasks("FCPLS", "self_improved", SELF_IMPROVED_MODEL_DIR, TABLE4_DATASETS)
    return tasks


def build_tasks(batch: str) -> list[RunTask]:
    if batch == "figures":
        return _figure_tasks()
    if batch == "tables":
        return _table_tasks()
    if batch != "all":
        raise ValueError(f"unknown batch: {batch}")
    deduplicated = {}
    for task in _figure_tasks() + _table_tasks():
        deduplicated.setdefault(task.task_id, task)
    return list(deduplicated.values())


def task_output_dir(output_root: Path, task: RunTask) -> Path:
    leaf = f"seed_{task.seed}" if task.seed is not None else "bsp"
    return output_root / "raw" / task.model_tag / task.method.lower() / task.dataset / leaf


def task_result_path(output_root: Path, task: RunTask) -> Path:
    result_dir = task_output_dir(output_root, task)
    if task.method == "FCP":
        name = f"test_result_FCP_4layer_{task.dataset}_seed_sample.csv"
    elif task.method == "PCP":
        name = f"test_result_PCP_cp_4layer_{task.dataset}_seed_sample.csv"
    elif task.method == "FCPLS":
        name = f"test_result_FCPLS_4layer_{task.dataset}_seed_sample.csv"
    else:
        name = f"test_result_BSP_{task.dataset}.csv"
    return result_dir / name


def build_command(root: Path, python: Path, output_root: Path, task: RunTask) -> list[str]:
    source = root / "src" / "deterministic"
    script_names = {
        "FCP": "test_FCP_multi_model_avg.py",
        "PCP": "test_PCP_cp_multi_model_avg.py",
        "FCPLS": "test_FCPLS_score_cached_lp.py",
        "BSP": "test_BSP.py",
    }
    command = [
        str(python),
        str(source / script_names[task.method]),
        "--data_dir",
        str(root),
        "--test_subdirs",
        f"data/deterministic/{task.dataset}",
        "--result_dir",
        str(task_output_dir(output_root, task)),
    ]
    if task.seed is not None:
        command += [
            "--model_dir",
            task.model_dir,
            "--layers",
            "4",
            "--seeds",
            str(task.seed),
        ]
    if task.method == "FCPLS":
        command.append("--no-plot")
    return command


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_task_result(output_root: Path, task: RunTask) -> tuple[bool, int, str]:
    result_path = task_result_path(output_root, task)
    if not result_path.exists():
        return False, 0, "result file missing"
    try:
        rows = read_csv_rows(result_path)
    except Exception as exc:
        return False, 0, f"CSV unreadable: {exc}"
    count = len(rows)
    if not task.minimum_successes <= count <= task.expected_samples:
        return (
            False,
            count,
            f"rows={count}, expected range={task.minimum_successes}..{task.expected_samples}",
        )
    if task.seed is not None:
        seeds = {int(row["seed"]) for row in rows}
        if seeds != {task.seed}:
            return False, count, f"unexpected seed values: {sorted(seeds)}"
    return True, count, "complete"


def verify_full_gurobi_license() -> None:
    import gurobipy as gp

    model = gp.Model("wine26_remaining_full_license_probe")
    variables = model.addVars(3000, vtype=gp.GRB.BINARY)
    model.addConstr(gp.quicksum(variables.values()) <= 1)
    model.setObjective(gp.quicksum(variables.values()), gp.GRB.MAXIMIZE)
    model.Params.OutputFlag = 0
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL or abs(model.ObjVal - 1.0) > 1e-9:
        raise RuntimeError(f"full-license probe failed: status={model.Status}")
    model.dispose()


def _write_json_atomic(destination: Path, payload: dict) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def combine_completed_outputs(tasks: list[RunTask], output_root: Path) -> list[dict]:
    grouped = defaultdict(list)
    for task in tasks:
        grouped[(task.model_tag, task.method, task.dataset)].append(task)

    combined_records = []
    for (model_tag, method, dataset), group in grouped.items():
        if method == "BSP":
            source = task_result_path(output_root, group[0])
            destination = output_root / "combined" / "baseline" / "bsp" / f"{dataset}.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            rows = read_csv_rows(destination)
        else:
            if {task.seed for task in group} != set(SEEDS):
                raise RuntimeError(f"incomplete seed group: {model_tag}/{method}/{dataset}")
            rows = []
            for task in sorted(group, key=lambda item: int(item.seed)):
                rows.extend(read_csv_rows(task_result_path(output_root, task)))
            destination = (
                output_root
                / "combined"
                / model_tag
                / method.lower()
                / f"{dataset}_seed_sample.csv"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        combined_records.append(
            {
                "model_tag": model_tag,
                "method": method,
                "dataset": dataset,
                "rows": len(rows),
                "path": str(destination),
            }
        )
    return combined_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=["figures", "tables", "all"], default="all")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gurobi-license", type=Path)
    parser.add_argument("--skip-license-probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else root / "results" / "remaining_exact_rerun"
    )
    tasks = build_tasks(args.batch)
    if args.dry_run:
        for task in tasks:
            print(task.task_id)
        print(f"TOTAL_TASKS={len(tasks)}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    license_path = args.gurobi_license
    if license_path is None and environment.get("GRB_LICENSE_FILE"):
        license_path = Path(environment["GRB_LICENSE_FILE"])
    if license_path is None:
        license_path = Path.home() / ".gurobi" / "gurobi.lic"
    if license_path.exists():
        environment["GRB_LICENSE_FILE"] = str(license_path.resolve())
        os.environ["GRB_LICENSE_FILE"] = environment["GRB_LICENSE_FILE"]
    environment["PYTHONUNBUFFERED"] = "1"
    if not args.skip_license_probe:
        verify_full_gurobi_license()

    manifest_path = output_root / f"{args.batch}_run_manifest.json"
    manifest = {
        "batch": args.batch,
        "runtime_comparison_required": False,
        "task_count": len(tasks),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
    }
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tasks"].update(previous.get("tasks", {}))
            manifest["original_started_utc"] = previous.get("original_started_utc", previous.get("started_utc"))
        except (json.JSONDecodeError, OSError):
            pass

    for index, task in enumerate(tasks, start=1):
        complete, row_count, reason = validate_task_result(output_root, task)
        if complete:
            manifest["tasks"][task.task_id] = {
                "status": "complete",
                "rows": row_count,
                "result": str(task_result_path(output_root, task)),
                "resume_action": "skipped_existing",
            }
            print(f"[{index}/{len(tasks)}] SKIP {task.task_id} rows={row_count}", flush=True)
            _write_json_atomic(manifest_path, manifest)
            continue

        result_dir = task_output_dir(output_root, task)
        result_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_root / "logs" / f"{task.task_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_command(root, args.python, output_root, task)
        started = datetime.now(timezone.utc).isoformat()
        print(f"[{index}/{len(tasks)}] RUN {task.task_id}; previous={reason}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        complete, row_count, reason = validate_task_result(output_root, task)
        manifest["tasks"][task.task_id] = {
            "status": "complete" if completed.returncode == 0 and complete else "failed",
            "rows": row_count,
            "validation": reason,
            "returncode": completed.returncode,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "log": str(log_path),
            "result": str(task_result_path(output_root, task)),
        }
        _write_json_atomic(manifest_path, manifest)
        if completed.returncode != 0 or not complete:
            print(
                f"FAILED {task.task_id}: returncode={completed.returncode}, validation={reason}",
                flush=True,
            )
            return 1

    combined = combine_completed_outputs(tasks, output_root)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["combined"] = combined
    manifest["status"] = "complete"
    _write_json_atomic(manifest_path, manifest)
    print(f"BATCH_COMPLETE batch={args.batch} tasks={len(tasks)} groups={len(combined)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
