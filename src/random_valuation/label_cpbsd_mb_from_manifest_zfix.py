import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

# ZFIX: import the z-fixed MB solver (lb=-inf on the per-(customer,bundle) profit var)
# so the training labels are generated under the correct mixed-bundling optimum.
from solve_mb_bsp_on_cpbsd_v2_zfix import json_default, load_instance, solve_mb


def read_manifest(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Batch label CPBSD instances with the MB solver using manifest files.")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=1e-2)
    parser.add_argument("--output-flag", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    result_rows = []
    started_at = time.time()
    for idx, row in enumerate(rows, 1):
        instance_path = Path(row["instance_path"])
        split = row.get("split", "unknown")
        stem = instance_path.stem
        result_path = out_dir / f"{stem}__mb.json"

        if result_path.exists() and not args.overwrite:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                result = {"error": f"Failed to read cached result: {exc}"}
        else:
            try:
                v_kn, c_n = load_instance(instance_path)
                result = solve_mb(
                    v_kn=v_kn,
                    c_n=c_n,
                    time_limit=args.time_limit,
                    mip_gap=args.mip_gap,
                    output_flag=args.output_flag,
                )
            except Exception as exc:
                result = {"error": str(exc)}
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=json_default),
                encoding="utf-8",
            )

        solver_status = result.get("solver_status", -1)
        runtime = result.get("runtime")
        wall_time = result.get("wall_time")
        mip_gap = result.get("mip_gap")
        objective = result.get("objective")
        chosen_product_matrix = result.get("chosen_product_matrix")
        x_positive_rate = None
        x_mean = None
        if chosen_product_matrix is not None:
            x_arr = np.asarray(chosen_product_matrix, dtype=float)
            x_positive_rate = float((x_arr > 0.5).mean())
            x_mean = float(x_arr.mean())

        result_rows.append(
            {
                "split": split,
                "index_in_split": row.get("index_in_split", idx),
                "filename": row.get("filename", instance_path.name),
                "instance_path": str(instance_path),
                "result_path": str(result_path),
                "solver_status": solver_status,
                "runtime": runtime,
                "wall_time": wall_time,
                "mip_gap": mip_gap,
                "objective": objective,
                "x_positive_rate": x_positive_rate,
                "x_mean": x_mean,
                "has_solution": bool(result.get("feasible")) and chosen_product_matrix is not None,
                "error": result.get("error", ""),
            }
        )

        if idx % 10 == 0 or idx == len(rows):
            elapsed = time.time() - started_at
            print(
                f"[{idx}/{len(rows)}] labeled {stem} | status={solver_status} | "
                f"feasible={result.get('feasible', False)} | elapsed={elapsed:.1f}s"
            )

    result_manifest_path = out_dir / f"{manifest_path.stem}__mb_results.csv"
    write_manifest(result_manifest_path, result_rows)
    summary = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "count": len(result_rows),
        "time_limit": args.time_limit,
        "mip_gap": args.mip_gap,
        "has_solution_count": sum(1 for r in result_rows if r["has_solution"]),
        "status_counts": {},
        "result_manifest": str(result_manifest_path),
    }
    for row in result_rows:
        key = str(row["solver_status"])
        summary["status_counts"][key] = summary["status_counts"].get(key, 0) + 1

    summary_path = out_dir / f"{manifest_path.stem}__mb_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
