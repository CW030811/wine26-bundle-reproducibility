"""Resumable, parallel MB label regeneration with the z-fixed solver.

Re-solves every CPBSD N=5 random_ind instance referenced by the tracked
train/eval/test manifests, writing chosen_product_matrix labels under
labels/{split}/. Skips instances already labelled (idempotent / resumable).
After labelling, writes fixed manifests labels/manifest_{split}__mb_zfix.csv
that point each instance at its new label, for the training step.

Self-contained (no /tmp dependency) so it survives across runs.
"""
import sys, csv, json, time
from pathlib import Path
from multiprocessing import Pool

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
REPRO = CODE.parent
PACKAGE_ROOT = CODE.parents[1]
LABEL_ROOT = PACKAGE_ROOT / "artifacts" / "random_valuation" / "labels"
MANIFESTS = {
    "train": LABEL_ROOT / "manifest_train__portable.csv",
    "eval":  LABEL_ROOT / "manifest_eval__portable.csv",
    "test":  LABEL_ROOT / "manifest_test__portable.csv",
}


def _label_path(split, instance_path):
    return LABEL_ROOT / split / f"{Path(instance_path).stem}__mb.json"


def collect():
    work = []
    for split, man in MANIFESTS.items():
        (LABEL_ROOT / split).mkdir(parents=True, exist_ok=True)
        for r in csv.DictReader(open(man)):
            if r.get("has_solution", "").lower() not in {"true", "1"}:
                continue
            ip = r["instance_path"]
            work.append((split, ip, str(_label_path(split, ip))))
    return work


def solve_one(args):
    split, ip, op = args
    if Path(op).exists():
        return "cached"
    from solve_mb_bsp_on_cpbsd_v2_zfix import solve_mb, load_instance, json_default
    try:
        v, c = load_instance(Path(ip))
        res = solve_mb(v_kn=v, c_n=c, time_limit=300.0, mip_gap=0.01, output_flag=0)
        Path(op).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        return "solved"
    except Exception as exc:  # pragma: no cover
        return f"err:{exc}"


def write_fixed_manifests():
    for split, man in MANIFESTS.items():
        rows = []
        for r in csv.DictReader(open(man)):
            if r.get("has_solution", "").lower() not in {"true", "1"}:
                continue
            ip = r["instance_path"]
            op = _label_path(split, ip)
            if op.exists():
                rows.append({"instance_path": ip, "result_path": str(op), "has_solution": "true"})
        mp = LABEL_ROOT / f"manifest_{split}__mb_zfix.csv"
        with open(mp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["instance_path", "result_path", "has_solution"])
            w.writeheader(); w.writerows(rows)
        print(f"manifest {split}: {len(rows)} rows -> {mp}", flush=True)


if __name__ == "__main__":
    work = collect()
    todo = [w for w in work if not Path(w[2]).exists()]
    print(f"total={len(work)} already_done={len(work) - len(todo)} todo={len(todo)}", flush=True)
    t0 = time.time(); done = 0; errs = 0
    if todo:
        with Pool(6, maxtasksperchild=40) as p:
            for status in p.imap_unordered(solve_one, todo, chunksize=4):
                done += 1
                if status.startswith("err:"):
                    errs += 1
                if done % 200 == 0:
                    print(f"{done}/{len(todo)} elapsed={time.time()-t0:.0f}s errs={errs}", flush=True)
    write_fixed_manifests()
    print(f"ALL_DONE solved={done} errs={errs} elapsed={time.time()-t0:.0f}s", flush=True)
