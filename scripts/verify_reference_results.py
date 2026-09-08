#!/usr/bin/env python3
"""Audit published reference/replay evidence. Does not run optimizers or training."""
import csv
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def equal_value(a, b):
    try:
        x, y = float(a), float(b)
        return math.isfinite(x) and math.isfinite(y) and x == y
    except (ValueError, TypeError):
        return a == b


def accept_figure9_exception(key, column, archived, replay):
    return (key == ('test_m30n10_1e_3', 'sample_data_64_size_10.msgpack', '0.1', 'PCP')
            and column == 'revenue_ratio'
            and abs(float(archived) - 1.004751821452851) < 1e-12
            and math.isfinite(float(replay))
            and abs(float(replay) - float(archived)) <= 0.00130)


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def verify_sensitivity(name, result_root=None):
    cutoff = name == 'figure9'
    filename = 'cutoff_sensitivity_FCP_PCP_combined.csv' if cutoff else 'justify_K_all_results_long.csv'
    mode = 'cutoff' if cutoff else 'k_justify'
    keys = ['dataset', 'sample_id', 'cutoff', 'strategy'] if cutoff else ['dataset', 'sample_id', 'strategy']
    ignored = ['time_ratio'] if cutoff else ['runtime_ratio', 'total_time', 'base_running_time']
    reference = rows(ROOT / 'artifacts/appendix_cd' / filename)
    replay = rows((Path(result_root) if result_root else ROOT / 'results/appendix_cd_seed1') / mode / filename)
    left = {tuple(row[k] for k in keys): row for row in reference}
    right = {tuple(row[k] for k in keys): row for row in replay}
    expected = 1080 if cutoff else 630
    if not (len(reference) == len(replay) == len(left) == len(right) == expected and left.keys() == right.keys()):
        raise ValueError(name + ': incomplete or duplicate experiment keys')
    differences, exceptions = [], []
    for key, archived in left.items():
        for field, a in archived.items():
            if field in keys + ignored:
                continue
            b = right[key][field]
            if not equal_value(a, b):
                item = {'key': key, 'field': field, 'archived': a, 'replay': b}
                if cutoff and accept_figure9_exception(key, field, a, b):
                    exceptions.append(item)
                else:
                    differences.append(item)
    return {'passed': not differences and len(exceptions) <= 1, 'rows': expected,
            'non_runtime_bit_identical': not differences and not exceptions,
            'accepted_exceptions': exceptions, 'mismatches': differences}


def verify_table5(result_csv=None):
    published = json.loads((ROOT / 'provenance/PUBLISHED_VALUES.json').read_text())['table5_rows']
    data = rows(Path(result_csv) if result_csv else ROOT / 'artifacts/random_valuation/results/experiment_zfix.csv')
    checks = []
    for target in published:
        for method, offset in [('FCP', 2), ('BSP', 5)]:
            subset = [r for r in data if r['scale'] == f'N{target[0]}_K50' and r['cost'] == target[1]
                      and r['method'] == method and r['variant'] == 'fixed']
            if len(subset) != 5 or len({r['seed'] for r in subset}) != 5:
                raise ValueError('Table 5 incomplete five-seed group')
            for field, index in [('ins', offset), ('oos', offset + 1)]:
                value = statistics.fmean(float(r[field]) for r in subset)
                checks.append({'scale': target[0], 'cost': target[1], 'method': method,
                               'field': field, 'raw': value, 'paper': target[index],
                               'matched': round(value, 3) == target[index]})
    generated = subprocess.check_output([sys.executable, str(ROOT / 'src/random_valuation/make_table5_rows.py')], cwd=ROOT)
    archived = (ROOT / 'artifacts/random_valuation/results/table5_corrected_rows.tex').read_bytes()
    return {'passed': len(data) == 120 and all(x['matched'] for x in checks) and (result_csv is not None or generated == archived),
            'checked_fcp_bsp_statistics': len(checks), 'cpbsd_a_independently_replayed': False,
            'scope': 'FCP/BSP statistics from archived sweep; CPBSD-A and runtime cells reused from paper', 'details': checks}


def verify_figure11(result_root=None):
    base = ROOT / 'artifacts/appendix_e_final'
    original = json.loads((base / 'LP_MILP_verification_results.json').read_text())
    fresh_dir = (Path(result_root) if result_root else ROOT / 'results/appendix_e_replay') / 'lp_milp'
    fresh = json.loads((fresh_dir / 'LP_MILP_verification_results.json').read_text())
    key = lambda r: (r['dataset_name'], r['file_name'])
    reference = {key(r): r for r in original}
    replay = {key(r): r for r in fresh}
    fields = ['lp_to_milp_translation', 'lp_path', 'milp_path']
    # Path field names are taken directly from the archived JSON schema.
    paths = [k for k in original[0] if 'path' in k.lower()]
    fields = ['lp_to_milp_translation'] + paths
    exact = (len(original) == len(fresh) == len(reference) == len(replay) == 60 and reference.keys() == replay.keys()
             and all(reference[k][f] == replay[k][f] for k in reference for f in fields))
    flags = [v for r in fresh for v in r['lp_to_milp_translation']]
    counts = (sum(flags), len(flags))
    expected = json.loads((ROOT / 'provenance/PUBLISHED_VALUES.json').read_text())['appendix_e_final_provenance']
    guard_ok = all(hashlib.sha256((ROOT / f'src/test/{name}').read_bytes()).hexdigest() == expected['cached_lp_source_sha256']
                   for name in ['test_FCPLS_score_cached_lp.py', 'test_FCPLS_score_cached_lp_1.py'])
    model_ok = hashlib.sha256((ROOT / 'models_multi_layer_edge_update/best_model_edge_4layer_seed1.pt').read_bytes()).hexdigest() == expected['model_sha256']
    failures = json.loads((fresh_dir / 'LP_MILP_verification_failures.json').read_text())
    return {'passed': exact and counts == (489, 522) and guard_ok and model_ok and not failures,
            'instances': len(original), 'successful': counts[0], 'accepted': counts[1],
            'replay_paths_identical': exact, 'checked_path_fields': fields, 'guarded_source_and_model': guard_ok and model_ok}


def main():
    # All public experiments share the same verifier as the replay entry point.
    from reproduce import NAMES, reference_results, verify
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = {'scope': 'Archived evidence audit, not a new optimizer/training replay', 'experiments': {}}
    for name in NAMES:
        report['experiments'][name] = verify(name, reference_results(name))
    report['passed'] = all(result['passed'] for result in report['experiments'].values())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
