#!/usr/bin/env python3
"""One interface for preparing, replaying, checking and plotting paper experiments."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from extract_bundle import extract_bundle
import run_remaining_exact as remaining
import run_table2_exact as table2
import verify_remaining_results as verify_remaining

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.mplconfig'))
os.environ.setdefault('MPLBACKEND', 'Agg')
NAMES = ['table2', 'table3', 'table4', 'table5', 'table6', 'table7',
         'figure6', 'figure7', 'figure8', 'figure9', 'figure10', 'figure11']
SEEDS = ','.join(map(str, range(1, 11)))
BASE = 'models/main_base_4layer_correct_lr_3'
IMPROVED = 'models/self_improved_m10n50_correct_lr_3'


def replay_commands(name, output, root=ROOT, python=sys.executable):
    root, output = Path(root).resolve(), Path(output).resolve()
    def script(relative, *args):
        return [str(python), str(root / relative), *map(str, args)]
    def fcp(relative, datasets, destination):
        return script('src/deterministic/' + relative, '--data_dir', root,
                      '--test_subdirs', datasets, '--model_dir', BASE,
                      '--layers', 4, '--seeds', SEEDS, '--result_dir', destination)
    if name == 'table2':
        commands = []
        for item in table2.build_commands(root, Path(python), ['FCP', 'PCP', 'FCPLS', 'BSP'], SEEDS):
            command = item['command']
            command[command.index('--result_dir') + 1] = str(output / item['method'].lower())
            commands.append(command)
        return commands
    if name in {'table3', 'table4', 'figure6', 'figure7', 'figure8'}:
        return [script('scripts/run_remaining_exact.py', '--experiment', name,
                       '--root', root, '--output-root', output)]
    if name == 'table5':
        return [script('src/random_valuation/run_experiment_zfix.py', '--model-path',
                       root / 'artifacts/random_valuation/models/best_model_edge_cpbsd_mb_x_2layer_seed1000.pt',
                       '--out-csv', output / 'experiment_zfix.csv')]
    if name == 'table6':
        return [fcp('test_FCP_multi_model_avg.py', 'data/deterministic/test_m10n10_correct_1e_3', output)]
    if name == 'table7':
        return [fcp('test_FCP_multi_model_avg.py',
                    'data/ood/test_m10n10_beta_5_5_correct_1e_3;data/ood/test_m10n10_beta_half_half_correct_1e_3', output / 'beta'),
                fcp('test_FCP_multi_model_avg_log.py', 'data/ood/test_m10n10_log_correct_1e_3', output / 'log'),
                fcp('test_FCP_multi_model_avg_f33.py', 'data/ood/test_m10n10_f0.33_correct_1e_3', output / 'cube_root')]
    if name in {'figure9', 'figure10'}:
        return [script('src/appendix/rerun_seed1.py', '--mode', 'cutoff' if name == 'figure9' else 'k',
                       '--output-root', output, '--resume')]
    if name == 'figure11':
        return [script('src/appendix_rerun/run_guarded_sensitivity.py', '--experiment', 'lp_milp',
                       '--output-dir', output, '--layer', 4, '--seed', 1,
                       '--lp-milp-samples', 10, '--max-iterations', 50, '--tolerance', '1e-6')]
    raise ValueError(name)


def reference_results(name):
    if name in {'table2', 'table6', 'table7'}:
        return ROOT / 'results/main_exact_rerun' / name
    if name in {'table3', 'table4', 'figure6', 'figure7', 'figure8'}:
        return ROOT / 'results/remaining_exact_rerun'
    if name == 'table5':
        return ROOT / 'artifacts/random_valuation/results'
    if name in {'figure9', 'figure10'}:
        return ROOT / 'results/appendix_cd_seed1'
    return ROOT / 'results/appendix_e_replay'


def verify(name, output):
    import verify_reference_results as reference
    published_path = ROOT / 'provenance/PUBLISHED_VALUES.json'
    if name in {'table3', 'table4', 'figure6', 'figure7', 'figure8'}:
        published = json.loads(published_path.read_text())
        return verify_remaining.verify_experiment(name, published, output)
    if name in {'table2', 'table6', 'table7'}:
        command = [sys.executable, str(ROOT / 'scripts/verify_main_results.py'), '--experiment', name]
        if name == 'table2':
            command += ['--table2-results-root', str(output)]
        elif name == 'table6':
            command += ['--seed-sample-results', str(output / 'test_result_FCP_4layer_test_m10n10_correct_1e_3_seed_sample.csv')]
        else:
            command += ['--table7-results-root', str(output)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if not result.stdout.strip():
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout)
    if name == 'table5':
        return reference.verify_table5(output / 'experiment_zfix.csv')
    if name in {'figure9', 'figure10'}:
        return reference.verify_sensitivity(name, output)
    return reference.verify_figure11(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'prepare', 'replay', 'verify', 'plot'])
    parser.add_argument('experiment', nargs='?', default='all', choices=NAMES + ['all'])
    parser.add_argument('--output-dir', type=Path, help='New replay directory for one experiment; defaults to results/reproduced/EXPERIMENT')
    parser.add_argument('--reference', action='store_true', help='Verify/plot the packaged results instead of a new replay')
    parser.add_argument('--check-only', action='store_true', help='Prepare: check archives without extraction')
    parser.add_argument('--dry-run', action='store_true', help='Print exact commands; do not extract or run')
    args = parser.parse_args()
    if args.output_dir and args.experiment == 'all':
        parser.error('--output-dir requires one experiment')
    if args.reference and args.action not in {'verify', 'plot'}:
        parser.error('--reference applies to verify or plot')
    if args.check_only and args.action != 'prepare':
        parser.error('--check-only applies only to prepare')
    registry = json.loads((ROOT / 'EXPERIMENTS.json').read_text())
    if args.action == 'list':
        for entry in sorted(registry['experiments'], key=lambda e: NAMES.index(e['key'])):
            print(f"{entry['key']:9} {entry['status']:35} {entry['title']}")
        return 0
    manifest = json.loads((ROOT / 'bundles/manifest.json').read_text())
    names = NAMES if args.experiment == 'all' else [args.experiment]
    reports = {}
    for name in names:
        output = (args.output_dir.resolve() if args.output_dir else ROOT / 'results/reproduced' / name)
        if any((ROOT / member).resolve().is_relative_to(output) or output == (ROOT / member).resolve()
               for entry in manifest['archives'].values() for member in entry['files']):
            parser.error('Output directory overlaps packaged reference files; choose a new results/reproduced directory')
        input_results = reference_results(name) if args.reference else output
        if args.dry_run:
            if args.action == 'replay':
                for command in replay_commands(name, output):
                    print(shlex.join(command))
            else:
                print(json.dumps({'action': args.action, 'experiment': name, 'result_directory': str(input_results),
                                  'archive': manifest['archives'][name]['file']}))
            continue
        dependencies = next((entry.get('training_dependencies', []) for entry in registry['experiments'] if entry['key'] == name), [])
        for bundle in [name, *dependencies]:
            extract_bundle(ROOT, manifest['archives'][bundle], args.check_only)
        if args.action == 'prepare':
            print(f'{name}: archive verified' + ('' if args.check_only else ' and extracted'))
            continue
        output.mkdir(parents=True, exist_ok=True)
        if args.action == 'replay':
            environment = os.environ.copy()
            environment['PYTHONUNBUFFERED'] = '1'
            # No automatic license installation or machine-specific fallback.
            for index, command in enumerate(replay_commands(name, output), 1):
                print(f'{name}: step {index}: {shlex.join(command)}', flush=True)
                subprocess.run(command, cwd=ROOT, env=environment, check=True)
        if args.action == 'plot':
            from plot_results import plot_experiment
            plot_experiment(name, input_results, output / 'plots')
        else:
            report = verify(name, input_results)
            report['experiment'] = name
            report['evidence'] = 'packaged replay' if args.reference else 'new replay'
            report['runtime_fields_checked'] = 0
            reports[name] = report
            (output / ('reference_verification.json' if args.reference else 'verification.json')).write_text(json.dumps(report, indent=2) + '\n')
            from plot_results import write_statistics
            write_statistics(name, input_results, output / ('reference_statistics.csv' if args.reference else 'statistics.csv'), report)
            print(f"{name}: {'PASS' if report['passed'] else 'FAIL'}", flush=True)
    return 0 if all(report['passed'] for report in reports.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
