"""Common statistics CSV and figures computed from actual released/replayed results."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['experiment', 'case', 'method', 'metric', 'value', 'reference', 'matched']


def statistics_rows(name, result_dir, report=None):
    if report is None:
        from reproduce import verify
        report = verify(name, result_dir)
    output = []
    def add(case, method, metric, value, target='', matched=''):
        output.append(dict(zip(FIELDS, [name, case, method, metric, value, target, matched])))
    for detail in report.get('details', []):
        case = str(detail.get('dataset', detail.get('size', detail.get('variant', detail.get('scale', f"seed{detail.get('seed', '')}")))))
        method = detail.get('paper_method', detail.get('method', 'FCP'))
        if detail.get('model_tag') == 'self_improved' and not method.endswith('_I'):
            method += '_I'
        matched = detail.get('matched', False)
        if 'statistics' in detail:
            stats = detail['statistics']
            if stats:
                add(case, method, 'profit_mean', stats['mean'], detail['expected_mean'], matched)
                add(case, method, 'profit_population_std', stats['population_std'], detail['expected_std'], matched)
                add(case, method, 'profit_sample_std', stats['sample_std'], detail['expected_std'], matched)
        elif 'field' in detail:
            add(case + '/' + detail['cost'], method, detail['field'], detail['raw'], detail['paper'], matched)
        elif name == 'table6':
            for metric, value, target in zip(['train_loss', 'validation_loss', 'profit_mean', 'profit_population_std'],
                                              detail['observed_non_runtime'], detail['published_non_runtime']):
                add(case, method, metric, value, target, matched)
        elif 'raw_mean' in detail:
            target = detail.get('expected_mean', detail.get('published_non_runtime', [''])[0])
            add(case, method, 'profit_mean', detail['raw_mean'], target, matched)
            for key, metric in [('raw_population_std', 'profit_population_std'), ('raw_sample_std', 'profit_sample_std')]:
                if key in detail:
                    add(case, method, metric, detail[key], detail['published_non_runtime'][1], matched)
    if name in {'figure9', 'figure10'}:
        import pandas as pd
        cutoff = name == 'figure9'
        filename = 'cutoff/cutoff_sensitivity_FCP_PCP_combined.csv' if cutoff else 'k_justify/justify_K_all_results_long.csv'
        data = pd.read_csv(result_dir / filename)
        keys = ['dataset', 'strategy'] + (['cutoff'] if cutoff else [])
        numeric = data.select_dtypes('number').columns.difference(keys)
        for key, group in data.groupby(keys):
            case = str(key[0]) + ('/cutoff=' + str(key[2]) if cutoff else '')
            for field in numeric:
                add(case, key[1], str(field) + '_mean', float(group[field].mean()))
    if name == 'figure11':
        add('all', 'LP_to_MILP', 'successful_moves', report['successful'], 489, report['passed'])
        add('all', 'LP_to_MILP', 'accepted_moves', report['accepted'], 522, report['passed'])
    return output


def write_statistics(name, result_dir, destination, report=None):
    output = statistics_rows(name, result_dir, report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    return output


def plot_experiment(name, result_dir, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    output.mkdir(parents=True, exist_ok=True)
    stats = write_statistics(name, result_dir, output / 'statistics.csv')
    if name in {'figure9', 'figure10'}:
        spec = importlib.util.spec_from_file_location('appendix_plot', ROOT / 'src/appendix/rerun_seed1.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if name == 'figure9':
            module.plot_cutoff(pd.read_csv(result_dir / 'cutoff/cutoff_sensitivity_FCP_PCP_combined.csv'), output)
        else:
            data = pd.read_csv(result_dir / 'k_justify/justify_K_all_results_long.csv')
            module.plot_k_justify(data, output)
            module.plot_k_balance_bars(data, output)
        return
    if name == 'figure11':
        records = json.loads((result_dir / 'lp_milp/LP_MILP_verification_results.json').read_text())
        groups = sorted({r['dataset_name'] for r in records})
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, dataset in zip(axes.flat, groups):
            group = [r for r in records if r['dataset_name'] == dataset]
            length = max(max(len(r['lp_path']), len(r['milp_path'])) for r in group)
            for field, label in [('lp_path', 'LP'), ('milp_path', 'MILP')]:
                paths = np.array([r[field] + [r[field][-1]] * (length - len(r[field])) for r in group])
                mean, std = paths.mean(axis=0), paths.std(axis=0)
                ax.plot(mean, label=label)
                ax.fill_between(range(length), mean - std, mean + std, alpha=.15)
            ax.set(title=dataset, xlabel='Improvement step', ylabel='Profit ratio')
            ax.legend()
        fig.tight_layout()
    elif name in {'figure6', 'figure7', 'figure8'}:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        coordinate = 'm' if name == 'figure6' else 'n'
        methods = sorted({r['method'] for r in stats})
        for method in methods:
            means = [r for r in stats if r['method'] == method and r['metric'] == 'profit_mean']
            means.sort(key=lambda r: int(re.search(coordinate + r'(\d+)', r['case'])[1]))
            xs = [int(re.search(coordinate + r'(\d+)', r['case'])[1]) for r in means]
            ys = np.array([r['value'] for r in means])
            stds = np.array([next((r['value'] for r in stats if r['case'] == mean['case'] and r['method'] == method
                                  and r['metric'] == 'profit_population_std'), 0) for mean in means])
            axes[0].plot(xs, ys, 'o-', label=method)
            axes[0].fill_between(xs, ys - stds, ys + stds, alpha=.12)
            timings = []
            for row in means:
                if method == 'BSP':
                    timings.append(1.)
                else:
                    tag = 'self_improved' if method.endswith('_I') else 'base'
                    source = result_dir / 'combined' / tag / method.removesuffix('_I').lower() / (row['case'] + '_seed_sample.csv')
                    data = pd.read_csv(source)
                    samples = data.groupby('sample_file')['runtime_ratio'].mean()
                    if name == 'figure6' and 'm80n20' in row['case']:
                        selection = json.loads((ROOT / 'provenance/FIGURE6_SAMPLE_SELECTION.json').read_text())
                        samples = samples.loc[selection['selected_samples']]
                    timings.append(samples.mean())
            axes[1].plot(xs, timings, 'o-', label=method)
        for ax, ylabel in zip(axes, ['Profit ratio', 'Runtime ratio (machine dependent)']):
            ax.set(xlabel=coordinate, ylabel=ylabel)
            ax.legend()
            ax.grid(alpha=.2)
        fig.tight_layout()
    else:
        # Tables are exported as one common long CSV, plus a readable graphic.
        fig, ax = plt.subplots(figsize=(11, max(3, len(stats) * .16)))
        ax.axis('off')
        cells = [[r['case'], r['method'], r['metric'], f"{r['value']:.6g}"] for r in stats]
        ax.table(cellText=cells, colLabels=['Case', 'Method', 'Metric', 'Value'], loc='center')
        fig.tight_layout()
    fig.savefig(output / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
