#!/usr/bin/env python3
"""Generate labeled data or train models with explicit, portable input/output paths."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.mplconfig'))
os.environ.setdefault('MPLBACKEND', 'Agg')
RECIPES = ['base', 'bsp', 'self_improved', 'log', 'cube_root', 'beta_5_5', 'beta_half_half', 'random']


def load_generator(relative):
    filename = ROOT / 'src' / relative
    sys.path.insert(0, str(filename.parent))
    spec = importlib.util.spec_from_file_location('wine_generator', filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def train_command(recipe, input_dir, output, python=sys.executable):
    if recipe == 'random':
        command = [python, str(ROOT / 'src/random_valuation/Training_multi_layer_cpbsd_mb_x.py'),
                   '--device', 'cpu', '--no_cleanup', '--model_dir', str(output / 'models'),
                   '--charts_dir', str(output / 'charts'), '--log_dir', str(output / 'logs')]
        if input_dir:
            for split in ['train', 'eval', 'test']:
                command += [f'--{split}_manifest', str(input_dir / f'manifest_{split}.csv')]
        return command
    if recipe not in {'base', 'self_improved'}:
        raise ValueError('Training recipes are base, self_improved, and random')
    data = input_dir or ROOT / 'data/deterministic' / (
        'train_m10n10_correct_1e_3' if recipe == 'base' else 'train_PCP_m10n50_correct_4layer_seed9_lr3')
    return [python, str(ROOT / 'src/deterministic/Training_multi_layer.py'),
            '--data_dir', str(ROOT), '--train_subdir', str(data), '--no_cleanup',
            '--model_subdir', str(output / 'models'), '--charts_subdir', str(output / 'charts'),
            '--log_subdir', str(output / 'logs')]


def generate(args, output):
    if output.exists():
        raise ValueError('Generation requires a new output directory: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.recipe == 'random':
        module = load_generator('random_valuation/generate_data_CPBSD.py')
        paths = module.generate_batch(str(output), args.products, 50, 'normal', 0.0, 'full',
                                      'random_ind', args.count, args.seed)
        limits = [0, int(args.count * .75), int(args.count * .90), args.count]
        for index, split in enumerate(['train', 'eval', 'test']):
            with (output / f'manifest_{split}.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['split', 'instance_path'])
                writer.writeheader()
                for path in paths[limits[index]:limits[index+1]]:
                    writer.writerow({'split': split, 'instance_path': Path(path).relative_to(ROOT).as_posix()})
        return
    filenames = {'base': 'generate_data_bundle.py', 'bsp': 'generate_data_BSP.py',
                 'self_improved': 'generate_data_PCP_cp.py', 'log': 'generate_data_bundle_log.py',
                 'cube_root': 'generate_data_bundle_f0.33.py', 'beta_5_5': 'generate_data_bundle_beta.py',
                 'beta_half_half': 'generate_data_bundle_beta.py'}
    module = load_generator('deterministic/' + filenames[args.recipe])
    parameters = [args.segments, args.products, args.products + 1, args.count, str(output) + os.sep]
    if args.recipe == 'self_improved':
        model = args.model or ROOT / 'models/main_base_4layer_correct_lr_3/best_model_edge_4layer_seed9.pt'
        if not model.is_file():
            raise FileNotFoundError(model)
        parameters += [str(model), torch.device('cpu'), 0.5]
    elif args.recipe.startswith('beta_'):
        shape = 5 if args.recipe == 'beta_5_5' else 0.5
        parameters += [shape, shape]
    module.generate_sample(*parameters)


def label_random(input_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'eval', 'test']:
        manifest = input_dir / f'manifest_{split}.csv'
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        subprocess.run([sys.executable, str(ROOT / 'src/random_valuation/label_cpbsd_mb_from_manifest_zfix.py'),
                        '--manifest', str(manifest), '--out-dir', str(output / split)], cwd=ROOT, check=True)
        with (output / split / f'manifest_{split}__mb_results.csv').open(newline='') as handle:
            records = list(csv.DictReader(handle))
        if not records or any(row['has_solution'].lower() != 'true' for row in records):
            raise RuntimeError(f'{split}: labels incomplete; inspect solver summary')
        for record in records:
            for key in ['instance_path', 'result_path']:
                path = Path(record[key])
                if path.is_absolute() and path.is_relative_to(ROOT):
                    record[key] = path.relative_to(ROOT).as_posix()
        with (output / f'manifest_{split}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['generate', 'label', 'train'])
    parser.add_argument('recipe', choices=RECIPES)
    parser.add_argument('--input-dir', type=Path, help='Training dataset directory, or directory with random training manifests')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='PCP labeling checkpoint; default is base seed 9')
    parser.add_argument('--manifest', type=Path, help='Random-valuation input manifest with instance_path column')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--count', type=int)
    parser.add_argument('--products', type=int)
    parser.add_argument('--segments', type=int, default=10)
    parser.add_argument('--epochs', type=int, help='Optional training-length override; paper recipe defaults to 200')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.seed = args.seed if args.seed is not None else 9 if args.recipe == 'self_improved' else 1000 if args.recipe == 'random' else 2026
    args.count = args.count if args.count is not None else 3000 if args.recipe in {'base', 'self_improved'} else 4000 if args.recipe == 'random' else 100
    args.products = args.products if args.products is not None else 50 if args.recipe == 'self_improved' else 5 if args.recipe == 'random' else 10
    if min(args.count, args.products, args.segments) < 1:
        parser.error('count, products and segments must be positive')
    output = args.output_dir.resolve()
    if output == ROOT or not output.is_relative_to(ROOT / 'results'):
        parser.error('Use a new output directory below the repository results/ directory')
    if args.action == 'label':
        if args.recipe != 'random' or not (args.manifest or args.input_dir):
            parser.error('Separate label stage requires random --input-dir or --manifest; deterministic generation includes optimization labels')
        if args.input_dir:
            if args.dry_run:
                print(json.dumps({'action': 'label', 'input': str(args.input_dir), 'output': str(output), 'splits': ['train', 'eval', 'test']}))
            else:
                label_random(args.input_dir.resolve(), output)
            return 0
        command = [sys.executable, str(ROOT / 'src/random_valuation/label_cpbsd_mb_from_manifest_zfix.py'),
                   '--manifest', str(args.manifest.resolve()), '--out-dir', str(output)]
    elif args.action == 'train':
        command = train_command(args.recipe, args.input_dir.resolve() if args.input_dir else None, output)
        if args.epochs is not None:
            command += ['--epochs', str(args.epochs)]
    else:
        command = None
    if args.dry_run:
        print(shlex.join(command) if command else json.dumps({'recipe': args.recipe, 'seed': args.seed,
              'samples': args.count, 'products': args.products, 'segments': args.segments,
              'output': str(output), 'includes_labels': args.recipe != 'random'}))
        return 0
    if command:
        if args.action == 'train' and output.exists() and any(output.iterdir()):
            parser.error('Training requires a new or empty output directory')
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        generate(args, output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
