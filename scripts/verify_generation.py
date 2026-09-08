#!/usr/bin/env python3
"""Rebuild archived random draws without solving MILPs or retraining models."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import msgpack
import msgpack_numpy as mnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return msgpack.unpackb(path.read_bytes(), object_hook=mnp.decode, strict_map_key=False)


def equal(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, np.asarray(right))
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def audit_random(root):
    spec = importlib.util.spec_from_file_location('cpbsd_generation_audit', root / 'src/random_valuation/generate_data_CPBSD.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    files = sorted((root / 'data/random_valuation_n5').glob('*.msgpack'))
    mismatches, seeds = [], []
    for path in files:
        archived = load(path)
        setup = module.CPBSDSetup(**archived['setup'])
        fresh = module.generate_cpbsd_instance(setup)
        seeds.append(setup.seed)
        if not equal(archived, fresh):
            mismatches.append(path.name)
    return {'instances': len(files), 'unique_seeds': len(set(seeds)), 'mismatches': mismatches,
            'passed': len(files) == 4000 and set(seeds) == set(range(1000, 5000)) and not mismatches}


def audit_self_improved(root):
    directory = root / 'data/deterministic/train_PCP_m10n50_correct_4layer_seed9_lr3'
    np.random.seed(9)
    mismatches = []
    for index in range(1, 3001):
        n = np.random.randint(50, 51)
        unit_cs = np.random.rand(1, n)
        ship_cs = np.random.rand(10, 1)
        unit_us = np.random.rand(10, n)
        xs = np.random.rand(10, 1)
        fresh = {'unit_cs': unit_cs, 'ship_cs': ship_cs, 'unit_us': unit_us, 'Ns': xs / np.sum(xs)}
        filename = f'sample_data_{index}_size_50_pcp.msgpack'
        archived = load(directory / filename)
        fields = [key for key in fresh if not np.array_equal(archived[key], fresh[key])]
        if fields:
            mismatches.append({'file': filename, 'fields': fields})
    return {'instances': 3000, 'seed': 9, 'checked_fields': ['unit_cs', 'ship_cs', 'unit_us', 'Ns'],
            'mismatches': mismatches, 'passed': not mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = {'scope': 'Decoded input values only; no label optimization or model retraining',
              'random': audit_random(args.root), 'self_improved': audit_self_improved(args.root)}
    report['passed'] = report['random']['passed'] and report['self_improved']['passed']
    rendered = json.dumps(report, indent=2) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    print(rendered)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
