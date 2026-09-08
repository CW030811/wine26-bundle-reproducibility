import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import reproduce
import plot_results
import verify_main_results


class UnifiedInterfaceTests(unittest.TestCase):
    def test_every_experiment_has_its_own_plan_and_data_bundle(self):
        manifest = json.loads((ROOT / 'bundles/manifest.json').read_text())
        for name in reproduce.NAMES:
            self.assertIn(name, manifest['archives'])
            output = ROOT / 'results/test_plan' / name
            commands = reproduce.replay_commands(name, output)
            self.assertTrue(commands)
            for command in commands:
                self.assertTrue(Path(command[1]).is_file())
                self.assertTrue(any(str(output) in argument for argument in command))

    def test_figure8_plan_covers_all_five_curves(self):
        tasks = reproduce.remaining.build_experiment_tasks('figure8')
        self.assertEqual(len(tasks), 164)
        self.assertEqual({(t.model_tag, t.method) for t in tasks},
                         {('base', 'FCP'), ('base', 'PCP'), ('self_improved', 'FCP'),
                          ('self_improved', 'PCP'), ('baseline', 'BSP')})

    def test_invalid_check_only_cannot_start_a_replay(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce.py'),
                                 'replay', 'table2', '--check-only'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'applies only to prepare', result.stderr)

    def test_replay_cannot_overwrite_packaged_references(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce.py'),
                                 'replay', 'table2', '--output-dir', str(ROOT / 'results/main_exact_rerun/table2'),
                                 '--dry-run'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'overlaps packaged reference files', result.stderr)

    def test_table5_integer_scale_exports_common_statistics(self):
        rows = plot_results.statistics_rows('table5', ROOT, {'details': [
            {'scale': 10, 'cost': 'zero', 'method': 'FCP', 'field': 'ins', 'raw': 51.911,
             'paper': 51.911, 'matched': True}]})
        self.assertEqual(rows[0]['case'], '10/zero')
        self.assertEqual(rows[0]['metric'], 'ins')

    def test_fixed_figure6_subset_has_stable_identity(self):
        selection = json.loads((ROOT / 'provenance/FIGURE6_SAMPLE_SELECTION.json').read_text())
        self.assertEqual(len(set(selection['selected_samples'])), 29)
        self.assertEqual(selection['excluded_samples'], ['sample_data_3_size_20_sizepricing.msgpack'])
        self.assertFalse(set(selection['selected_samples']) & set(selection['excluded_samples']))

    def test_table6_duplicate_rows_are_not_accepted(self):
        source = ROOT / 'results/main_exact_rerun/table6/test_result_FCP_4layer_test_m10n10_correct_1e_3_seed_sample.csv'
        if not source.is_file():
            self.skipTest('prepare bundles to run the archived-data regression')
        with source.open(newline='') as handle:
            reader = csv.DictReader(handle)
            fieldnames, rows = reader.fieldnames, list(reader)
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / 'duplicate.csv'
            with duplicate.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows + rows)
            report = verify_main_results.verify_table6(ROOT / 'provenance/PUBLISHED_VALUES.json',
                        ROOT / 'models/main_base_4layer_correct_lr_3/seed_training_summary.csv', duplicate)
            self.assertFalse(report['passed'])


if __name__ == '__main__':
    unittest.main()
