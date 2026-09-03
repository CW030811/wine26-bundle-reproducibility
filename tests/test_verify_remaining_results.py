import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_remaining_results.py"
PUBLISHED = ROOT / "provenance" / "PUBLISHED_VALUES.json"


class VerifyRemainingResultsTest(unittest.TestCase):
    def _load_module(self):
        self.assertTrue(SCRIPT.exists(), "remaining-result verifier must exist")
        spec = importlib.util.spec_from_file_location("verify_remaining_results", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_published_manifest_contains_figure_band_statistics(self):
        published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
        self.assertEqual(published["figure7"]["FCP_profit_std"], [0.0404, 0.0387, 0.0474, 0.0489])
        self.assertEqual(published["figure7"]["PCP_profit_std"], [0.0419, 0.0367, 0.0367, 0.0346])
        self.assertEqual(published["figure8"]["FCP_I_profit_std"], [0.0403, 0.0356, 0.0361, 0.0335])
        self.assertEqual(published["figure8"]["PCP_I_profit_std"], [0.0405, 0.0370, 0.0371, 0.0362])

    def test_aggregates_only_samples_with_every_required_seed(self):
        module = self._load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "seed_sample.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["seed", "sample_file", "revenue_ratio"],
                )
                writer.writeheader()
                writer.writerow({"seed": 1, "sample_file": "a.msgpack", "revenue_ratio": 0.5})
                writer.writerow({"seed": 2, "sample_file": "a.msgpack", "revenue_ratio": 1.0})
                writer.writerow({"seed": 1, "sample_file": "partial.msgpack", "revenue_ratio": 0.9})

            summary = module.aggregate_complete_seed_samples(
                csv_path,
                "revenue_ratio",
                required_seeds=2,
            )

        self.assertEqual(summary["raw_rows"], 3)
        self.assertEqual(summary["complete_values"], {"a.msgpack": 0.75})
        self.assertEqual(summary["partial_samples"], {"partial.msgpack": [1]})

    def test_finds_unique_leave_one_out_subset_for_figure6(self):
        module = self._load_module()
        selection = module.select_matching_samples(
            {"a": 0.5, "b": 1.0, "extra": 0.1},
            expected_count=2,
            expected_mean=0.75,
            expected_std=0.25,
            decimals=3,
        )

        self.assertTrue(selection["matched"])
        self.assertEqual(selection["selected_samples"], ["a", "b"])
        self.assertEqual(selection["excluded_samples"], ["extra"])

    def test_bounded_rounding_tolerance_does_not_hide_material_difference(self):
        module = self._load_module()
        self.assertTrue(module.matches_paper_display(1.161349, 1.1614, decimals=4))
        self.assertFalse(module.matches_paper_display(1.1608, 1.1614, decimals=4))


if __name__ == "__main__":
    unittest.main()
