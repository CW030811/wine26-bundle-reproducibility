import csv
import sys
import tempfile
import unittest
from pathlib import Path


DETERMINISTIC_SRC = Path(__file__).resolve().parents[1] / "src" / "deterministic"
sys.path.insert(0, str(DETERMINISTIC_SRC))

import test_FCP_multi_model_avg as evaluator
import test_FCP_multi_model_avg_f33 as f33_evaluator
import test_FCP_multi_model_avg_log as log_evaluator
import test_PCP_cp_multi_model_avg as pcp_evaluator


class SeedSampleOutputTest(unittest.TestCase):
    def test_writes_one_auditable_row_per_seed_and_sample(self):
        rows = [
            {
                "method": "FCP",
                "layers": 4,
                "seed": 1,
                "sample_file": "sample_data_1.msgpack",
                "m_segments": 10,
                "n_products": 10,
                "revenue_ratio": 0.99,
                "runtime_ratio": 0.01,
                "total_time": 0.02,
                "solve_time": 0.019,
                "gcn_time": 0.001,
            },
            {
                "method": "FCP",
                "layers": 4,
                "seed": 2,
                "sample_file": "sample_data_1.msgpack",
                "m_segments": 10,
                "n_products": 10,
                "revenue_ratio": 0.98,
                "runtime_ratio": 0.02,
                "total_time": 0.03,
                "solve_time": 0.029,
                "gcn_time": 0.001,
            },
        ]

        for module in (evaluator, log_evaluator, f33_evaluator, pcp_evaluator):
            with self.subTest(module=module.__name__):
                self.assertTrue(
                    hasattr(module, "_save_seed_sample_results"),
                    f"{module.__name__} must expose the seed-by-sample CSV writer",
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = module._save_seed_sample_results(
                        rows,
                        tmpdir,
                        "test_m10n10_correct_1e_3",
                    )
                    with open(path, newline="", encoding="utf-8") as handle:
                        saved = list(csv.DictReader(handle))

                self.assertEqual(len(saved), 2)
                self.assertEqual(saved[0]["sample_file"], "sample_data_1.msgpack")
                self.assertEqual(saved[0]["seed"], "1")
                self.assertEqual(saved[1]["seed"], "2")


if __name__ == "__main__":
    unittest.main()
