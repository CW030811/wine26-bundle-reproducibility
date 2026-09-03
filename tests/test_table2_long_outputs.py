import csv
import sys
import tempfile
import unittest
from pathlib import Path


DETERMINISTIC_SRC = Path(__file__).resolve().parents[1] / "src" / "deterministic"
sys.path.insert(0, str(DETERMINISTIC_SRC))

import test_BSP as bsp_evaluator
import test_FCPLS_score_cached_lp as fcpls_evaluator


class Table2LongOutputTest(unittest.TestCase):
    def test_fcpls_writes_one_auditable_row_per_seed_and_sample(self):
        rows = [
            {
                "method": "FCPLS",
                "layers": 4,
                "seed": 1,
                "sample_file": "sample_data_1.msgpack",
                "m_segments": 10,
                "n_products": 10,
                "competitive_ratio": 0.99,
                "runtime_ratio": 0.01,
                "total_time": 0.02,
                "gcn_time": 0.001,
                "initial_milp_time": 0.004,
                "local_search_time": 0.015,
                "gnn_gurobi_time": 0.018,
            },
            {
                "method": "FCPLS",
                "layers": 4,
                "seed": 2,
                "sample_file": "sample_data_1.msgpack",
                "m_segments": 10,
                "n_products": 10,
                "competitive_ratio": 0.98,
                "runtime_ratio": 0.02,
                "total_time": 0.03,
                "gcn_time": 0.001,
                "initial_milp_time": 0.005,
                "local_search_time": 0.024,
                "gnn_gurobi_time": 0.028,
            },
        ]

        self.assertTrue(
            hasattr(fcpls_evaluator, "_save_seed_sample_results"),
            "FCPLS evaluator must expose the seed-by-sample CSV writer",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = fcpls_evaluator._save_seed_sample_results(
                rows,
                tmpdir,
                "test_m10n10_correct_1e_3",
            )
            with open(path, newline="", encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))

        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0]["sample_file"], "sample_data_1.msgpack")
        self.assertEqual(saved[0]["competitive_ratio"], "0.99")
        self.assertEqual(saved[1]["seed"], "2")

    def test_bsp_writes_one_auditable_row_per_sample(self):
        rows = [
            {
                "filename": "sample_data_1.msgpack",
                "product_num": 10,
                "segment_num": 10,
                "revenue_ratio": 0.88,
                "time_ratio": 0.1,
                "bsp_time": 0.2,
                "bsp_rev": 8.8,
                "opt_rev": 10.0,
                "opt_time": 2.0,
            }
        ]

        self.assertTrue(
            hasattr(bsp_evaluator, "_save_sample_results"),
            "BSP evaluator must expose the sample-level CSV writer",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = bsp_evaluator._save_sample_results(
                rows,
                tmpdir,
                "test_m10n10_correct_1e_3",
            )
            with open(path, newline="", encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["filename"], "sample_data_1.msgpack")
        self.assertEqual(saved[0]["revenue_ratio"], "0.88")


if __name__ == "__main__":
    unittest.main()
