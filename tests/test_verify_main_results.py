import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_main_results.py"


class VerifyMainResultsTest(unittest.TestCase):
    def test_paper_display_tolerance_accepts_only_near_rounding_boundary_noise(self):
        self.assertTrue(SCRIPT.exists(), "main-result verifier must exist")
        spec = importlib.util.spec_from_file_location("verify_main_results_tolerance", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertTrue(
            hasattr(module, "_within_paper_display_tolerance"),
            "verifier must expose its bounded display-tolerance rule",
        )
        self.assertTrue(module._within_paper_display_tolerance(0.994485872, 0.995))
        self.assertTrue(module._within_paper_display_tolerance(0.863495926, 0.864))
        self.assertFalse(module._within_paper_display_tolerance(0.9939, 0.995))

    def test_table6_ignores_runtime_and_checks_paper_rounding(self):
        self.assertTrue(SCRIPT.exists(), "main-result verifier must exist")
        spec = importlib.util.spec_from_file_location("verify_main_results", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            published = root / "published.json"
            summary = root / "training_summary.csv"
            results = root / "seed_sample.csv"

            published.write_text(
                json.dumps({"table6": {"1": [0.1, 0.2, 0.75, 0.25, 99.0, 88.0]}}),
                encoding="utf-8",
            )
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["seed", "final_train_loss", "best_val_loss"],
                )
                writer.writeheader()
                writer.writerow(
                    {"seed": 1, "final_train_loss": 0.10001, "best_val_loss": 0.20001}
                )
            with results.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["seed", "revenue_ratio", "total_time"],
                )
                writer.writeheader()
                writer.writerow({"seed": 1, "revenue_ratio": 0.5, "total_time": 123.0})
                writer.writerow({"seed": 1, "revenue_ratio": 1.0, "total_time": 456.0})

            report = module.verify_table6(published, summary, results)

        self.assertTrue(report["passed"])
        self.assertEqual(report["checked_seeds"], 1)
        self.assertEqual(report["runtime_fields_checked"], 0)

    def test_table7_checks_each_ood_profit_mean_and_std(self):
        self.assertTrue(SCRIPT.exists(), "main-result verifier must exist")
        spec = importlib.util.spec_from_file_location("verify_main_results_table7", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            published = root / "published.json"
            published.write_text(
                json.dumps(
                    {
                        "table7": {
                            "log1p": [0.75, 0.25, 99.0, 88.0],
                            "cube_root": [0.75, 0.25, 99.0, 88.0],
                            "beta_5_5": [0.75, 0.25, 99.0, 88.0],
                            "beta_half_half": [0.991, 0.008, 99.0, 88.0],
                        }
                    }
                ),
                encoding="utf-8",
            )
            result_paths = {}
            for key in ("log1p", "cube_root", "beta_5_5", "beta_half_half"):
                path = root / f"{key}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["revenue_ratio", "total_time"])
                    writer.writeheader()
                    if key == "beta_half_half":
                        for index in range(100):
                            offset = 0.00749 if index % 2 == 0 else -0.00749
                            writer.writerow(
                                {"revenue_ratio": 0.991 + offset, "total_time": 123.0}
                            )
                    else:
                        writer.writerow({"revenue_ratio": 0.5, "total_time": 123.0})
                        writer.writerow({"revenue_ratio": 1.0, "total_time": 456.0})
                result_paths[key] = path

            self.assertTrue(
                hasattr(module, "verify_table7"),
                "main-result verifier must expose Table 7 verification",
            )
            report = module.verify_table7(published, result_paths)

        self.assertTrue(report["passed"])
        self.assertEqual(report["checked_variants"], 4)
        self.assertEqual(report["runtime_fields_checked"], 0)

    def test_table2_checks_profit_only_after_averaging_seeds_per_sample(self):
        self.assertTrue(SCRIPT.exists(), "main-result verifier must exist")
        spec = importlib.util.spec_from_file_location("verify_main_results_table2", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            published = root / "published.json"
            published.write_text(
                json.dumps(
                    {
                        "table2": {
                            "m10": {
                                method: [0.75, 0.25, 999.0, 888.0]
                                for method in ("FCP", "PCP", "FCPLS", "BSP")
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            result_paths = {}
            for method in ("FCP", "PCP", "FCPLS"):
                path = root / f"{method}.csv"
                value_column = "competitive_ratio" if method == "FCPLS" else "revenue_ratio"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["seed", "sample_file", value_column, "total_time"],
                    )
                    writer.writeheader()
                    for seed in (1, 2):
                        writer.writerow(
                            {
                                "seed": seed,
                                "sample_file": "sample_1.msgpack",
                                value_column: 0.5,
                                "total_time": 123.0,
                            }
                        )
                        writer.writerow(
                            {
                                "seed": seed,
                                "sample_file": "sample_2.msgpack",
                                value_column: 1.0,
                                "total_time": 456.0,
                            }
                        )
                result_paths[("m10", method)] = path

            bsp_path = root / "BSP.csv"
            with bsp_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["filename", "revenue_ratio", "bsp_time"],
                )
                writer.writeheader()
                writer.writerow(
                    {"filename": "sample_1.msgpack", "revenue_ratio": 0.5, "bsp_time": 1.0}
                )
                writer.writerow(
                    {"filename": "sample_2.msgpack", "revenue_ratio": 1.0, "bsp_time": 2.0}
                )
            result_paths[("m10", "BSP")] = bsp_path

            self.assertTrue(
                hasattr(module, "verify_table2"),
                "main-result verifier must expose Table 2 verification",
            )
            report = module.verify_table2(
                published,
                result_paths,
                expected_samples=2,
                expected_seeds=2,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["checked_cells"], 4)
        self.assertEqual(report["runtime_fields_checked"], 0)
        fcp_detail = next(item for item in report["details"] if item["method"] == "FCP")
        self.assertEqual(fcp_detail["raw_rows"], 4)
        self.assertEqual(fcp_detail["aggregated_samples"], 2)


if __name__ == "__main__":
    unittest.main()
