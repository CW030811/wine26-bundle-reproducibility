import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_table2_exact.py"


class RunTable2ExactDriverTest(unittest.TestCase):
    def test_builds_all_final_table2_method_commands(self):
        self.assertTrue(SCRIPT.exists(), "Table 2 top-level driver must exist")
        spec = importlib.util.spec_from_file_location("run_table2_exact", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            commands = module.build_commands(
                root=root,
                python=Path("/opt/repro/python"),
                methods=["FCP", "PCP", "FCPLS", "BSP"],
                seeds="1,2,3,4,5,6,7,8,9,10",
            )

        self.assertEqual([item["method"] for item in commands], ["FCP", "PCP", "FCPLS", "BSP"])
        for item in commands:
            command = item["command"]
            self.assertEqual(command[0], "/opt/repro/python")
            self.assertIn(
                "data/deterministic/test_m10n10_correct_1e_3;"
                "data/deterministic/test_m20n10_correct_1e_3;"
                "data/deterministic/test_m30n10_correct_1e_3",
                command,
            )

        self.assertIn("test_FCP_multi_model_avg.py", commands[0]["command"][1])
        self.assertIn("test_PCP_cp_multi_model_avg.py", commands[1]["command"][1])
        self.assertIn("test_FCPLS_score_cached_lp.py", commands[2]["command"][1])
        self.assertIn("--no-plot", commands[2]["command"])
        self.assertIn("test_BSP.py", commands[3]["command"][1])
        self.assertNotIn("--seeds", commands[3]["command"])


if __name__ == "__main__":
    unittest.main()
