import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_remaining_exact.py"


class RunRemainingExactDriverTest(unittest.TestCase):
    def _load_module(self):
        self.assertTrue(SCRIPT.exists(), "remaining-experiment top-level driver must exist")
        spec = importlib.util.spec_from_file_location("run_remaining_exact", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_builds_deduplicated_figure_and_table_batches(self):
        module = self._load_module()

        figure_tasks = module.build_tasks("figures")
        table_tasks = module.build_tasks("tables")
        all_tasks = module.build_tasks("all")

        self.assertEqual(len(figure_tasks), 208)
        self.assertEqual(len(table_tasks), 320)
        self.assertEqual(len(all_tasks), 458)
        self.assertEqual(len({task.task_id for task in all_tasks}), len(all_tasks))

    def test_uses_final_models_and_preserves_figure6_m80_success_rule(self):
        module = self._load_module()
        tasks = module.build_tasks("figures")

        improved = [task for task in tasks if task.model_tag == "self_improved"]
        self.assertTrue(improved)
        self.assertTrue(
            all(task.model_dir == "models/self_improved_m10n50_correct_lr_3" for task in improved)
        )

        m80_fcp = [
            task
            for task in tasks
            if task.method == "FCP"
            and task.model_tag == "base"
            and task.dataset == "test_BSP_m80n20_correct_1e_3"
        ]
        self.assertEqual(len(m80_fcp), 10)
        self.assertTrue(all(task.expected_samples == 30 for task in m80_fcp))
        self.assertTrue(all(task.minimum_successes == 29 for task in m80_fcp))

    def test_table_batch_omits_unreported_pcp_cells(self):
        module = self._load_module()
        tasks = module.build_tasks("tables")
        pcp_datasets = {
            task.dataset
            for task in tasks
            if task.method == "PCP" and task.model_tag == "base"
        }
        improved_pcp_datasets = {
            task.dataset
            for task in tasks
            if task.method == "PCP" and task.model_tag == "self_improved"
        }

        self.assertNotIn("test_BSP_m20n40_correct_1e_3", pcp_datasets)
        self.assertEqual(
            improved_pcp_datasets,
            {
                "test_BSP_m10n60_correct_1e_3",
                "test_BSP_m10n80_correct_1e_3",
                "test_BSP_m10n100_correct_1e_3",
            },
        )


if __name__ == "__main__":
    unittest.main()
