from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_generation import NumericComparison


class GenerationComparisonTests(unittest.TestCase):
    def test_float64_roundoff_is_reported_and_bounded(self):
        comparison = NumericComparison()
        self.assertTrue(comparison.equal(np.array([1.0]), np.array([1.0 + 1e-13])))
        self.assertGreater(comparison.max_absolute_difference, 0)
        self.assertEqual(comparison.numeric_leaves_different, 1)
        self.assertFalse(comparison.equal(np.array([1.0]), np.array([1.0 + 1e-6])))
        # Observed x86 Linux versus macOS CDF/erfinv tail round-trip difference.
        self.assertTrue(comparison.equal(np.array([0.0]), np.array([7.962519532611623e-12])))

    def test_structure_integer_identities_and_nonfinite_values_remain_strict(self):
        comparison = NumericComparison()
        self.assertFalse(comparison.equal(np.array([1.0]), np.array([[1.0]])))
        self.assertFalse(comparison.equal({'seed': 1000}, {'seed': 1001}))
        self.assertFalse(comparison.equal(np.array([float('nan')]), np.array([float('nan')])))
        self.assertFalse(comparison.equal({'a': 1}, {'b': 1}))


if __name__ == '__main__':
    unittest.main()
