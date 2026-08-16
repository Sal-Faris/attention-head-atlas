import unittest

import numpy as np

from head_atlas.diagnostics import operator_record, operator_table
from head_atlas.operators import HeadOperator


class DiagnosticTests(unittest.TestCase):
    def test_operator_record_matches_known_spectrum(self):
        operator = HeadOperator(2, 3, "OV", np.diag([4.0, 3.0, 0.0, 0.0]))
        record = operator_record(operator, energy_cutoffs=(1, 2, 4))

        self.assertEqual(record["layer"], 2)
        self.assertEqual(record["head"], 3)
        self.assertEqual(record["kind"], "OV")
        self.assertAlmostEqual(record["frobenius_norm"], 5.0)
        self.assertAlmostEqual(record["spectral_norm"], 4.0)
        self.assertAlmostEqual(record["top_1_energy"], 16.0 / 25.0)
        self.assertAlmostEqual(record["top_2_energy"], 1.0)
        self.assertAlmostEqual(record["top_4_energy"], 1.0)

    def test_operator_table_has_one_row_per_operator(self):
        operators = [
            HeadOperator(0, 0, "OV", np.eye(2)),
            HeadOperator(0, 1, "OV", np.diag([2.0, 1.0])),
        ]
        self.assertEqual(len(operator_table(operators)), 2)


if __name__ == "__main__":
    unittest.main()
