import tempfile
import unittest
from pathlib import Path

import numpy as np

from head_atlas.model_io import load_operator_bundle, save_operator_bundle
from head_atlas.operators import HeadOperator


class ModelIoTests(unittest.TestCase):
    def test_operator_bundle_round_trip(self):
        operators = [
            HeadOperator(0, 0, "OV", np.eye(3)),
            HeadOperator(0, 1, "OV", np.diag([1.0, 2.0, 3.0])),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "operators.npz"
            save_operator_bundle(path, operators, {"model": "synthetic"})
            loaded, metadata = load_operator_bundle(path)
        self.assertEqual(metadata["model"], "synthetic")
        self.assertEqual([(x.layer, x.head, x.kind) for x in loaded], [(0, 0, "OV"), (0, 1, "OV")])
        np.testing.assert_array_equal(loaded[1].matrix, operators[1].matrix)


if __name__ == "__main__":
    unittest.main()

